/**
 * @file job_info_update.cpp
 * @brief Alpha-j counter management for the Stochastic Online Scheduler.
 *
 * AVX-512 / AVX2 vectorisation (Intel Xeon 4th-gen "Sapphire Rapids")
 * =====================================================================
 *
 * Original hot-spots and their SIMD replacements
 * -----------------------------------------------
 *
 * 1. stack_handler_jiu()
 *    Pointer-chasing with loop-carried dependencies — kept scalar.
 *    Called at most once per machine per tick.
 *
 * 2. alpha_j_loop  (JOBS_PER_MACHINE = 10 iterations inside
 *    alpha_j_update_machine())
 *
 *    Scalar loop body per slot i:
 *      if top_job_id != INVALID && alpha_j_cam[i].job_id == top_job_id:
 *          alpha_j_cam[i].alpha_j -= 1
 *          if alpha_j_cam[i].alpha_j == 0:
 *              invalidate slot i, push index to stack  (pop = 1)
 *          else:
 *              pop = 0
 *
 *    SIMD replacement:
 *      a) Unpack alpha_j_cam into two flat CACHE_ALIGNED uint32_t arrays:
 *           job_ids[JPM_PAD]   and   alpha_js[JPM_PAD].
 *      b) Load job_ids[0..15] into a ZMM (10-bit mask = JOBS_LANE_MASK).
 *      c) Broadcast top_job_id → compare → __mmask16 hit_mask (0 or 1 bit).
 *      d) Masked decrement: alpha_js[hit] -= 1  via _mm512_mask_sub_epi32.
 *      e) Compare decremented alpha_js[hit] == 0 → zero_mask (0 or 1 bit).
 *      f) If zero_mask:
 *           - Extract hit index with __builtin_ctz.
 *           - Zero job_ids[hit] and alpha_js[hit] in place (scalar).
 *           - Call stack_handler_jiu PUSH (scalar, rare path).
 *           - Set *pop = 1, output->operation = JI_INVALIDATE.
 *         Else if hit_mask:
 *           - Set *pop = 0, output->operation = JI_UPDATE.
 *      g) Pack modified flat arrays back into alpha_j_cam structs.
 *
 *    New-job installation (after alpha_j_loop):
 *      - stack_handler_jiu POP (scalar) → address.
 *      - Write new_job_id and new alpha_j at that address (scalar,
 *        one write per tick — not worth vectorising).
 *
 * 3. jiu_all_machine_loop  (NUM_MACHINES = 5 iterations in job_info_update())
 *    Loop-carried dependency on static alpha_j_cam state per machine.
 *    Machines are processed sequentially; inner 10-slot work is vectorised.
 *    Software prefetch issued for the next machine's alpha_j_cam.
 *
 * Data layout requirements (data_types.hpp):
 *   alpha_j_info_t : alignas(8) { job_id(4B uint32), alpha_j(1B uint8),
 *                                  _pad[3] }
 *   job_info_update_input_t  : alignas(8) { new_job_id(4B), alpha_j(1B),
 *                                            _pad[3] }
 *   job_info_update_output_t : alignas(8) { job_id(4B), operation(1B),
 *                                            _pad[3] }
 */

#include "data_types.hpp"
#include "top_modules.hpp"
#include <cstdio>

/* ── Scratch-buffer width: JOBS_PER_MACHINE padded to 16 for ZMM stores ─── */
#define JPM_PAD 16

/* ═══════════════════════════════════════════════════════════════════════════
 * 1.  stack_handler_jiu  (scalar — unchanged)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Implements the free-address stack for alpha_j_cam slot management.
 * Control-flow dominated; kept scalar.  Called at most once per machine
 * per tick (on PUSH from invalidation or POP for new-job installation).
 */
static memory_length_t stack_handler_jiu(memory_length_t *stack,
                                          one_bit_t       *reset,
                                          memory_length_t *head,
                                          memory_length_t *tail,
                                          one_bit_t        operation,
                                          memory_length_t  push_address)
{
    memory_length_t new_job_address = INVALID_ADDRESS;

    /* Active-low reset */
    if (*reset == 0) {
        *head = 0;
        *tail = JOBS_PER_MACHINE;
        for (memory_length_t i = 0; i < JOBS_PER_MACHINE; i++)
            stack[i] = i;
        *reset = 1;
    }

    if (operation == PUSH) {
        if (*tail == JOBS_PER_MACHINE) {
            if (*head == 0) {
                new_job_address = INVALID_ADDRESS;              /* full */
            } else {
                stack[*tail]    = push_address;
                *tail           = 0;
                new_job_address = stack[*head];
            }
        } else {
            if ((*tail + 1) == *head) {
                new_job_address = INVALID_ADDRESS;              /* full */
            } else {
                stack[*tail]    = push_address;
                (*tail)        += 1;
                new_job_address = stack[*head];
            }
        }
    } else {                                                    /* POP */
        if (*head == *tail) {
            new_job_address = INVALID_ADDRESS;                  /* empty */
        } else {
            new_job_address = stack[*head];
            *head = (*head == JOBS_PER_MACHINE) ? 0 : *head + 1;
        }
    }
    return new_job_address;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2.  Unpack / pack helpers for alpha_j_cam
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * unpack_alpha_j_cam()
 * Explodes alpha_j_cam[JOBS_PER_MACHINE] into two flat uint32_t arrays.
 * Upper JPM_PAD - JOBS_PER_MACHINE slots are zero-padded so ZMM loads
 * never read uninitialised memory.
 *
 * Separation of job_ids and alpha_js into distinct arrays (struct-of-arrays
 * layout) lets the compiler / CPU issue two independent ZMM loads and keeps
 * the compare and arithmetic pipelines fed without port contention.
 */
SIMD_TARGET_AVX512
static inline void unpack_alpha_j_cam(
        const alpha_j_info_t * __restrict__ cam,
        uint32_t             * __restrict__ job_ids,   /* [JPM_PAD] */
        uint32_t             * __restrict__ alpha_js)  /* [JPM_PAD] */
{
    for (int j = 0; j < JOBS_PER_MACHINE; j++) {
        job_ids [j] = cam[j].job_id;
        alpha_js[j] = (uint32_t)cam[j].alpha_j;
    }
    /* Zero-pad to 16 for safe full-width ZMM store-back */
    for (int j = JOBS_PER_MACHINE; j < JPM_PAD; j++) {
        job_ids [j] = INVALID_JOB_ID;
        alpha_js[j] = 0;
    }
}

/**
 * pack_alpha_j_cam()
 * Writes modified flat arrays back into the struct array.
 * Only the JOBS_PER_MACHINE valid slots are touched.
 */
SIMD_TARGET_AVX512
static inline void pack_alpha_j_cam(
        alpha_j_info_t       * __restrict__ cam,
        const uint32_t       * __restrict__ job_ids,
        const uint32_t       * __restrict__ alpha_js)
{
    for (int j = 0; j < JOBS_PER_MACHINE; j++) {
        cam[j].job_id  = job_ids [j];
        cam[j].alpha_j = (uint8_t)alpha_js[j];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3.  alpha_j_update_machine_simd()
 *     Vectorised replacement for alpha_j_update_machine()
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * alpha_j_update_machine_simd()
 *
 * For one machine, updates the alpha_j CAM and determines the pop signal:
 *
 *  Step 1 — Unpack alpha_j_cam into flat SIMD buffers.
 *
 *  Step 2 — Top-job alpha_j decrement (AVX-512):
 *    a. Load job_ids[0..15] into ZMM_ids  (valid_mask = 0x3FF).
 *    b. Broadcast top_job_id → ZMM_target.
 *    c. VPCMPEQD with valid_mask  → hit_mask (__mmask16, 0 or 1 bit set).
 *    d. If hit_mask != 0:
 *         i.  Load alpha_js[0..15] into ZMM_aj.
 *         ii. Masked subtract 1: ZMM_aj[hit] -= 1.
 *         iii.Compare ZMM_aj[hit] == 0 → zero_mask.
 *         iv. If zero_mask:
 *               - hit_idx = __builtin_ctz(hit_mask).
 *               - job_ids[hit_idx]  = INVALID_JOB_ID (scalar).
 *               - alpha_js[hit_idx] = 0              (scalar, already 0).
 *               - PUSH hit_idx to stack (scalar).
 *               - *pop = 1;  output->operation = JI_INVALIDATE.
 *             Else:
 *               - *pop = 0;  output->operation = JI_UPDATE.
 *         v.  Store ZMM_aj back with valid_mask (only modified slot changed).
 *         vi. output->job_id = top_job_id.
 *
 *  Step 3 — New-job installation (scalar, at most once per tick):
 *    a. POP free address from stack.
 *    b. Write job_id and alpha_j at that address in flat buffers.
 *
 *  Step 4 — Pack flat buffers back into alpha_j_cam structs.
 *
 * Notes:
 *  - hit_mask has AT MOST ONE bit set (one job_id matches at a time).
 *    The ZMM decrement is therefore a 1-lane masked operation — the
 *    vectorisation benefit comes from the compare (avoids 10 scalar
 *    comparisons) and from keeping the data in SIMD registers ready for
 *    the subsequent zero-check without an extra load.
 *  - The pack/unpack cost (10 scalar reads/writes each) is amortised
 *    across the 5-machine outer loop; prefetching hides the latency.
 */
SIMD_TARGET_AVX512
static void alpha_j_update_machine_simd(
        job_info_update_input_t   input,
        job_id_t                  top_job_id,
        alpha_j_info_t           *alpha_j_cam,
        memory_length_t          *job_address_stack,
        one_bit_t                *reset,
        memory_length_t          *head_pointer,
        memory_length_t          *tail_pointer,
        job_info_update_output_t *output,
        one_bit_t                *pop)
{
    /* ── Flat scratch buffers (stack-allocated, aligned for ZMM) ──────── */
    CACHE_ALIGNED uint32_t job_ids [JPM_PAD];
    CACHE_ALIGNED uint32_t alpha_js[JPM_PAD];

    /* ── Step 1: unpack ──────────────────────────────────────────────── */
    unpack_alpha_j_cam(alpha_j_cam, job_ids, alpha_js);

    /* Default pop state: unchanged (caller initialises to 0 before the
     * machine loop in job_info_update()) */
    *pop = 0;

    /* ── Step 2: top-job alpha_j decrement ───────────────────────────── */
    if (top_job_id != INVALID_JOB_ID) {

        const __mmask16 valid_mask = (__mmask16)JOBS_LANE_MASK; /* 0x3FF */

        /* 2a. Load job_ids into ZMM */
        __m512i zmm_ids    = _mm512_maskz_loadu_epi32(valid_mask, job_ids);

        /* 2b. Broadcast the top-job ID we are looking for */
        __m512i zmm_target = _mm512_set1_epi32((int)top_job_id);

        /* 2c. Compare: hit_mask[i] = 1 iff job_ids[i] == top_job_id */
        __mmask16 hit_mask = _mm512_mask_cmpeq_epi32_mask(valid_mask,
                                                            zmm_ids,
                                                            zmm_target);

        if (hit_mask) {
            /* 2d-i. Load alpha_js into ZMM */
            __m512i zmm_aj  = _mm512_maskz_loadu_epi32(valid_mask, alpha_js);

            /* 2d-ii. Masked decrement: only the matching slot decremented */
            __m512i zmm_one = _mm512_set1_epi32(1);
            zmm_aj = _mm512_mask_sub_epi32(zmm_aj, hit_mask, zmm_aj, zmm_one);

            /* 2d-iii. Check if the decremented value reached zero */
            __m512i  zmm_zero  = _mm512_setzero_si512();
            __mmask16 zero_mask = _mm512_mask_cmpeq_epi32_mask(hit_mask,
                                                                 zmm_aj,
                                                                 zmm_zero);

            /* 2d-v. Store updated alpha_js back (only hit slot changed) */
            _mm512_mask_storeu_epi32(alpha_js, valid_mask, zmm_aj);

            /* 2d-vi. Record which job triggered this update */
            output->job_id = top_job_id;

            if (zero_mask) {
                /* ── JI_INVALIDATE path ──────────────────────────────── */
                /* Extract the single set bit position */
                int hit_idx = __builtin_ctz((unsigned)hit_mask);

                /* Invalidate the slot in the flat buffers */
                job_ids [hit_idx] = INVALID_JOB_ID;
                alpha_js[hit_idx] = 0;

                /* Push the freed address back onto the stack */
                stack_handler_jiu(job_address_stack, reset,
                                   head_pointer, tail_pointer,
                                   PUSH, (memory_length_t)hit_idx);

                *pop               = 1;
                output->operation  = JI_INVALIDATE;

#if JIU_DEBUG
                printf("\033[34m[JIU] Invalidate job_id=%u at slot=%d\033[0m\n",
                       top_job_id, hit_idx);
#endif
            } else {
                /* ── JI_UPDATE path ──────────────────────────────────── */
                *pop              = 0;
                output->operation = JI_UPDATE;

#if JIU_DEBUG
                /* Extract the updated alpha_j for display */
                CACHE_ALIGNED uint32_t tmp[JPM_PAD];
                _mm512_mask_storeu_epi32(tmp, valid_mask, zmm_aj);
                int hit_idx = __builtin_ctz((unsigned)hit_mask);
                printf("\033[34m[JIU] Update job_id=%u slot=%d "
                       "alpha_j=%u\033[0m\n",
                       top_job_id, hit_idx, tmp[hit_idx]);
#endif
            }
        }
        /* If hit_mask == 0: top_job_id not found in this machine's CAM.
         * pop and output remain at their defaults. */
    }

    /* ── Step 3: new-job installation (scalar) ───────────────────────── */
    if (input.new_job_id != INVALID_JOB_ID) {
        memory_length_t addr = stack_handler_jiu(job_address_stack, reset,
                                                  head_pointer, tail_pointer,
                                                  POP,
                                                  /* push_addr ignored on POP */
                                                  INVALID_ADDRESS);
        if (addr != INVALID_ADDRESS) {
            job_ids [addr] = input.new_job_id;
            alpha_js[addr] = (uint32_t)input.alpha_j;

#if JIU_DEBUG
            printf("\033[34m[JIU] Install new job_id=%u alpha_j=%u "
                   "at slot=%u\033[0m\n",
                   input.new_job_id, input.alpha_j, addr);
#endif
        }
    }

    /* ── Step 4: pack flat buffers back into struct array ────────────── */
    pack_alpha_j_cam(alpha_j_cam, job_ids, alpha_js);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4.  job_info_update()  — top-level entry point (signature unchanged)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * job_info_update()
 *
 * Iterates over NUM_MACHINES = 5 machines, calling
 * alpha_j_update_machine_simd() for each.
 *
 * SIMD opportunities at this level:
 *   - Software prefetch for the next machine's alpha_j_cam issued at
 *     the top of each iteration to hide unpack latency.
 *   - pop[] and output[] arrays are written scalarly (5 elements each,
 *     no benefit from SIMD at this granularity).
 *
 * Static state (alpha_j_cam, job_address_stack, reset, head, tail) is
 * CACHE_ALIGNED to prevent false sharing across the 5 machine entries
 * and to enable aligned ZMM loads in the inner function.
 */
SIMD_TARGET_AVX512
void job_info_update(job_info_update_input_t  *input,
                     job_id_t                 *top_job_id,
                     job_info_update_output_t *output,
                     one_bit_t                *pop)
{
    /* ── Persistent per-machine state ───────────────────────────────── */
    static CACHE_ALIGNED alpha_j_info_t   alpha_j_cam      [NUM_MACHINES][JOBS_PER_MACHINE];
    static CACHE_ALIGNED memory_length_t  job_address_stack [NUM_MACHINES][JOBS_PER_MACHINE + 1];
    static one_bit_t                      reset             [NUM_MACHINES];
    static memory_length_t                head_pointer      [NUM_MACHINES];
    static memory_length_t                tail_pointer      [NUM_MACHINES];

    /* ── Per-tick initialisation of output and pop arrays ───────────── */
    /*
     * Zero output and pop for all machines before the loop.
     * Using a single _mm_storeu_si128 for the 5-byte pop array and a
     * small scalar loop for the 5 × 8-byte output array avoids the
     * conditional initialisation that was implicit in the scalar version.
     */
    {
        /* Zero pop[0..4] — 5 bytes: use a 32-bit + 8-bit store */
        *(uint32_t *)pop      = 0u;
        pop[4]                = 0u;

        /* Zero output[0..4] — 5 × 8 bytes = 40 bytes */
        const __m256i vzero = _mm256_setzero_si256();
        _mm256_storeu_si256((__m256i *)output,       vzero);  /* [0..3] */
        *(uint64_t *)(output + 4) = 0ULL;                     /* [4]    */
    }

    /* ── Per-machine update loop ────────────────────────────────────── */
    for (machine_id_t m = 0; m < NUM_MACHINES; m++) {

        /* Software prefetch: bring next machine's CAM into L1 while
         * this machine's ZMM work is in-flight (stride ≈ 80 bytes). */
        if (m + 1 < NUM_MACHINES) {
            _mm_prefetch((const char *)alpha_j_cam[m + 1], _MM_HINT_T0);
        }

#if JIU_DEBUG
        printf("\033[34m[JIU] Machine %u top_job=%u\033[0m\n",
               m, top_job_id[m]);
#endif

        alpha_j_update_machine_simd(
            input[m],
            top_job_id[m],
            alpha_j_cam[m],
            job_address_stack[m],
            &reset[m],
            &head_pointer[m],
            &tail_pointer[m],
            &output[m],
            &pop[m]);
    }
}
