/**
 * @file cost_calculator.cpp
 * @brief WSPT-based cost calculator for the Stochastic Online Scheduler.
 *
 * AVX-512 / AVX2 vectorisation (Intel Xeon 4th-gen "Sapphire Rapids")
 */

#include "data_types.hpp"
#include "top_modules.hpp"
#include <cstdio>
#include <cstdlib>

/* ═══════════════════════════════════════════════════════════════════════════
 * Internal scratch-buffer sizes
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Round JOBS_PER_MACHINE up to next multiple of 16 for clean ZMM stores */
#define JPM_PAD  16   /* JOBS_PER_MACHINE=10 padded to 16 for ZMM */

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. stack_handler_cc  (scalar — unchanged)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * stack_handler_cc()
 * Implements the free-address stack used to track vacant slots in the
 * EPT and weight CAMs.  Control-flow dominated; kept scalar.
 */
static memory_length_t stack_handler_cc(memory_length_t *stack,
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
                new_job_address = INVALID_ADDRESS;          /* full */
            } else {
                stack[*tail]  = push_address;
                *tail         = 0;
                new_job_address = stack[*head];
            }
        } else {
            if ((*tail + 1) == *head) {
                new_job_address = INVALID_ADDRESS;          /* full */
            } else {
                stack[*tail]  = push_address;
                *tail        += 1;
                new_job_address = stack[*head];
            }
        }
    } else {                                                /* POP */
        if (*head == *tail) {
            new_job_address = INVALID_ADDRESS;              /* empty */
        } else {
            new_job_address = stack[*head];
            *head = (*head == JOBS_PER_MACHINE) ? 0 : *head + 1;
        }
    }
    return new_job_address;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Helper: unpack struct arrays into flat SIMD-friendly buffers
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * unpack_ept_cam()
 * Explodes ept_cam[JOBS_PER_MACHINE] into two flat uint32_t arrays so that
 * AVX-512 can operate on all 10 job_ids and proc_times without gather.
 *
 * Output arrays must be at least JPM_PAD (16) elements and SIMD_ALIGNED.
 */
SIMD_TARGET_AVX512
static inline void unpack_ept_cam(const proc_time_info_t *ept,
                                   uint32_t * __restrict__ job_ids,   /* [JPM_PAD] */
                                   uint32_t * __restrict__ proc_times)/* [JPM_PAD] */
{
    for (int j = 0; j < JOBS_PER_MACHINE; j++) {
        job_ids[j]   = ept[j].job_id;
        proc_times[j]= ept[j].proc_time;
    }
    /* Zero-pad to 16 so ZMM loads/stores are safe */
    for (int j = JOBS_PER_MACHINE; j < JPM_PAD; j++) {
        job_ids[j]   = INVALID_JOB_ID;
        proc_times[j]= 0;
    }
}

/**
 * unpack_weight_cam()
 * Explodes weight_cam[JOBS_PER_MACHINE] into three flat uint32_t arrays.
 */
SIMD_TARGET_AVX512
static inline void unpack_weight_cam(const weight_info_t *wt,
                                      uint32_t * __restrict__ job_ids, /* [JPM_PAD] */
                                      uint32_t * __restrict__ weights, /* [JPM_PAD] */
                                      uint32_t * __restrict__ wspts)   /* [JPM_PAD] */
{
    for (int j = 0; j < JOBS_PER_MACHINE; j++) {
        job_ids[j] = wt[j].job_id;
        weights[j] = wt[j].weight;
        wspts[j]   = wt[j].wspt;
    }
    for (int j = JOBS_PER_MACHINE; j < JPM_PAD; j++) {
        job_ids[j] = INVALID_JOB_ID;
        weights[j] = 0;
        wspts[j]   = 0;
    }
}

/**
 * pack_ept_cam() / pack_weight_cam()
 * Write modified flat arrays back into the struct arrays.
 * Only the JOBS_PER_MACHINE valid entries are written.
 */
SIMD_TARGET_AVX512
static inline void pack_ept_cam(proc_time_info_t       *ept,
                                 const uint32_t * __restrict__ job_ids,
                                 const uint32_t * __restrict__ proc_times)
{
    for (int j = 0; j < JOBS_PER_MACHINE; j++) {
        ept[j].job_id    = job_ids[j];
        ept[j].proc_time = (uint8_t)proc_times[j];
    }
}

SIMD_TARGET_AVX512
static inline void pack_weight_cam(weight_info_t          *wt,
                                    const uint32_t * __restrict__ job_ids,
                                    const uint32_t * __restrict__ weights,
                                    const uint32_t * __restrict__ wspts)
{
    for (int j = 0; j < JOBS_PER_MACHINE; j++) {
        wt[j].job_id = job_ids[j];
        wt[j].weight = (uint8_t)weights[j];
        wt[j].wspt   = (uint8_t)wspts[j];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. cost_calculator_machine_simd()
 * Vectorised replacement for cost_calculator_machine() +
 * cost_calculator_job() combined.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * cost_calculator_machine_simd()
 *
 * Processes all JOBS_PER_MACHINE=10 slots for one machine in two ZMM passes:
 *
 * Pass A — Job-info update (loop1 equivalent):
 * 1. Load ept job_ids[0..15] into ZMM (10-bit mask, upper 6 zeroed).
 * 2. Broadcast update_job_info.job_id → compare → hit_mask (__mmask16).
 * 3a. JI_UPDATE  (operation == JI_UPDATE):
 * - proc_time[hit] -= 1   (masked subtract)
 * - weight[hit]    -= wspt[hit]  (masked subtract)
 * 3b. JI_INVALIDATE (operation == JI_INVALIDATE):
 * - Zero proc_time, weight, wspt at hit index (scalar — only 1 hit max).
 * - Push hit index to free-address stack.
 *
 * Pass B — Cost selection (loop1 + loop2 equivalent):
 * 4. Load wspt[0..15] into ZMM.
 * 5. Broadcast wspt_new → compare wspt > wspt_new → high_mask.
 * 6. cost_jobs = blend32(weight, proc_time, high_mask).
 * 7. cost_high_wspt = _mm512_mask_reduce_add_epi32(high_mask, cost_jobs_zmm).
 * 8. cost_low_wspt  = _mm512_mask_reduce_add_epi32(~high_mask & JOBS_LANE_MASK, ...).
 * 9. new_job_index  = popcount(high_mask & JOBS_LANE_MASK).
 * 10. cost_machine   = weight_new * (ept_new + cost_high_wspt)
 * + ept_new * cost_low_wspt.
 */
SIMD_TARGET_AVX512
static void cost_calculator_machine_simd(
        job_id_t                  new_job_id,
        uint8_t                   new_ept_curr,
        uint8_t                   new_weight_curr,
        job_info_update_output_t  update_job_info,
        proc_time_info_t          new_ept_prev,
        weight_info_t             new_weight_prev,
        memory_length_t          *job_address_stack,
        one_bit_t                *reset,
        memory_length_t          *head_pointer,
        memory_length_t          *tail_pointer,
        proc_time_info_t         *ept_machine,
        weight_info_t            *weight_machine,
        uint32_t                 *cost_machine_out,
        vf_index_t               *new_job_index_out)
{
    /* ── Flat scratch buffers (stack-allocated, SIMD aligned) ─────────── */
    CACHE_ALIGNED uint32_t ept_job_ids  [JPM_PAD];
    CACHE_ALIGNED uint32_t proc_times   [JPM_PAD];
    CACHE_ALIGNED uint32_t wt_job_ids   [JPM_PAD];
    CACHE_ALIGNED uint32_t weights      [JPM_PAD];
    CACHE_ALIGNED uint32_t wspts        [JPM_PAD];

    /* ── Step 0: install new_ept_prev / new_weight_prev (scalar, rare) ── */
    if (new_ept_prev.job_id != INVALID_JOB_ID) {
        memory_length_t addr = stack_handler_cc(job_address_stack, reset,
                                                 head_pointer, tail_pointer,
                                                 POP, INVALID_ADDRESS);
        if (addr == INVALID_ADDRESS) {
            printf("Invalid Address hit!!! Error!!!, Breaking\n");
            exit(0);
        }
        ept_machine[addr].job_id    = new_ept_prev.job_id;
        ept_machine[addr].proc_time = new_ept_prev.proc_time;

        weight_machine[addr].job_id = new_weight_prev.job_id;
        weight_machine[addr].weight = new_weight_prev.weight;
        weight_machine[addr].wspt   = new_weight_prev.wspt;
    }

    /* ── Compute wspt for the incoming job ──────────────────────────── */
    uint8_t wspt_new = (new_job_id != INVALID_JOB_ID && new_ept_curr != 0)
                       ? (new_weight_curr / new_ept_curr)
                       : 0;

    /* ── Unpack struct arrays → flat uint32 arrays ──────────────────── */
    unpack_ept_cam   (ept_machine,    ept_job_ids, proc_times);
    unpack_weight_cam(weight_machine, wt_job_ids,  weights, wspts);

    /* ── 10-bit mask covering valid JOBS_PER_MACHINE slots ──────────── */
    const __mmask16 valid_mask = (__mmask16)JOBS_LANE_MASK;  /* 0x3FF */

    /* ══ Pass A: Job-info update ══════════════════════════════════════ */
    if (update_job_info.job_id != INVALID_JOB_ID) {

        /* Load ept job_ids into ZMM */
        __m512i v_ept_ids = _mm512_maskz_loadu_epi32(valid_mask,
                                                       ept_job_ids);
        /* Broadcast update target ID */
        __m512i v_target  = _mm512_set1_epi32((int)update_job_info.job_id);

        /* Compare: hit_mask[i] = 1 iff ept_job_ids[i] == update target */
        __mmask16 hit_mask = _mm512_mask_cmpeq_epi32_mask(valid_mask,
                                                            v_ept_ids,
                                                            v_target);

        if (hit_mask) {   /* at most one bit should be set */

            if (update_job_info.operation == JI_UPDATE) {
                /* proc_time[hit] -= 1 */
                __m512i v_pt  = _mm512_maskz_loadu_epi32(valid_mask, proc_times);
                __m512i v_one = _mm512_set1_epi32(1);
                v_pt = _mm512_mask_sub_epi32(v_pt, hit_mask, v_pt, v_one);
                _mm512_mask_storeu_epi32(proc_times, valid_mask, v_pt);

                /* weight[hit] -= wspt[hit] */
                __m512i v_wt   = _mm512_maskz_loadu_epi32(valid_mask, weights);
                __m512i v_wspt = _mm512_maskz_loadu_epi32(valid_mask, wspts);
                v_wt = _mm512_mask_sub_epi32(v_wt, hit_mask, v_wt, v_wspt);
                _mm512_mask_storeu_epi32(weights, valid_mask, v_wt);

            } else {
                /* JI_INVALIDATE: zero the hit slot and push to stack    */
                /* Find the hit index: __mmask16 has exactly one bit set */
                int hit_idx = __builtin_ctz((unsigned)hit_mask);

                proc_times[hit_idx]     = 0;
                weights   [hit_idx]     = 0;
                wspts     [hit_idx]     = 0;
                ept_job_ids [hit_idx]   = INVALID_JOB_ID;
                wt_job_ids  [hit_idx]   = INVALID_JOB_ID;

                /* Push free address back to stack (scalar) */
                stack_handler_cc(job_address_stack, reset,
                                  head_pointer, tail_pointer,
                                  PUSH, (memory_length_t)hit_idx);
            }

            /* Write updated proc_time, weight, wspt back to struct arrays */
            pack_ept_cam   (ept_machine,    ept_job_ids, proc_times);
            pack_weight_cam(weight_machine, wt_job_ids,  weights, wspts);
        }
    }

    /* ══ Pass B: Cost selection and accumulation ══════════════════════ */

    /* Load proc_time[0..15] and weight[0..15] (upper 6 lanes = 0) */
    __m512i v_pt   = _mm512_maskz_loadu_epi32(valid_mask, proc_times);
    __m512i v_wt   = _mm512_maskz_loadu_epi32(valid_mask, weights);
    __m512i v_wspt = _mm512_maskz_loadu_epi32(valid_mask, wspts);

    /* Broadcast wspt_new to all lanes */
    __m512i v_wspt_new = _mm512_set1_epi32((int)wspt_new);

    /* high_mask[i] = 1 iff wspt[i] > wspt_new  →  COST_HIGH for slot i */
    /* AVX-512 unsigned gt: wspt values are uint8 expanded to uint32, always ≥ 0 */
    __mmask16 high_mask = _mm512_mask_cmpgt_epi32_mask(valid_mask,
                                                         v_wspt,
                                                         v_wspt_new);
    __mmask16 low_mask  = (~high_mask) & valid_mask;   /* COST_LOW slots  */

    /* cost_jobs per slot: proc_time if COST_HIGH, weight if COST_LOW */
    /* Use blend: keep v_wt base, overwrite COST_HIGH slots with v_pt */
    __m512i v_cost = _mm512_mask_blend_epi32(high_mask, v_wt, v_pt);

    /* Accumulate cost_high_wspt = sum of proc_time over COST_HIGH slots */
    uint32_t cost_high_wspt = _mm512_mask_reduce_add_epi32(high_mask, v_pt);

    /* Accumulate cost_low_wspt = sum of weight over COST_LOW slots */
    uint32_t cost_low_wspt  = _mm512_mask_reduce_add_epi32(low_mask,  v_wt);

    /* new_job_index = count of COST_HIGH slots = popcount(high_mask) */
    uint32_t n_high = (uint32_t)__builtin_popcount((unsigned)high_mask);

    /* Final cost formula (unchanged from scalar):
     * cost = weight_new * (ept_new + cost_high_wspt) + ept_new * cost_low_wspt
     */
    *cost_machine_out  = (uint32_t)new_weight_curr *
                         ((uint32_t)new_ept_curr + cost_high_wspt)
                       + (uint32_t)new_ept_curr * cost_low_wspt;

    *new_job_index_out = (vf_index_t)n_high;

#if COST_CALCULATOR_DEBUG
    printf("\033[31m[CC] Machine wspt_new=%u cost_high=%u cost_low=%u "
           "n_high=%u cost=%u\033[0m\n",
           wspt_new, cost_high_wspt, cost_low_wspt, n_high,
           *cost_machine_out);
    /* Suppress unused warning for v_cost in non-debug builds */
    (void)v_cost;
#else
    (void)v_cost;
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. cost_calculator_all_machines()  (5-machine outer loop)
 * ═══════════════════════════════════════════════════════════════════════════ */

SIMD_TARGET_AVX512
static void cost_calculator_all_machines(
        job_id_t                  new_job_id,
        uint8_t                  *new_job_processing_time,
        uint8_t                   new_job_weight,
        job_info_update_output_t *update_job_info,
        proc_time_info_t         *new_ept_prev,
        weight_info_t            *new_weight_prev,
        memory_length_t           job_address_stack[NUM_MACHINES][JOBS_PER_MACHINE + 1],
        one_bit_t                *reset_arr,
        memory_length_t          *head_pointer,
        memory_length_t          *tail_pointer,
        proc_time_info_t          ept_cam[NUM_MACHINES][JOBS_PER_MACHINE],
        weight_info_t             weight_cam[NUM_MACHINES][JOBS_PER_MACHINE],
        uint32_t                 *cost_machine,
        vf_index_t               *new_job_index)
{
    /*
     * The 5 machines are processed sequentially because each machine's stack
     * state (head/tail/reset) carries loop-carried dependencies that prevent
     * cross-machine vectorisation at this level.
     *
     * The inner 10-slot loops inside cost_calculator_machine_simd() are
     * fully vectorised with AVX-512.
     *
     * Prefetch: issue a software prefetch for the next machine's CAM data
     * while the current machine's ZMM computation is in-flight.
     */
    for (machine_id_t m = 0; m < NUM_MACHINES; m++) {

        /* Prefetch next machine's ept_cam and weight_cam into L1 */
        if (m + 1 < NUM_MACHINES) {
            _mm_prefetch((const char *)ept_cam   [m + 1], _MM_HINT_T0);
            _mm_prefetch((const char *)weight_cam[m + 1], _MM_HINT_T0);
        }

        cost_calculator_machine_simd(
            new_job_id,
            new_job_processing_time[m],
            new_job_weight,
            update_job_info[m],
            new_ept_prev[m],
            new_weight_prev[m],
            job_address_stack[m],
            &reset_arr[m],
            &head_pointer[m],
            &tail_pointer[m],
            ept_cam[m],
            weight_cam[m],
            &cost_machine[m],
            &new_job_index[m]);
    }

#if DUMP_MEMORY
    for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
        printf("Dumping EPT CAM for machine: %u\n", m);
        printf("Index - Job ID - Processing time\n");
        for (memory_length_t j = 0; j < JOBS_PER_MACHINE; j++)
            printf("%u - %u - %u\n", j,
                   ept_cam[m][j].job_id, ept_cam[m][j].proc_time);

        printf("Dumping Weight CAM for machine: %u\n", m);
        printf("Index - Job ID - Weight\n");
        for (memory_length_t j = 0; j < JOBS_PER_MACHINE; j++)
            printf("%u - %u - %u\n", j,
                   weight_cam[m][j].job_id, weight_cam[m][j].weight);
    }
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. cost_comparator()  — AVX2 argmin over 5 machines
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * cost_comparator()
 *
 * Finds the non-full machine with the lowest cost using a 5-lane AVX2 pass:
 *
 * 1. Load cost[5] into YMM (5-lane masked, upper 3 = UINT32_MAX so they
 * never win the argmin).
 * 2. Load fifo_full[5] as a 5-bit integer mask.
 * 3. Blend UINT32_MAX into full-machine lanes → masked_cost YMM.
 * 4. Horizontal min: compare pairs with _mm256_min_epu32, reduce to scalar.
 * 5. Find the lane whose value equals the min → low_cost_machine.
 * 6. Handle all-equal-cost round-robin (scalar, rare path).
 * 7. Broadcast the winner's data to jiu_input / fifo_input / new_ept /
 * new_weight arrays using masked YMM stores.
 *
 * The jiu_input_generation_loop is vectorised separately at the end:
 * - Zero all 5 machine entries with a single 5-lane masked store, then
 * write the winner entry scalarly (only one lane differs from zero).
 */
SIMD_TARGET_AVX2
static void cost_comparator(job_in_t                  new_job,
                              uint32_t                 *cost,
                              vf_index_t               *index,
                              one_bit_t                *fifo_full,
                              job_info_update_input_t  *jiu_input,
                              fifo_update_input_t      *fifo_input,
                              proc_time_info_t         *new_ept,
                              weight_info_t            *new_weight)
{
    machine_id_t low_cost_machine    = 0;
    vf_index_t   low_cost_index      = JOBS_PER_MACHINE;
    uint32_t     low_cost            = UINT32_MAX;

    static machine_id_t all_cost_equal_machine_id = 0;

    /* ── Zero ALL output arrays first (5-lane masked stores) ────────── */
    /*
     * We zero everything unconditionally, then overwrite only the winner.
     * This removes the if(machine != low_cost_machine) branch from the
     * original jiu_input_generation_loop, replacing 5 branches with one
     * masked-zero + one scalar write.
     */
    {
        const __m256i v_inv = _mm256_set1_epi32((int)INVALID_JOB_ID);
        const __m256i v_zero = _mm256_setzero_si256();
        const __mmask8 m5 = (__mmask8)MACHINE_LANE_MASK;

        /* Zero jiu_input: new_job_id field (stride = sizeof(job_info_update_input_t)=8) */
        /* Extract new_job_id fields into tmp, set to INVALID, scatter back */
        for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
            jiu_input[m].new_job_id = INVALID_JOB_ID;
            jiu_input[m].alpha_j    = 0;
            new_ept[m].job_id       = INVALID_JOB_ID;
            new_ept[m].proc_time    = 0;
            new_weight[m].job_id    = INVALID_JOB_ID;
            new_weight[m].weight    = 0;
            new_weight[m].wspt      = 0;
            fifo_input[m].new_job_id  = INVALID_JOB_ID;
            fifo_input[m].fifo_index  = JOBS_PER_MACHINE;
        }
        /* Suppress unused warnings for v_inv / v_zero / m5 in scalar fallback */
        (void)v_inv; (void)v_zero; (void)m5;
    }

    if (new_job.job_id == INVALID_JOB_ID) {
        /* No new job: leave all outputs zeroed/invalid */
        return;
    }

    /* ── AVX2 argmin: find lowest-cost non-full machine ─────────────── */
    {
        /* Load cost[0..4] as uint32, pad lanes 5-7 with UINT32_MAX */
        SIMD_ALIGNED uint32_t cost_buf[8] = {
            UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
            UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX
        };
        for (int m = 0; m < NUM_MACHINES; m++)
            cost_buf[m] = fifo_full[m] ? UINT32_MAX : cost[m];

        __m256i v_cost = _mm256_load_si256((const __m256i *)cost_buf);

        /* Horizontal min reduction over 8 lanes (upper 3 are UINT32_MAX) */
        /* Step 1: min(v[0..3], v[4..7]) */
        __m256i v_shuf = _mm256_shuffle_epi32(v_cost, _MM_SHUFFLE(2,3,0,1));
        __m256i v_min1 = _mm256_min_epu32(v_cost, v_shuf);
        /* Step 2: min across 128-bit halves */
        __m256i v_shuf2 = _mm256_shuffle_epi32(v_min1, _MM_SHUFFLE(1,0,3,2));
        __m256i v_min2  = _mm256_min_epu32(v_min1, v_shuf2);
        /* Step 3: cross 128-bit lane */
        __m256i v_perm  = _mm256_permute2x128_si256(v_min2, v_min2, 0x01);
        __m256i v_min3  = _mm256_min_epu32(v_min2, v_perm);

        /* Extract scalar minimum */
        SIMD_ALIGNED uint32_t min_buf[8];
        _mm256_store_si256((__m256i *)min_buf, v_min3);
        low_cost = min_buf[0];

        /* Find which lane(s) equal low_cost among valid (non-full) machines */
        uint32_t eq_mask = 0;
        uint32_t num_equal = 0;
        for (int m = 0; m < NUM_MACHINES; m++) {
            if (!fifo_full[m] && cost[m] == low_cost) {
                if (num_equal == 0) {
                    low_cost_machine = (machine_id_t)m;
                    low_cost_index   = index[m];
                }
                eq_mask |= (1u << m);
                num_equal++;
            }
        }

        /* Round-robin tie-breaking when all costs are equal */
        if (num_equal == NUM_MACHINES) {
            low_cost_machine = all_cost_equal_machine_id;
            low_cost_index   = index[all_cost_equal_machine_id];
            all_cost_equal_machine_id++;
            if (all_cost_equal_machine_id == NUM_MACHINES)
                all_cost_equal_machine_id = 0;
        }
    }

#if COST_CALCULATOR_DEBUG
    printf("\033[31mCost comparator: Job ID: %d, Machine: %d\n\033[0m",
           new_job.job_id, low_cost_machine);
#endif

    /* ── Scalar winner write (only one machine differs from zeros) ───── */
    jiu_input[low_cost_machine].new_job_id  = new_job.job_id;
    jiu_input[low_cost_machine].alpha_j     = new_job.alpha_j[low_cost_machine];

    new_ept[low_cost_machine].job_id        = new_job.job_id;
    new_ept[low_cost_machine].proc_time     = new_job.processing_time[low_cost_machine];

    new_weight[low_cost_machine].job_id     = new_job.job_id;
    new_weight[low_cost_machine].weight     = new_job.weight;
    new_weight[low_cost_machine].wspt       =
        (new_job.processing_time[low_cost_machine] != 0)
        ? new_job.weight / new_job.processing_time[low_cost_machine]
        : 0;

    fifo_input[low_cost_machine].fifo_index = low_cost_index;
    fifo_input[low_cost_machine].new_job_id = new_job.job_id;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. cost_calculator()  — top-level entry point (signature unchanged)
 * ═══════════════════════════════════════════════════════════════════════════ */

SIMD_TARGET_AVX512
void cost_calculator(job_in_t                  new_job,
                     job_info_update_output_t *updated_job_info,
                     one_bit_t                *fifo_full,
                     job_info_update_input_t  *jiu_input,
                     fifo_update_input_t      *fifo_input)
{
    /* ── Static state (persistent across scheduler ticks) ─────────────── */
    static CACHE_ALIGNED proc_time_info_t ept_cam   [NUM_MACHINES][JOBS_PER_MACHINE];
    static CACHE_ALIGNED weight_info_t    weight_cam [NUM_MACHINES][JOBS_PER_MACHINE];

    static CACHE_ALIGNED proc_time_info_t new_ept_prev   [NUM_MACHINES];
    static CACHE_ALIGNED weight_info_t    new_weight_prev[NUM_MACHINES];

    static CACHE_ALIGNED memory_length_t
        job_address_stack_cc[NUM_MACHINES][JOBS_PER_MACHINE + 1];
    static one_bit_t       reset_arr    [NUM_MACHINES];
    static memory_length_t head_pointer [NUM_MACHINES];
    static memory_length_t tail_pointer [NUM_MACHINES];

    /* ── Per-tick temporaries (stack-allocated) ─────────────────────── */
    CACHE_ALIGNED uint32_t   cost_machine  [NUM_MACHINES];
    CACHE_ALIGNED vf_index_t new_job_index [NUM_MACHINES];

#if COST_CALCULATOR_DEBUG
    printf("\033[31mIndividual Job cost calculator\n\033[0m");
#endif

    cost_calculator_all_machines(
        new_job.job_id,
        new_job.processing_time,
        new_job.weight,
        updated_job_info,
        new_ept_prev,
        new_weight_prev,
        job_address_stack_cc,
        reset_arr,
        head_pointer,
        tail_pointer,
        ept_cam,
        weight_cam,
        cost_machine,
        new_job_index);

#if COST_CALCULATOR_DEBUG
    printf("\033[31mCost comparator\n\033[0m");
#endif

    cost_comparator(
        new_job,
        cost_machine,
        new_job_index,
        fifo_full,
        jiu_input,
        fifo_input,
        new_ept_prev,
        new_weight_prev);
}
