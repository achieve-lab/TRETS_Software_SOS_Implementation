/**
 * @file ext_mem_interface.cpp
 * @brief External memory interface for the Stochastic Online Scheduler.
 *
 * AVX-512 / AVX2 vectorisation (Intel Xeon 4th-gen "Sapphire Rapids")
 * =====================================================================
 *
 * Role of this file
 * -----------------
 * schedule_jobs() is the outer driver loop that feeds MEM_DATA_SIZE (100)
 * jobs through the scheduler tick-by-tick until all 100 jobs have been
 * popped from the FIFOs.  It sits between the host (ext_mem_host.cpp) and
 * the single-tick scheduler().
 *
 * Original hot-spots and their SIMD replacements
 * -----------------------------------------------
 *
 * 1. all_fifo_full_check loop  (NUM_MACHINES = 5 iterations per tick)
 *    ─────────────────────────────────────────────────────────────────
 *    Original:
 *      all_fifo_full = 1;
 *      for (machine) { all_fifo_full &= fifo_full[machine]; ... }
 *
 *    SIMD replacement:
 *      all_bits_set_5(fifo_full)  from top_modules.hpp
 *      Uses PCMPEQB + PMOVMSKB: loads 5 bytes into XMM, checks all
 *      non-zero in ~3 cycles vs. 5 scalar AND-branches.
 *
 * 2. Popped-job bookkeeping inner loop  (NUM_MACHINES = 5 per tick)
 *    ─────────────────────────────────────────────────────────────────
 *    Original:
 *      for (machine) {
 *          if (popped_jobs[machine] != INVALID_JOB_ID) {
 *              output_temp.scheduled_jobs[id].popped_tick = tick;
 *              output_temp.scheduled_jobs[id].machine_scheduled = machine;
 *              output_temp.num_jobs[machine]++;
 *              popped_jobs_count++;
 *          }
 *      }
 *
 *    SIMD replacement  (AVX2 5-lane):
 *      a. Load popped_jobs[0..4] into a YMM (5-lane masked load,
 *         upper 3 lanes = INVALID_JOB_ID via zero-mask).
 *      b. Compare != INVALID_JOB_ID → valid_mask (__mmask8, 5 bits).
 *      c. For each set bit in valid_mask (at most 5, usually 0-2):
 *           scalar write to scheduled_jobs[id] (scatter — struct fields
 *           at variable offsets, not amenable to SIMD store).
 *      d. Increment num_jobs[machine] for each set bit:
 *           Load num_jobs[0..7] into XMM (8 × uint16), add valid_mask
 *           expanded to 8 × uint16, store back.
 *      e. popped_jobs_count += __builtin_popcount(valid_mask).
 *
 *    The per-job struct writes (step c) remain scalar because
 *    perf_measurement_info_t has non-uniform field widths and scatter
 *    indices (job IDs) are data-dependent.
 *
 * 3. Software prefetch of input_stream
 *    ─────────────────────────────────────────────────────────────────
 *    Each new_job_data_host_t is 64 bytes (one cache line).  We issue
 *    _mm_prefetch with _MM_HINT_T0 two entries ahead of the current
 *    scheduled_jobs index so the next job's data is in L1 by the time
 *    scheduler() returns.
 *
 * 4. Non-temporal stores for output_temp.scheduled_jobs[]
 *    ─────────────────────────────────────────────────────────────────
 *    perf_measurement_info_t is 8 bytes and is written once (at pop time)
 *    and never read again inside schedule_jobs().  Non-temporal (streaming)
 *    stores bypass the L1/L2 write-allocate path, freeing cache capacity
 *    for the hot scheduler state.
 *    Implemented via a helper write_perf_info_nt() using _mm_stream_si64.
 *
 * Function signature is UNCHANGED from the scalar version.
 */

#include "data_types.hpp"
#include "top_modules.hpp"
#include "extm_data_types.hpp"
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstring>

/* ═══════════════════════════════════════════════════════════════════════════
 * Helper: non-temporal write of one perf_measurement_info_t entry
 *
 * perf_measurement_info_t layout (8 bytes, alignas(8)):
 *   uint32_t  popped_tick        [offset 0]
 *   uint16_t  machine_scheduled  [offset 4]
 *   uint8_t   _pad[2]            [offset 6]
 *
 * We pack the two meaningful fields into a single uint64_t and issue one
 * _mm_stream_si64 (MOVNTI 64-bit).  This avoids a write-allocate on the
 * output array (which is large: MANAGER_SIZE × 8 bytes ≈ 896 bytes) and
 * keeps the cache available for the hot scheduler working set.
 * ═══════════════════════════════════════════════════════════════════════════ */

SIMD_TARGET_AVX2
static inline void write_perf_info_nt(perf_measurement_info_t *dst,
                                       uint32_t   popped_tick,
                                       machine_id_t machine)
{
    /* Pack into a 64-bit integer:
     *   bits  0-31 : popped_tick
     *   bits 32-47 : machine_scheduled
     *   bits 48-63 : 0 (padding)
     */
    uint64_t packed = (uint64_t)popped_tick
                    | ((uint64_t)(uint16_t)machine << 32);

    /* _mm_stream_si64 requires 8-byte aligned destination.
     * perf_measurement_info_t is alignas(8) so dst is always aligned. */
    _mm_stream_si64(reinterpret_cast<long long *>(dst), (long long)packed);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Helper: increment num_jobs counters for all popped machines in one pass
 *
 * num_jobs is uint16_t[8] (padded from NUM_MACHINES=5 to 8 in
 * scheduler_interface_output_t — see extm_data_types.hpp).
 *
 * valid_pop_mask : bitmask where bit m is set iff popped_jobs[m] != INVALID.
 * We expand this 5-bit mask to a uint16_t[8] increment vector and add it
 * to num_jobs in one SSE2 PADDW instruction.
 * ═══════════════════════════════════════════════════════════════════════════ */

SIMD_TARGET_AVX2
static inline void increment_num_jobs_simd(uint16_t   *num_jobs,  /* [8] */
                                            uint32_t    valid_pop_mask)
{
    /* Load current counters */
    __m128i v_counts = _mm_load_si128((const __m128i *)num_jobs);

    /* Build increment vector: 1 for each set bit in valid_pop_mask, else 0.
     * Expand bit m → lane m of a uint16_t[8] vector.
     *
     * Strategy:
     *   1. Broadcast valid_pop_mask (32-bit) to all 8 lanes of a YMM.
     *   2. AND each lane with its positional bit mask {1,2,4,8,16,0,0,0}.
     *   3. Compare != 0 → 0xFFFF for active lanes.
     *   4. AND with 0x0001 → increment of 1 for active lanes, 0 otherwise.
     */
    const __m128i v_bit_masks = _mm_set_epi16(
        0, 0, 0,                        /* lanes 7,6,5 — always 0 (padding) */
        (short)(1 << 4),                /* lane 4 — machine 4               */
        (short)(1 << 3),                /* lane 3 — machine 3               */
        (short)(1 << 2),                /* lane 2 — machine 2               */
        (short)(1 << 1),                /* lane 1 — machine 1               */
        (short)(1 << 0)                 /* lane 0 — machine 0               */
    );

    /* Broadcast the 32-bit mask to all 8 × 16-bit lanes */
    __m128i v_mask = _mm_set1_epi16((short)(valid_pop_mask & 0x1F));

    /* AND with positional bits: non-zero iff the machine's bit is set */
    __m128i v_active = _mm_and_si128(v_mask, v_bit_masks);

    /* Compare each 16-bit lane != 0 → 0xFFFF, else 0x0000 */
    __m128i v_zero  = _mm_setzero_si128();
    __m128i v_neq   = _mm_andnot_si128(
                          _mm_cmpeq_epi16(v_active, v_zero),
                          _mm_set1_epi16(1));  /* 1 where active, 0 elsewhere */

    /* Add increments to existing counts */
    v_counts = _mm_add_epi16(v_counts, v_neq);

    /* Store back (aligned: num_jobs is alignas(16) in scheduler_interface_output_t) */
    _mm_store_si128((__m128i *)num_jobs, v_counts);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Helper: build the popped-jobs valid mask for one tick
 *
 * Loads popped_jobs[0..4] (uint32_t × 5) into a YMM, compares each lane
 * against INVALID_JOB_ID (0), and returns a 5-bit mask where bit m = 1
 * iff popped_jobs[m] != INVALID_JOB_ID.
 * ═══════════════════════════════════════════════════════════════════════════ */

SIMD_TARGET_AVX2
static inline uint32_t popped_jobs_valid_mask(const job_id_t *popped_jobs)
{
#if HAVE_AVX512VL
    /* AVX-512VL: 5-lane masked compare-neq → __mmask8 directly */
    const __mmask8 m5   = (__mmask8)MACHINE_LANE_MASK;
    __m256i v_pj        = _mm256_maskz_loadu_epi32(m5, popped_jobs);
    __m256i v_inv       = _mm256_setzero_si256();   /* INVALID_JOB_ID = 0 */
    __mmask8 valid      = _mm256_mask_cmpneq_epu32_mask(m5, v_pj, v_inv);
    return (uint32_t)valid & MACHINE_LANE_MASK;
#else
    /* AVX2 fallback: load 8 lanes (pad with INVALID=0), compare, movemask */
    SIMD_ALIGNED uint32_t buf[8] = {0,0,0,0,0,0,0,0};
    for (int m = 0; m < NUM_MACHINES; m++) buf[m] = popped_jobs[m];
    __m256i v_pj  = _mm256_load_si256((const __m256i *)buf);
    __m256i v_inv = _mm256_setzero_si256();
    /* cmpeq gives 0xFFFFFFFF where equal; we want not-equal */
    __m256i v_eq  = _mm256_cmpeq_epi32(v_pj, v_inv);
    /* movemask extracts the MSB of each 32-bit lane → 8-bit mask */
    /* Invert: bit m set iff NOT equal (i.e. != INVALID) */
    int eq_mask   = _mm256_movemask_epi8(v_eq);
    /* movemask_epi8 gives one bit per byte (32 bits for 8 × 4-byte lanes).
     * Extract the sign bit of each 32-bit element (bit 31 of each lane).
     * For cmpeq_epi32 result: 0xFFFFFFFF → MSB=1, 0x00000000 → MSB=0.
     * We want 1 where NOT equal, so invert. */
    /* Use a per-lane approach: check bit 31 of each 4-byte group */
    uint32_t valid = 0;
    SIMD_ALIGNED uint32_t tmp[8];
    _mm256_store_si256((__m256i *)tmp, v_eq);
    for (int m = 0; m < NUM_MACHINES; m++)
        if (tmp[m] == 0)          /* cmpeq gave 0 → not equal → valid pop */
            valid |= (1u << m);
    return valid & MACHINE_LANE_MASK;
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * schedule_jobs()  — top-level entry point (signature unchanged)
 * ═══════════════════════════════════════════════════════════════════════════ */

SIMD_TARGET_AVX512
void schedule_jobs(new_job_data_host_t          *input_stream,
                   uint32_t                      initial_tick,
                   scheduler_interface_output_t *output)
{
    uint32_t  tick             = initial_tick;
    uint16_t  popped_jobs_count = 0;
    uint16_t  scheduled_jobs   = 0;

    /* Per-tick scheduler outputs */
    CACHE_ALIGNED job_id_t   popped_jobs[NUM_MACHINES];
    CACHE_ALIGNED one_bit_t  fifo_full  [NUM_MACHINES];

    /* Zero-initialise outputs */
    *(uint32_t *)popped_jobs = 0u; popped_jobs[4] = 0u;
    *(uint32_t *)fifo_full   = 0u; fifo_full  [4] = 0u;

    /* Working copy of the current unscheduled input job */
    new_job_data_host_t input_job;
    memset(&input_job, 0, sizeof(input_job));

    /* Zero the output struct (non-temporal to avoid cache pollution) */
    zero_output_simd(output);

    uint8_t input_job_scheduled = 1;
    uint8_t all_fifo_full       = 0;

    /* ── Main scheduling loop ─────────────────────────────────────────
     * Runs until all MEM_DATA_SIZE jobs have been popped from the FIFOs.
     * ────────────────────────────────────────────────────────────────── */
    while (popped_jobs_count < MEM_DATA_SIZE) {

        /* ── Load next input job if the previous one was scheduled ──── */
        job_in_t new_job;
        memset(&new_job, 0, sizeof(new_job));

        if ((scheduled_jobs < MEM_DATA_SIZE) && (input_job_scheduled == 1)) {
            /* Prefetch the job two entries ahead into L1 */
            // if (scheduled_jobs + 2 < MEM_DATA_SIZE) {
            //     _mm_prefetch(
            //         (const char *)&input_stream[scheduled_jobs + 2],
            //         _MM_HINT_T0);
            // } //Removed Preloading as in online situation would not have access to future jobs yet.
            input_job           = input_stream[scheduled_jobs];
            input_job_scheduled = 0;
            scheduled_jobs     += 1;
        }

        /* ── Decide whether to pass the job to the scheduler this tick  */
        if ((input_job.job_data.job_id != INVALID_JOB_ID) &&
            (input_job.release_tick    <= tick)            &&
            (all_fifo_full             == 0)) {
            new_job             = input_job.job_data;
            input_job_scheduled = 1;
            input_job.job_data.job_id = INVALID_JOB_ID;
        }

#if EXT_MEM_INTERFACE_DEBUG_GEN
        printf("Current_Tick: %u\n", tick);
        printf("\nJob info:\n");
        printf("Id: %d", new_job.job_id);
        printf(" Jobs Received: %d ", scheduled_jobs);
        printf(" Processing time: %d ", new_job.processing_time[0]);
        printf(" Weight: %d ", new_job.weight);
        printf(" Next_Tick: %d\n",
               (unsigned int)input_job.release_tick);
        printf("---------------------------------------------\n");
#endif

        /* ── Single-tick scheduling step ────────────────────────────── */
        scheduler(new_job, popped_jobs, fifo_full);

        /* ── SIMD all_fifo_full check ────────────────────────────────
         * all_bits_set_5() uses PCMPEQB + PMOVMSKB (top_modules.hpp).
         * Replaces the scalar  all_fifo_full &= fifo_full[machine] loop.
         * ─────────────────────────────────────────────────────────── */
        all_fifo_full = (uint8_t)all_bits_set_5(fifo_full);

        /* ── SIMD popped-job bookkeeping ─────────────────────────────
         *
         * 1. Build valid_pop_mask: bit m = 1 iff popped_jobs[m] != INVALID.
         * 2. Increment num_jobs[m] for each set bit via SSE2 PADDW.
         * 3. Write perf_measurement_info via non-temporal _mm_stream_si64.
         * 4. Update popped_jobs_count via popcount.
         * ─────────────────────────────────────────────────────────── */
        uint32_t valid_pop_mask = popped_jobs_valid_mask(popped_jobs);

        if (valid_pop_mask) {
            /* Increment num_jobs counters for all popped machines at once */
            increment_num_jobs_simd(output->num_jobs, valid_pop_mask);

            /* Per-job struct writes and count update (scalar — data-dependent
             * scatter indices prevent SIMD store) */
            uint32_t remaining = valid_pop_mask;
            while (remaining) {
                int m = __builtin_ctz(remaining);   /* lowest set bit */
                remaining &= remaining - 1;         /* clear lowest bit */

                job_id_t jid = popped_jobs[m];

#if EXT_MEM_INTERFACE_DEBUG_DETAIL
                printf("\033[32mMachine %d:\033[0m %d\n",
                       m, (int)output->num_jobs[m]);
                printf("\033[32m\nJob %d\033[0m", jid);
                printf("\033[32m Scheduled in tick %d \033[0m", (int)tick);
                printf("\033[32m-----------------------------------\033[0m\n");
#endif
                /* Non-temporal write: popped_tick and machine_scheduled */
                write_perf_info_nt(&output->scheduled_jobs[jid],
                                   tick, (machine_id_t)m);
            }

            /* Update popped count: popcount of valid_pop_mask */
            popped_jobs_count += (uint16_t)__builtin_popcount(valid_pop_mask);
        }
#if EXT_MEM_INTERFACE_DEBUG_DETAIL
        else {
            /* No pops this tick — still print machine counters in debug */
            for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
                printf("\033[32mMachine %d:\033[0m %d\n",
                       m, (int)output->num_jobs[m]);
                printf("\033[32m-----------------------------------\033[0m\n");
            }
        }
#endif

        tick += 1;

    } /* end while (popped_jobs_count < MEM_DATA_SIZE) */

    /* Flush non-temporal (streaming) stores before the caller reads output */
    _mm_sfence();

#if EXT_MEM_INTERFACE_DEBUG_GEN
    printf("\nEnd\n");
    for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
        printf("Machine %d: %d\n", (m + 1), (int)output->num_jobs[m]);
    }
#endif

    output->final_tick = tick;
}
