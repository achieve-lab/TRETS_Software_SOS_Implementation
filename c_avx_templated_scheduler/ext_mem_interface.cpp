/**
 * @file ext_mem_interface.cpp
 * @brief External memory interface for the Stochastic Online Scheduler.
 *
 * Fully generalized for dynamic NUM_MACHINES using strip-mined loops.
 */

#include "data_types.hpp"
#include "top_modules.hpp"
#include "extm_data_types.hpp"
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstring>

/* ── 1. Helper: Non-temporal write ───────────────────────────────────────── */
SIMD_TARGET_AVX2
static inline void write_perf_info_nt(perf_measurement_info_t *dst,
                                       uint32_t   popped_tick,
                                       machine_id_t machine)
{
    uint64_t packed = (uint64_t)popped_tick
                    | ((uint64_t)(uint16_t)machine << 32);
    _mm_stream_si64(reinterpret_cast<long long *>(dst), (long long)packed);
}

/* ── 2. schedule_jobs (Generalized Driver) ───────────────────────────────── */
SIMD_TARGET_AVX512
void schedule_jobs(new_job_data_host_t          *input_stream,
                   uint32_t                      initial_tick,
                   scheduler_interface_output_t *output)
{
    uint32_t  tick              = initial_tick;
    uint16_t  popped_jobs_count = 0;
    uint16_t  scheduled_jobs    = 0;

    /* Padded internal state arrays */
    CACHE_ALIGNED job_id_t   popped_jobs[MAC_PAD];
    CACHE_ALIGNED one_bit_t  fifo_full  [MAC_PAD];

    /* Zero the entire arrays to clear stack garbage and safely pad bounds */
    memset(popped_jobs, 0, sizeof(popped_jobs));
    memset(fifo_full, 0, sizeof(fifo_full));
    
    new_job_data_host_t input_job;
    memset(&input_job, 0, sizeof(input_job));

    zero_output_simd(output);

    uint8_t input_job_scheduled = 1;
    uint8_t all_fifo_full       = 0;

    while (popped_jobs_count < MEM_DATA_SIZE) {
        job_in_t new_job;
        memset(&new_job, 0, sizeof(new_job));

        if ((scheduled_jobs < MEM_DATA_SIZE) && (input_job_scheduled == 1)) {
            input_job           = input_stream[scheduled_jobs];
            input_job_scheduled = 0;
            scheduled_jobs     += 1;
        }

        if ((input_job.job_data.job_id != INVALID_JOB_ID) &&
            (input_job.release_tick    <= tick)            &&
            (all_fifo_full             == 0)) {
            new_job             = input_job.job_data;
            input_job_scheduled = 1;
            input_job.job_data.job_id = INVALID_JOB_ID;
        }

        /* Call the top-level AVX scheduler tick */
        scheduler(new_job, popped_jobs, fifo_full);

        /* ── Generalized all_fifo_full check ────────────────────────────── */
        uint32_t full_count = 0;
        for (int i = 0; i < NUM_MACHINES; i += AVX512_INT32_LANES) {
            int remain = NUM_MACHINES - i;
            __mmask16 valid_mask = (remain >= 16) ? 0xFFFF : ((1U << remain) - 1);
            
            __m128i v_full_8 = _mm_maskz_loadu_epi8(valid_mask, fifo_full + i);
            __m512i v_full_32 = _mm512_cvtepu8_epi32(v_full_8);
            __mmask16 full_mask = _mm512_cmpneq_epi32_mask(v_full_32, _mm512_setzero_si512());
            
            full_count += __builtin_popcount(full_mask);
        }
        all_fifo_full = (full_count == NUM_MACHINES) ? 1 : 0;

        /* ── Generalized popped-job bookkeeping ─────────────────────────── */
        /* Since scatter stores to output->scheduled_jobs inherently inhibit 
           full vectorization, an explicit loop over valid bounds is safest 
           and scales past the 64-machine limit. */
        for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
            job_id_t jid = popped_jobs[m];
            if (jid != INVALID_JOB_ID) {
                output->num_jobs[m]++;
                write_perf_info_nt(&output->scheduled_jobs[jid], tick, m);
                popped_jobs_count++;
            }
        }

        tick += 1;
    }

    _mm_sfence();
    output->final_tick = tick;
}