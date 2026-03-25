/**
 * @file scheduler.cpp
 * @brief Top-level module for the Stochastic Online Scheduler.
 *
 * Fully generalized for dynamic NUM_MACHINES and JOBS_PER_MACHINE.
 */

#include "top_modules.hpp"
#include "data_types.hpp"

SIMD_TARGET_AVX512
void scheduler(job_in_t   new_job,
               job_id_t  *popped_job_id,
               one_bit_t *fifo_full)
{
    /* * 1. Internal connecting wires/buses.
     * Statically allocated and padded to MAC_PAD to guarantee safe 
     * vector loads in the chunked sub-modules.
     */
    static CACHE_ALIGNED job_info_update_output_t updated_job_info[MAC_PAD];
    static CACHE_ALIGNED job_info_update_input_t  jiu_input       [MAC_PAD];
    static CACHE_ALIGNED fifo_update_input_t      fifo_input      [MAC_PAD];
    static CACHE_ALIGNED one_bit_t                pop_signal      [MAC_PAD];
    static CACHE_ALIGNED job_id_t                 top_job_id      [MAC_PAD];

    /* * 2. Initialize the padding region. 
     * We only strictly need to zero the area between NUM_MACHINES and MAC_PAD 
     * to prevent out-of-bounds lanes from processing ghost jobs.
     */
    for (int m = NUM_MACHINES; m < MAC_PAD; m++) {
        updated_job_info[m].job_id    = INVALID_JOB_ID;
        updated_job_info[m].operation = JI_UPDATE;
        
        jiu_input[m].new_job_id  = INVALID_JOB_ID;
        jiu_input[m].alpha_j     = 0;
        
        fifo_input[m].new_job_id = INVALID_JOB_ID;
        fifo_input[m].fifo_index = JOBS_PER_MACHINE;
        
        pop_signal[m]            = 0;
        fifo_full[m]             = 0;
        top_job_id[m]            = INVALID_JOB_ID;
        popped_job_id[m]         = INVALID_JOB_ID;
    }

    /* ── Sub-Module Execution Pipeline ───────────────────────────────────── */

    /* Pass A: Cost Calculator evaluates the new job and selects the best machine */
    cost_calculator(
        new_job, 
        updated_job_info, 
        fifo_full, 
        jiu_input, 
        fifo_input
    );

    /* Pass B: Virtual FIFO manages job insertion and extracts top jobs */
    virtual_fifo(
        fifo_input, 
        pop_signal, 
        top_job_id, 
        popped_job_id, 
        fifo_full
    );

    /* Pass C: Job Info Update maintains alpha_j counters and generates pop signals */
    job_info_update(
        jiu_input, 
        top_job_id, 
        updated_job_info, 
        pop_signal
    );
}