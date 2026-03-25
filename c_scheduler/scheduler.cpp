
/*
 * This file contains the implementation of entire scheduler
 *
 */

#include "data_types.hpp"
#include "top_modules.hpp"


/*
 * This function is the top module for the scheduler
 *
 * param[in] new_job_data    Data of the new job
 * param[out] popped_job_id  IDs of the top job
 */
void scheduler (job_in_t new_job, job_id_t popped_job_id[NUM_MACHINES],
                one_bit_t fifo_full[NUM_MACHINES]) {

#if SCHEDULER_TOP
    //#pragma HLS INTERFACE m_axi port=new_job bundle=gmem0
    //#pragma HLS INTERFACE m_axi port=popped_job_id_stream bundle=gmem1
    //#pragma HLS INTERFACE m_axi port=fifo_full_stream bundle=gmem2
#endif

    static job_info_update_output_t updated_job_info[NUM_MACHINES] = {0};
    static job_id_t top_job_id[NUM_MACHINES] = {0};
    one_bit_t pop[NUM_MACHINES] = {0};
    job_info_update_input_t jiu_input[NUM_MACHINES] = {0};
    fifo_update_input_t fifo_input[NUM_MACHINES] = {0};

    //#pragma HLS array_partition variable=popped_job_id type=complete dim=0
    //#pragma HLS array_partition variable=fifo_full type=complete dim=0
    //#pragma HLS array_partition variable=updated_job_info type=complete dim=0
    //#pragma HLS array_partition variable=top_job_id type=complete dim=0
    //#pragma HLS array_partition variable=pop type=complete dim=0
    //#pragma HLS array_partition variable=jiu_input type=complete dim=0
    //#pragma HLS array_partition variable=fifo_input type=complete dim=0

#if SCHEDULER_DEBUG
    printf("\033[31mCost Calculator\n\033[0m");
#endif

    cost_calculator(new_job, updated_job_info, fifo_full, jiu_input, fifo_input);

#if SCHEDULER_DEBUG    
    printf("\033[31mJob Info Update \n\033[0m");
#endif

    job_info_update(jiu_input, top_job_id, updated_job_info, pop);

#if SCHEDULER_DEBUG    
    printf("\033[31mVirtual FIFO \n\033[0m");
#endif

    virtual_fifo(fifo_input, pop, top_job_id, popped_job_id, fifo_full);

#if SCHEDULER_DEBUG
    printf("\033[31m--------------------------------------------------------------\033[0m\n");
    for (machine_id_t machine = 0; machine < NUM_MACHINES; machine++) {
        printf("\033[31mMachine: %u Top job: %u Popped Job: %u Fifo Full: %u \033[0m\n",
                machine, top_job_id[machine], popped_job_id[machine], (unsigned int)fifo_full[machine]);
    }
    printf("\033[31m--------------------------------------------------------------\033[0m\n");
#endif
}