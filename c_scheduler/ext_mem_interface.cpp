//External memory interface
//Allows us to manage and decide what to with incoming jobs in the case of internal FIFO saturation
//Also handles seperation of internal/external buffers for accurate memory traffic for real world imp.

#include "data_types.hpp"
#include "top_modules.hpp"
#include "extm_data_types.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>

void schedule_jobs(new_job_data_host_t *input_stream, uint32_t initial_tick, scheduler_interface_output_t *output) {
    
    uint32_t tick = initial_tick;
    new_job_data_host_t input_job = {0};
    uint16_t popped_jobs_count = 0;
    uint16_t scheduled_jobs = 0;
    job_id_t popped_jobs[NUM_MACHINES] = {0};      //array of ids
    scheduler_interface_output_t output_temp = {0};  //long counts
    one_bit_t fifo_full[NUM_MACHINES] = {0};
    uint8_t all_fifo_full = 0;
    uint8_t input_job_scheduled = 1;

    //#pragma HLS array_partition variable=popped_jobs type=complete dim=0
    //#pragma HLS array_partition variable=fifo_full type=complete dim=0

execute:
    //Loop until popped jobs reach MEM_DATA_SIZE
    while (popped_jobs_count < MEM_DATA_SIZE) {
        //check if all the input jobs are scheduled
        job_in_t new_job = {0};
        if ((scheduled_jobs < MEM_DATA_SIZE) && (input_job_scheduled == 1)) {
            //MEM_DATA_SIZE jobs not read and already read input job is scheduled
            input_job = input_stream[scheduled_jobs];
            input_job_scheduled = 0;
            scheduled_jobs += 1;
        }

        if ((input_job.job_data.job_id != INVALID_JOB_ID) && (input_job.release_tick <= tick) && (all_fifo_full == 0)) {
            new_job = input_job.job_data;
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
        printf(" Next_Tick: %d\n", static_cast<unsigned int>(input_job.release_tick));
        printf("--------------------------------------------------------------\n");
#endif

        scheduler(new_job, popped_jobs, fifo_full);

        all_fifo_full = 1;

all_fifo_full_check:
        for (machine_id_t machine = 0; machine < NUM_MACHINES; machine++) {
            //#pragma HLS unroll

#if EXT_MEM_INTERFACE_DEBUG_DETAIL
            printf("\033[32mMachine %d:\033[0m", machine);
            printf(" %d\n", output_temp.num_jobs[machine]);
#endif

            all_fifo_full &= fifo_full[machine];

            //check if a job is popped
            if (popped_jobs[machine] != INVALID_JOB_ID) {
                output_temp.scheduled_jobs[popped_jobs[machine]].popped_tick = tick;
                output_temp.scheduled_jobs[popped_jobs[machine]].machine_scheduled = machine;
                output_temp.num_jobs[machine] += 1;
                popped_jobs_count += 1;
#if EXT_MEM_INTERFACE_DEBUG_DETAIL
                printf("\033[32m\nJob %d\033[0m", popped_jobs[machine]);
                printf("\033[32m Scheduled in tick %d \033[0m", (int)tick);
#endif
            }
#if EXT_MEM_INTERFACE_DEBUG_DETAIL
            printf("\033[32m--------------------------------------------------------------\033[0m\n");
#endif
        }

        tick += 1;
    }

#if EXT_MEM_INTERFACE_DEBUG_GEN 
    printf("\nEnd\n");
    //Result returns/writes
    for (machine_id_t machine = 0; machine < NUM_MACHINES; machine++){
        printf("Machine %d:", (machine + 1));
        printf(" %d\n", output_temp.num_jobs[machine]);
    }
#endif

    output_temp.final_tick = tick;

    *output = output_temp;

}




