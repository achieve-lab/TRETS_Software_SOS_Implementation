#ifndef __TOP_MODULES__
#define __TOP_MODULES__

#include "data_types.hpp"
#include "extm_data_types.hpp"

#define VF_TOP 0
#define JIU_TOP 0
#define SCHEDULER_TOP 0
#define COST_CALCULATOR_TOP 0

#define DEBUG 1

#if DEBUG
#define VF_DEBUG 1
#define JIU_DEBUG 1
#define SCHEDULER_DEBUG 1 
#define COST_CALCULATOR_DEBUG 1
#define EXT_MEM_INTERFACE_DEBUG_GEN 1
#define EXT_MEM_INTERFACE_DEBUG_DETAIL 1 
#define DUMP_MEMORY 1
#else
#define VF_DEBUG 0
#define JIU_DEBUG 0
#define SCHEDULER_DEBUG 0
#define COST_CALCULATOR_DEBUG 0
#define EXT_MEM_INTERFACE_DEBUG_GEN 0
#define EXT_MEM_INTERFACE_DEBUG_DETAIL 0
#define DUMP_MEMORY 0
#endif

void virtual_fifo(fifo_update_input_t *input, one_bit_t *pop,
                  job_id_t *top_job_id, job_id_t *popped_job_id,
                  one_bit_t *fifo_full);

void job_info_update(job_info_update_input_t *input, job_id_t *top_job_id, 
                     job_info_update_output_t *output,
                     one_bit_t *pop);

void scheduler (job_in_t new_job, job_id_t popped_job_id[NUM_MACHINES],
                one_bit_t fifo_full[NUM_MACHINES]) ;

void cost_calculator(job_in_t new_job, job_info_update_output_t *updated_job_info, 
                     one_bit_t *fifo_full, job_info_update_input_t* jiu_input,
                     fifo_update_input_t *fifo_input);

void schedule_jobs(new_job_data_host_t *input_stream, uint32_t initial_tick, scheduler_interface_output_t *output);

//void ext_mem_interface(scheduler_interface_input_t *input, scheduler_interface_output_t *output); 

#endif
