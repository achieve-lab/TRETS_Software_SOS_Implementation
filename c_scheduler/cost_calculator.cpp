/*
 * This file contains the implementation of virtual FIFO
 * Submodules in this file:
 *      individual_job_cost_calculator
 *      cost_comparator
 */

#include "data_types.hpp"
#include "top_modules.hpp"

/*
 * This function implements the stack to be used for finding the new job address
 * 
 * param[in] stack  Pointer to the stack
 * param[in] reset  Reset signal - Active low reset handling
 * param[in] head   Head of the stack
 * param[in] tail   Tail of the stack
 * param[in] operation    Stack operation
 * param[in] push_address Adress to be pushed to the stack
 * 
 * param[out] pop_address Address were the new job will go
 */
memory_length_t stack_handler_cc(memory_length_t *stack, one_bit_t *reset, memory_length_t *head,
                              memory_length_t *tail, one_bit_t operation, memory_length_t push_address) {

    /* Note: Irrespective of the operation type, peek operation on stack is always done
     * That is, always the top of the stack is returned by the function
     * Only on pop operation, head pointer is updated
     */
    //#pragma HLS inline off

    memory_length_t new_job_address = INVALID_ADDRESS;

    /* Reset handling */
    if (*reset == 0) {
        *head = 0;
        *tail = JOBS_PER_MACHINE;

stack_handler_reset_loop_cc:
        for (memory_length_t index = 0; index < JOBS_PER_MACHINE; index++) {
            //#pragma HLS UNROLL
            stack[index] = index;
        }
        *reset = 1;
    }

    if (operation == PUSH) {
        /* Stack full condition checking */
        if (*tail == JOBS_PER_MACHINE) {
            if (*head == 0) {
                /* One empty slot should always be present, so tail has one empty spot and the next
                   spot is head, so the stack is considered to be full */
                new_job_address = INVALID_ADDRESS;
            } else {
                /* Stack is not full, push_address goes at location zero */
                stack[*tail] = push_address;
                *tail = 0;
                new_job_address = stack[*head];
            }
        } else {
            if ((*tail+1) == *head) {
                /* stack is full */
                new_job_address = INVALID_ADDRESS;
            } else {
                /* Stack is not full, push_address goes to tail + 1 */
                stack[*tail] = push_address;
                *tail = *tail + 1;
                new_job_address = stack[*head];
            }
        }
    } else {
        /*i Operation is pop */
        if (*head == *tail) {
            /* Stack is empty */
            new_job_address = INVALID_ADDRESS;
        } else {
            new_job_address = stack[*head];
            if (*head == JOBS_PER_MACHINE) {
                /* Wrap around condition - this if check is avoid modulus operation */
                *head = 0;
            } else {
                *head = *head + 1;
            }
        }
    }
    return new_job_address;
}

/*
 * This function implements the job calculator for each machine
 *
 * param[in] wspt_new       Wspt of the new job
 * param[in] updated_ept    Updated processing time 
 * param[in] updated_weight Updated weight
 * param[in] fifo_full      Fifo full status
 * 
 * param[out] cost_high_wspt
 * param[out] cost_low_wspt
 */
void cost_calculator_job(uint8_t wspt_new, job_info_update_output_t update_job_info, 
                         memory_length_t job_index,
                         memory_length_t *job_address_stack, one_bit_t *reset,
                         memory_length_t *head_pointer, memory_length_t *tail_pointer,
                         proc_time_info_t *ept_job, weight_info_t *weight_job, 
                         uint8_t *cost_job, one_bit_t *cost_high_low_selector) {
    /*
    uint8_t ept_job_local = ept_job->proc_time; //CAUTION: DATA TYPE NEEDS TO BE CHANGED WHEN PROCESSING TIME PRECISION IS CHANGED
    uint8_t weight_job_local = weight_job->weight; //CAUTION: DATA TYPE NEEDS TO BE CHANGED WHEN WEIGHT AND WSPT PRECISION IS CHANGED
    uint8_t wspt_local = weight_job->wspt;
    uint8_t job_id = ept_job->job_id;
    */
    memory_length_t new_job_address = INVALID_ADDRESS;
    //#pragma HLS inline off

    //Check if the current job details needs to be updated
    if (update_job_info.job_id != INVALID_JOB_ID) {
        if (update_job_info.job_id == ept_job->job_id) {
            if (update_job_info.operation == JI_UPDATE) {
                ept_job->proc_time -= 1;
                weight_job->weight -= weight_job->wspt;
            } else {
                //Job invalidation
                ept_job->proc_time = 0;
                weight_job->weight = 0;
                weight_job->wspt = 0;
                new_job_address = stack_handler_cc(job_address_stack, reset, head_pointer, tail_pointer, PUSH, job_index);
            }
        }
    }

    if (weight_job->wspt > wspt_new) {
        *cost_job = ept_job->proc_time;
        *cost_high_low_selector = COST_HIGH;
    } else {
        *cost_job = weight_job->weight;
        *cost_high_low_selector = COST_LOW;
    }

    //ept_job->proc_time = ept_job_local;
    //weight_job->weight = weight_job_local;
    //weight_job->wspt = wspt_local;
}

void cost_calculator_machine (job_id_t new_job_id, uint8_t new_ept_curr,
                              uint8_t new_weight_curr , job_info_update_output_t update_job_info,
                              proc_time_info_t new_ept_prev, weight_info_t new_weight_prev,
                              memory_length_t *job_address_stack, one_bit_t *reset,
                              memory_length_t *head_pointer, memory_length_t *tail_pointer,
                              proc_time_info_t *ept_machine, weight_info_t *weight_machine, 
                              uint8_t *cost_jobs, one_bit_t *cost_high_low_selector,
                              uint32_t *cost_machine, vf_index_t *new_job_index) {

    //#pragma HLS inline off

    //uint8_t wspt_new = 0;
    //memory_length_t cost_index = 0; //index of the new job
    uint16_t cost_high_wspt = 0;
    uint16_t cost_low_wspt = 0;
    uint8_t wspt = 0;

    if (new_ept_prev.job_id != INVALID_JOB_ID) {
    

        memory_length_t new_job_address  = stack_handler_cc(job_address_stack, reset, head_pointer, 
			tail_pointer, POP, INVALID_ADDRESS);
        
        if (new_job_address == INVALID_ADDRESS) {
            printf("Invalid Address hit!!! Error!!!, Breaking\n");
            exit(0);
        }
	ept_machine[new_job_address].job_id = new_ept_prev.job_id;
        ept_machine[new_job_address].proc_time = new_ept_prev.proc_time;


        weight_machine[new_job_address].job_id = new_weight_prev.job_id;
        weight_machine[new_job_address].weight = new_weight_prev.weight;
        weight_machine[new_job_address].wspt = new_weight_prev.wspt;
    }

    if (new_job_id != INVALID_JOB_ID) {
        wspt = new_weight_curr / new_ept_curr;
    } else {
        wspt = 0;
    }

cost_calculator_machine_loop1:
    for (memory_length_t job = 0; job < JOBS_PER_MACHINE; job++) {
        //#pragma HLS unroll

        cost_calculator_job(wspt, update_job_info, job, job_address_stack, reset,
                            head_pointer, tail_pointer, &ept_machine[job],  &weight_machine[job], 
                            &cost_jobs[job], &cost_high_low_selector[job]);

        //cost_high_wspt += ((cost_high_low_selector[job] == COST_HIGH) ? cost_jobs[job] : 0);
        //cost_low_wspt += ((cost_high_low_selector[job] == COST_LOW) ? cost_jobs[job] : 0);
        //cost_index += ((cost_high_low_selector[job] == COST_HIGH) ? 1 : 0);
    }

cost_calculator_machine_loop2:
    for (job_id_t job = 0; job < JOBS_PER_MACHINE; job++) {
        //Separate loop added to see if expression balancing is working
        //#pragma HLS pipeline off
        //#pragma HLS unroll
        cost_high_wspt += ((cost_high_low_selector[job] == COST_HIGH) ? cost_jobs[job] : 0);
        cost_low_wspt += ((cost_high_low_selector[job] == COST_LOW) ? cost_jobs[job] : 0);
        *new_job_index += ((cost_high_low_selector[job] == COST_HIGH) ? 1 : 0);
    }

    *cost_machine = (new_weight_curr * (new_ept_curr + cost_high_wspt) + (new_ept_curr * cost_low_wspt));
    //*new_job_index = cost_index;
    //*wspt = wspt_new;

}


void cost_calculator_all_machines (job_id_t new_job_id, uint8_t *new_job_processing_time,
                                   uint8_t new_job_weight,
                                   job_info_update_output_t *update_job_info,
                                   proc_time_info_t *new_ept_prev, weight_info_t *new_weight_prev,
                                   memory_length_t job_address_stack[NUM_MACHINES][JOBS_PER_MACHINE + 1], one_bit_t *reset,
                                   memory_length_t *head_pointer, memory_length_t *tail_pointer,
                                   proc_time_info_t ept_cam[NUM_MACHINES][JOBS_PER_MACHINE], 
                                   weight_info_t weight_cam[NUM_MACHINES][JOBS_PER_MACHINE], 
                                   uint8_t cost_jobs[NUM_MACHINES][JOBS_PER_MACHINE], 
                                   one_bit_t cost_high_low_selector[NUM_MACHINES][JOBS_PER_MACHINE],
                                   uint32_t *cost_machine, vf_index_t *new_job_index) {
    //#pragma HLS inline off
cost_calculator_all_machines_loop1:
    for (machine_id_t machine = 0; machine < NUM_MACHINES; machine++) {
        //#pragma HLS unroll factor=UNROLL_FACTOR
        cost_calculator_machine(new_job_id, new_job_processing_time[machine], 
                                new_job_weight, update_job_info[machine], new_ept_prev[machine],
                                new_weight_prev[machine], job_address_stack[machine], &reset[machine],
                                &head_pointer[machine], &tail_pointer[machine], ept_cam[machine], 
                                weight_cam[machine], cost_jobs[machine], cost_high_low_selector[machine], &cost_machine[machine],
                                &new_job_index[machine]);
    }

#if DUMP_MEMORY
    for (machine_id_t machine = 0; machine < NUM_MACHINES; machine++) {
        printf("Dumping EPT CAM for machine: %u\n", machine);
        printf("Index - Job ID - Processing time\n");
        for (memory_length_t job = 0; job < JOBS_PER_MACHINE; job++) {
            printf("%u - %u - %u\n", job, ept_cam[machine][job].job_id, ept_cam[machine][job].proc_time);
        }

        printf("Dumping Weight CAM for machine: %u\n", machine);
        printf("Index - Job ID - Weight\n");
        for (memory_length_t job = 0; job < JOBS_PER_MACHINE; job++) {
            printf("%u - %u - %u\n", job, weight_cam[machine][job].job_id, weight_cam[machine][job].weight);
        }
    }
#endif

//#if COST_CALCULATOR_DEBUG
//    for (machine_id_t machine = 0; machine < NUM_MACHINES; machine++) {
//        printf("\033[31mCost calculator: Machine: %u\n\033[0m", machine);
//        printf("\033[31mCost calculator: Updated Job ID: %u, Operation: %u\n\033[0m", 
//        update_job_info[machine].job_id, (unsigned int)update_job_info[machine].operation); 
//        printf("\033[31mCost calculator: Cost: %u\n\033[0m", cost_machine[machine]);
//    }
//#endif

}  

/*
 * This function compares the cost of all the machines and generates the input
 * for each machine
 */
void cost_comparator(job_in_t new_job, uint32_t *cost, vf_index_t* index, 
                     one_bit_t *fifo_full,
                     job_info_update_input_t* jiu_input,
                     fifo_update_input_t *fifo_input,
                     proc_time_info_t *new_ept,
                     weight_info_t *new_weight) {


    machine_id_t low_cost_machine = 0;
    vf_index_t low_cost_index = JOBS_PER_MACHINE;
    uint32_t low_cost = 4294967295; // 2^32-1

    /* 
     * machine to be chosen when all cost are equal. If all costs are equal, without
     * special handling, all the jobs will be assigned to machine 0. To avoid that,
     * when costs are equal, follow round robin scheduling
     */
    static machine_id_t all_cost_equal_machine_id = 0; 
    machine_id_t num_machines_equal_cost = 1;
    
    if (new_job.job_id != INVALID_JOB_ID) {
cost_comparator_loop:
        for (machine_id_t machine = 0; machine < NUM_MACHINES; machine++) {

            if (fifo_full[machine] != 1) {
                if (cost[machine] < low_cost) {
                    low_cost_machine = machine;
                    low_cost_index = index[machine];
                    low_cost = cost[machine];
                } else if (cost[machine] == cost[low_cost_machine]) {
                    //two machines have equal cost
                    num_machines_equal_cost += 1;
                }
            }
        }

        if (num_machines_equal_cost == NUM_MACHINES) {
            low_cost_machine = all_cost_equal_machine_id;
            low_cost_index = index[all_cost_equal_machine_id];
            all_cost_equal_machine_id += 1;
            if (all_cost_equal_machine_id == NUM_MACHINES) {
                all_cost_equal_machine_id = 0;
            }
        }
    }

#if COST_CALCULATOR_DEBUG
    printf("\033[31mCost comparator: Job ID: %d, Machine: %d\n\033[0m", new_job.job_id, low_cost_machine);
#endif

jiu_input_generation_loop:
    for (machine_id_t machine = 0; machine < NUM_MACHINES; machine++) {
        //#pragma HLS UNROLL

        if (machine != low_cost_machine) {
            jiu_input[machine].new_job_id = INVALID_JOB_ID;
            jiu_input[machine].alpha_j = 0;

            new_ept[machine].job_id = INVALID_JOB_ID;
            new_ept[machine].proc_time = 0;

            new_weight[machine].job_id = INVALID_JOB_ID;
            new_weight[machine].weight = 0;
            new_weight[machine].wspt = 0;

            fifo_input[machine].new_job_id = INVALID_JOB_ID;
            fifo_input[machine].fifo_index = JOBS_PER_MACHINE; //TODO: Check why JOBS_PER_MACHINE here
        }

    }

    if (new_job.job_id != INVALID_JOB_ID) {
        jiu_input[low_cost_machine].new_job_id = new_job.job_id;
        jiu_input[low_cost_machine].alpha_j = new_job.alpha_j[low_cost_machine];

        new_ept[low_cost_machine].job_id = new_job.job_id;
        new_ept[low_cost_machine].proc_time = new_job.processing_time[low_cost_machine];

        new_weight[low_cost_machine].job_id = new_job.job_id;
        new_weight[low_cost_machine].weight = new_job.weight;
        new_weight[low_cost_machine].wspt = new_job.weight/new_job.processing_time[low_cost_machine];

        fifo_input[low_cost_machine].fifo_index = index[low_cost_machine];
        fifo_input[low_cost_machine].new_job_id = new_job.job_id;
    } else {
        jiu_input[low_cost_machine].new_job_id = INVALID_JOB_ID;
        jiu_input[low_cost_machine].alpha_j = 0;

        new_ept[low_cost_machine].job_id = INVALID_JOB_ID;
        new_ept[low_cost_machine].proc_time = 0;

        new_weight[low_cost_machine].job_id = INVALID_JOB_ID;
        new_weight[low_cost_machine].weight = 0;
        new_weight[low_cost_machine].wspt = 0;

        fifo_input[low_cost_machine].new_job_id = INVALID_JOB_ID;
        fifo_input[low_cost_machine].fifo_index = JOBS_PER_MACHINE; 
    }

}

/*
 * This function is the top module for cost calculator
 *
 * param[in] new_job            Details of the new job
 * param[in] updated_ept        Updated processing time
 * param[in] updated_weight     Updated weight
 * param[in] fifo_full          Boolean array of fifo full conditions
 * 
 * param[out] jiu_input         Input to the job info update module
 */
void cost_calculator(job_in_t new_job, job_info_update_output_t *updated_job_info, 
                     one_bit_t *fifo_full, job_info_update_input_t* jiu_input,
                     fifo_update_input_t *fifo_input) {

#if COST_CALCULATOR_TOP
    //#pragma HLS INTERFACE m_axi port=new_job bundle=gmem0
    //#pragma HLS INTERFACE m_axi port=updated_ept bundle=gmem1
    //#pragma HLS INTERFACE m_axi port=updated_weight bundle=gmem2
    //#pragma HLS INTERFACE m_axi port=jiu_input bundle=gmem3
    //#pragma HLS INTERFACE m_axi port=fifo_full bundle=gmem4
#endif

    static proc_time_info_t ept_cam[NUM_MACHINES][JOBS_PER_MACHINE] = {0};
    static weight_info_t weight_cam[NUM_MACHINES][JOBS_PER_MACHINE] = {0};
    
    static proc_time_info_t new_ept_prev[NUM_MACHINES] = {0};
    static weight_info_t new_weight_prev[NUM_MACHINES] = {0};

    static memory_length_t job_address_stack_cc[NUM_MACHINES][JOBS_PER_MACHINE + 1] = {0};
    static one_bit_t reset[NUM_MACHINES] = {0};
    static memory_length_t head_pointer[NUM_MACHINES] = {0};
    static memory_length_t tail_pointer[NUM_MACHINES] = {0};

    uint8_t cost_jobs[NUM_MACHINES][JOBS_PER_MACHINE] = {0};
    one_bit_t cost_high_low_selector[NUM_MACHINES][JOBS_PER_MACHINE] = {0};
    uint32_t cost_machine[NUM_MACHINES] = {0};
    //uint8_t wspt_machine[NUM_MACHINES]  = {0};
    vf_index_t new_job_index[NUM_MACHINES] = {0};

    //#pragma HLS array_partition variable=ept_cam type=complete dim=0
    //#pragma HLS array_partition variable=weight_cam type=complete dim=0
    //#pragma HLS array_partition variable=new_ept_prev type=complete dim=0
    //#pragma HLS array_partition variable=new_weight_prev type=complete dim=0

    //#pragma HLS array_partition variable=job_address_stack type=complete dim=1
    //#pragma HLS array_partition variable=reset type=complete dim=0
    //#pragma HLS array_partition variable=head_pointer type=complete dim=0
    //#pragma HLS array_partition variable=tail_pointer type=complete dim=0

    //#pragma HLS array_partition variable=cost_jobs type=complete dim=0
    //#pragma HLS array_partition variable=cost_high_low_selector type=complete dim=0
    //#pragma HLS array_partition variable=cost_machine type=complete dim=0
    ////#pragma HLS array_partition variable=wspt_machine type=complete dim=0
    //#pragma HLS array_partition variable=new_job_index type=complete dim=0


#if COST_CALCULATOR_DEBUG
    printf("\033[31mIndividual Job cost calculator\n\033[0m");
#endif
    
    cost_calculator_all_machines(new_job.job_id, new_job.processing_time, new_job.weight,
                                updated_job_info, new_ept_prev, new_weight_prev,
                                job_address_stack_cc, reset, head_pointer, tail_pointer, ept_cam, 
                                weight_cam, cost_jobs, cost_high_low_selector, cost_machine, 
                                new_job_index);

#if COST_CALCULATOR_DEBUG
    printf("\033[31mCost comparator\n\033[0m");
#endif
    cost_comparator(new_job, cost_machine, new_job_index, fifo_full, 
                    jiu_input, fifo_input, new_ept_prev, new_weight_prev);

}
