/*
 * This file contains the implementation of virtual FIFO
 * Submodules in this file:
 *      ept_update
 *      weight_update
 *      alpha_j_update
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
memory_length_t stack_handler_jiu(memory_length_t *stack, one_bit_t *reset, memory_length_t *head,
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

stack_handler_reset_loop_jiu:
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
        /* Operation is pop */
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
 * This function implements the memory for alpha j details
 * and updates the alpha j of the top job
 * 
 * param[in] new_alpha_j  Alpha j time of new job
 * param[in] pop      Input pop value  
 */
void alpha_j_update_machine(job_info_update_input_t input, job_id_t top_job_id, 
                            alpha_j_info_t *alpha_j_cam, memory_length_t *job_address_stack,
                            one_bit_t *reset, memory_length_t *head_pointer, memory_length_t *tail_pointer,
                            job_info_update_output_t *output, one_bit_t *pop) {


    memory_length_t push_address = INVALID_ADDRESS;
    memory_length_t new_job_address = INVALID_ADDRESS;
    
    //update the alpha-j of top job
alpha_j_loop:
    for (memory_length_t index = 0; index < JOBS_PER_MACHINE; index++) {
        //#pragma HLS UNROLL
        if (top_job_id != INVALID_JOB_ID) {
            if (alpha_j_cam[index].job_id == top_job_id) {
                alpha_j_cam[index].alpha_j -= 1;
                output->job_id = top_job_id;
#if JIU_DEBUG
                printf("\033[34m Alpha-j: %u\033[0m\n", alpha_j_cam[index].alpha_j);
#endif
                if (alpha_j_cam[index].alpha_j == 0) {
                    alpha_j_cam[index].job_id = INVALID_JOB_ID;
                    *pop = 1;
                    output->operation = JI_INVALIDATE;
                    /* Push the invalid address to the stack */
                    push_address = index;
                    new_job_address = stack_handler_jiu(job_address_stack, reset, 
                                                        head_pointer, tail_pointer, PUSH, push_address);
                } else {
                    *pop = 0;
                    output->operation = JI_UPDATE;
                }
            }
        }
    }

    if (input.new_job_id != INVALID_JOB_ID) {
        //get the address from stack
        new_job_address = stack_handler_jiu(job_address_stack, reset, 
                                            head_pointer, tail_pointer, POP, push_address);

        if (new_job_address != INVALID_ADDRESS) {
            alpha_j_cam[new_job_address].job_id = input.new_job_id;
            alpha_j_cam[new_job_address].alpha_j = input.alpha_j;
        }
    }
}


/*
 * This function is the top module for job info update
 * 
 * param[in] input  Input to the job info update of type job_info_update_t
 * param[in] top_job_id    Job ID present at the top of the FIFO
 * 
 * param[out] output    Output indicating whether to update or invalidate the top job data
 * param[out] pop       Pop signal sent to virtual fifo
 */
void job_info_update(job_info_update_input_t *input, job_id_t *top_job_id, 
                     job_info_update_output_t *output,
                     one_bit_t *pop) {
#if JIU_TOP
    //#pragma HLS INTERFACE m_axi port=input bundle=gmem0
    //#pragma HLS INTERFACE m_axi port=top_job_id bundle=gmem1
    //#pragma HLS INTERFACE m_axi port=updated_ept bundle=gmem2
    //#pragma HLS INTERFACE m_axi port=updated_weight bundle=gmem3
    //#pragma HLS INTERFACE m_axi port=pop bundle=gmem4
#endif

    static alpha_j_info_t alpha_j_cam[NUM_MACHINES][JOBS_PER_MACHINE];
    static memory_length_t job_address_stack[NUM_MACHINES][JOBS_PER_MACHINE + 1];
    static one_bit_t reset[NUM_MACHINES];
    static memory_length_t head_pointer[NUM_MACHINES];
    static memory_length_t tail_pointer[NUM_MACHINES];

    //#pragma HLS array_partition variable=alpha_j_cam type=complete dim=0
    //#pragma HLS array_partition variable=job_address_stack type=complete dim=1
    //#pragma HLS array_partition variable=reset type=complete dim=0
    //#pragma HLS array_partition variable=head_pointer type=complete dim=0
    //#pragma HLS array_partition variable=tail_pointer type=complete dim=0

jiu_all_machine_loop:
    for (machine_id_t machine = 0; machine < NUM_MACHINES; machine++) {
        //#pragma HLS unroll
#if JIU_DEBUG
        printf("\033[34mHLS_PRINT Machine: %u\n\033[0m", machine);
#endif
        alpha_j_update_machine(input[machine], top_job_id[machine], alpha_j_cam[machine],
                               job_address_stack[machine], &reset[machine], 
                               &head_pointer[machine], &tail_pointer[machine], 
                               &output[machine], &pop[machine]);
    }

}