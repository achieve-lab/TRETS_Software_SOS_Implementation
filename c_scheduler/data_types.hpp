#ifndef __DATA_TYPES__
#define __DATA_TYPES__

#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include <string.h>
#include <fstream>
#include <iostream>



/*--------Scheduler configuration-------*/
#define NUM_MACHINES 5
#define JOBS_PER_MACHINE 10
#define FIFO_POINTER_BIT_LENGTH 4 //Update this when updating jobs per 
#define INVALID_JOB_ID 0
#define INVALID_ADDRESS (JOBS_PER_MACHINE + 1)
#define STREAM_LENGTH 1
#define MEM_DATA_SIZE 100 //Number of jobs in the input memory from the host //CAUTION: DO NOT INCREASE THIS: INCREASING THIS WILL RESULT IN HUGE DESIGN SIZE WHICH IS UNDESIRABLE
#define UNROLL_FACTOR NUM_MACHINES
#define TOTAL_NUM_JOBS 10000

#define PRINT_RESET   "\x1b[0m"
#define PRINT_RED     "\x1b[31m"
#define PRINT_GREEN   "\x1b[32m"
#define PRINT_YELLOW  "\x1b[33m"
#define PRINT_BLUE    "\x1b[34m"
#define PRINT_MAGENTA "\x1b[35m"
#define PRINT_CYAN    "\x1b[36m"
#define PRINT_WHITE   "\x1b[37m"
#define PRINT_BLACK   "\x1b[30m"
#define PRINT_BOLD    "\x1b[1m"
#define PRINT_UNDERLINE "\x1b[4m"
#define PRINT_COLOR_END "\033[0m"
#define PRINT_FAILURE std::cout << PRINT_BOLD << PRINT_UNDERLINE << PRINT_RED << "Run Failed!!!!!!!!!" << std::endl << PRINT_RESET

/*-------Stack operations--------*/
#define PUSH 0
#define POP 1

/*-------Job Info Update operations--------*/
#define JI_UPDATE 0 //Update the processing time and weight
#define JI_INVALIDATE 1 //Invalidate the processing time and weight

/*-------Cost High low selector-----------*/
#define COST_HIGH 0
#define COST_LOW 1

/*--------Data structures for maintaing job information------*/

/*--------Job ID---------*/
typedef uint32_t job_id_t;
typedef uint8_t memory_length_t; //Update when number of jobs per machine is updated
typedef uint16_t machine_id_t;
typedef memory_length_t vf_index_t;
typedef uint8_t data_selector_t;
typedef uint8_t one_bit_t;

/*--------Expected processing time---------*/
typedef struct proc_time_info {
    job_id_t job_id;
    uint8_t proc_time;
} proc_time_info_t;

/*--------Expected weight and wspt---------*/
typedef struct weight_info {
    job_id_t job_id;
    uint8_t weight;
    uint8_t wspt;
} weight_info_t;

/*--------Alpha j---------*/
typedef struct alpha_j_info {
    job_id_t job_id;
    uint8_t alpha_j;
} alpha_j_info_t;

/*--------New job info--------*/
typedef struct job_info_update_input {
    job_id_t new_job_id;
    uint8_t alpha_j;
} job_info_update_input_t;

typedef struct fifo_update_input {
    job_id_t new_job_id;
    vf_index_t fifo_index;
} fifo_update_input_t;

typedef struct job_info_update_output {
    job_id_t job_id;
    uint8_t operation;
} job_info_update_output_t;

/* Struct to represent a new job coming into the scheduler*/
typedef struct job_in {
    job_id_t job_id;
    uint8_t weight;
    uint8_t alpha_j[NUM_MACHINES];
    uint8_t processing_time[NUM_MACHINES];
} job_in_t;

#endif
