#ifndef __EXTM_DATA_TYPES__
#define __EXTM_DATA_TYPES__

#ifndef _DATA_TYPES_
#include "data_types.hpp"
#endif

#include <stdio.h>
#include <stdint.h>

#include <cstring>
#include <string.h>
#include "data_types.hpp"

/*--------JOB ID MANAGERS---------*/

#define MANAGER_SIZE (MEM_DATA_SIZE + 10) //Ensure theres enough IDs for all spots in fifos, plus more to account for jobs actively in pipeline

/*--------TB New Job Info---------*/
//17 bits allows tick to go up to 131071

//currently just one machine profile
typedef struct new_job_data_host {
    job_in_t job_data;
    uint32_t release_tick;
} new_job_data_host_t;

typedef struct scheduler_interface_input {
    new_job_data_host_t new_job_table[MEM_DATA_SIZE];
    uint32_t initial_tick;
} scheduler_interface_input_t;

typedef struct perf_measurement_info {
    uint32_t popped_tick;
    machine_id_t machine_scheduled;
} perf_measurement_info_t;

typedef struct scheduler_interface_output {
    perf_measurement_info_t scheduled_jobs[MANAGER_SIZE];
    uint16_t num_jobs[NUM_MACHINES];
    uint32_t final_tick;
} scheduler_interface_output_t;

typedef struct job_id_manager{
    //can find much smaller way of doing this
    //For now just get it done though I guess

    //In essence, avail_ids is circular buffer fifo, gives unused index in release tick manager as needed
    uint32_t release_tick[MANAGER_SIZE]; //gives us exactly as many indexes as we can point at with our amount of job ids
    job_id_t avail_ids[(MANAGER_SIZE - 1)];
    job_id_t head;
    job_id_t tail;

    job_id_manager(){
        head = 0;
        tail = 0;
        for (job_id_t i = 0; i < (MANAGER_SIZE - 1); i++){
            avail_ids[i] = i+1; //can't use 0, that's reserved as invalid id. Would be nice if we could do processing time instead
        }
    }

    //internal, pop means take the id at the front of the buffer
    job_id_t pop(){
        job_id_t id = avail_ids[head];
        head++;

        if (head > (MANAGER_SIZE - 2)){
            head = 0;
        } 

        return id;
    }

    void reset() {
        head = 0;
        tail = 0;
        for (job_id_t i = 0; i < (MANAGER_SIZE - 1); i++){
            avail_ids[i] = i+1; //can't use 0, that's reserved as invalid id. Would be nice if we could do processing time instead
        }
    }

    void push(job_id_t id){
        this->avail_ids[tail] = id;
        tail++;

        if (tail > (MANAGER_SIZE - 2)){
            tail = 0;
        } 
    }

    job_id_t assign_id(uint32_t released){
        job_id_t id = this->pop();
        this->release_tick[id] = released;
        return id;
    }

    uint32_t retrieve_id(job_id_t id){
        this->push(id);
        return release_tick[id];
    }

} job_id_manager;



#endif
