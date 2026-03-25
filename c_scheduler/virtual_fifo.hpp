#ifndef __VIRTUAL_FIFO__
#define __VIRTUAL_FIFO__

#include "data_types.hpp"
#define VF_STREAM_IMPLEMENTATION 1

/*-----Data selector------*/
#define W_DISABLE    (0) //write disable
#define RD_SELECT    (1) //right data select
#define LD_SELECT    (2) //left data select
#define NEW_D_SELECT (3) //new data select

/*-----Data structures used in Virtual fifo-----*/
typedef struct data_selector_input {
    job_id_t new_job_id;
    vf_index_t new_index;
} data_selector_input_t;


#endif