/**
 * @file data_types.hpp
 * @brief Core data type definitions for the Stochastic Online Scheduler.
 */

#ifndef __DATA_TYPES__
#define __DATA_TYPES__

#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include <string.h>
#include <fstream>
#include <iostream>
#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
#  define SIMD_ALIGNED   __attribute__((aligned(32)))
#  define CACHE_ALIGNED  __attribute__((aligned(64)))
#else
#  define SIMD_ALIGNED   __declspec(align(32))
#  define CACHE_ALIGNED  __declspec(align(64))
#endif

/* ── Configurable Scheduler Parameters ───────────────────────────────────── */
#ifndef NUM_MACHINES
#define NUM_MACHINES          5    
#endif

#ifndef JOBS_PER_MACHINE
#define JOBS_PER_MACHINE      10   
#endif

/* ── Vector Lane Widths ──────────────────────────────────────────────────── */
#define AVX2_INT32_LANES      8
#define AVX512_INT32_LANES    16

/* ── Dynamic Padding Macros (Round up to nearest AVX-512 boundary) ───────── */
#define MAC_PAD (((NUM_MACHINES + AVX512_INT32_LANES - 1) / AVX512_INT32_LANES) * AVX512_INT32_LANES)
#define JPM_PAD (((JOBS_PER_MACHINE + AVX512_INT32_LANES - 1) / AVX512_INT32_LANES) * AVX512_INT32_LANES)

/* * NOTE: MACHINE_LANE_MASK and JOBS_LANE_MASK are temporarily left here 
 * for compatibility with the un-chunked code, but will be phased out in Phase 2 
 * as single integer bitmasks cannot exceed 64 bits.
 */
#define MACHINE_LANE_MASK    ((1ULL << NUM_MACHINES) - 1ULL)   
#define JOBS_LANE_MASK       ((1ULL << JOBS_PER_MACHINE) - 1ULL) 

/* ── Remaining scheduler constants ───────────────────────────────────────── */
#define FIFO_POINTER_BIT_LENGTH  4 
#define INVALID_JOB_ID           0
#define INVALID_ADDRESS          (JOBS_PER_MACHINE + 1)
#define STREAM_LENGTH            1
#define MEM_DATA_SIZE            100
#define UNROLL_FACTOR            NUM_MACHINES
#define TOTAL_NUM_JOBS           10000

/* ── Terminal colour macros (unchanged) ─────────────────────────────────── */
#define PRINT_RESET     "\x1b[0m"
#define PRINT_RED       "\x1b[31m"
#define PRINT_GREEN     "\x1b[32m"
#define PRINT_YELLOW    "\x1b[33m"
#define PRINT_BLUE      "\x1b[34m"
#define PRINT_MAGENTA   "\x1b[35m"
#define PRINT_CYAN      "\x1b[36m"
#define PRINT_WHITE     "\x1b[37m"
#define PRINT_BLACK     "\x1b[30m"
#define PRINT_BOLD      "\x1b[1m"
#define PRINT_UNDERLINE "\x1b[4m"
#define PRINT_COLOR_END "\033[0m"
#define PRINT_FAILURE \
    std::cout << PRINT_BOLD << PRINT_UNDERLINE << PRINT_RED \
              << "Run Failed!!!!!!!!!" << std::endl << PRINT_RESET

/* ──  Op Codes  ───────────────────────────────────────────────────────────── */
#define PUSH           0
#define POP            1
#define JI_UPDATE      0
#define JI_INVALIDATE  1
#define COST_HIGH      0
#define COST_LOW       1

/* ── Scalar typedefs ─────────────────────────────────────────────────────── */
typedef uint32_t   job_id_t;
typedef uint8_t    memory_length_t;
typedef uint16_t   machine_id_t;
typedef memory_length_t vf_index_t;
typedef uint8_t    data_selector_t;
typedef uint8_t    one_bit_t;

/* ── Core data structures ────────────────────────────────────────────────── */

typedef struct alignas(8) proc_time_info {
    job_id_t job_id;   
    uint8_t  proc_time;
} proc_time_info_t;

typedef struct alignas(8) weight_info {
    job_id_t job_id;  
    uint8_t  weight;  
    uint8_t  wspt;    
} weight_info_t;

typedef struct alignas(8) alpha_j_info {
    job_id_t job_id;  
    uint8_t  alpha_j; 
} alpha_j_info_t;

typedef struct alignas(8) job_info_update_input {
    job_id_t new_job_id; 
    uint8_t  alpha_j;    
} job_info_update_input_t;

typedef struct alignas(8) fifo_update_input {
    job_id_t   new_job_id;  
    vf_index_t fifo_index; 
} fifo_update_input_t;

typedef struct alignas(8) job_info_update_output {
    job_id_t job_id;    
    uint8_t  operation; 
} job_info_update_output_t;

/**
 * job_in_t – a new job arriving at the scheduler.
 * * Arrays inside are sized to MAC_PAD rather than NUM_MACHINES. 
 * This ensures that a 512-bit vector load targeting the tail end of the 
 * machine array will safely read zeros rather than segfaulting on adjacent memory.
 */
typedef struct alignas(64) job_in {
    job_id_t job_id;                       
    uint8_t  weight;                       
    uint8_t  alpha_j[MAC_PAD];             
    uint8_t  processing_time[MAC_PAD];
} job_in_t;

#endif /* __DATA_TYPES__ */