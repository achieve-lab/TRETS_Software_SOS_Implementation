/**
 * @file extm_data_types.hpp
 * @brief External-memory interface data types for the Stochastic Online Scheduler.
 */

#ifndef __EXTM_DATA_TYPES__
#define __EXTM_DATA_TYPES__

#include "data_types.hpp"
#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include <string.h>

/* ── JOB ID MANAGER size ─────────────────────────────────────────────────── */
#define MANAGER_SIZE_RAW   (MEM_DATA_SIZE + 10)
#define MANAGER_SIZE       (((MANAGER_SIZE_RAW + AVX512_INT32_LANES - 1) \
                             / AVX512_INT32_LANES) * AVX512_INT32_LANES)

/* ── new_job_data_host_t ─────────────────────────────────────────────────── */
/**
 * Compiler automatically pads the end of this struct to hit the 64-byte 
 * alignment boundary requested by alignas(64).
 */
typedef struct alignas(64) new_job_data_host {
    job_in_t  job_data;        
    uint32_t  release_tick;    
} new_job_data_host_t;

/* ── scheduler_interface_input_t ─────────────────────────────────────────── */
typedef struct alignas(64) scheduler_interface_input {
    CACHE_ALIGNED new_job_data_host_t new_job_table[MEM_DATA_SIZE];
    uint32_t initial_tick;
} scheduler_interface_input_t;

/* ── perf_measurement_info_t ─────────────────────────────────────────────── */
typedef struct alignas(8) perf_measurement_info {
    uint32_t     popped_tick;        
    machine_id_t machine_scheduled;
} perf_measurement_info_t;

/* ── scheduler_interface_output_t ────────────────────────────────────────── */
/**
 * num_jobs[] is now dynamically padded to MAC_PAD so horizontal vector 
 * reductions won't read out of bounds.
 */
typedef struct alignas(64) scheduler_interface_output {
    CACHE_ALIGNED perf_measurement_info_t scheduled_jobs[MANAGER_SIZE];

    /* Padded to nearest SIMD boundary for chunked horizontal reduction */
    CACHE_ALIGNED uint16_t num_jobs[MAC_PAD]; 

    uint32_t final_tick;
} scheduler_interface_output_t;

/* ── job_id_manager (Implementation kept identical to original) ──────────── */
typedef struct alignas(64) job_id_manager {
    CACHE_ALIGNED uint32_t release_tick[MANAGER_SIZE];
    CACHE_ALIGNED uint32_t avail_ids[MANAGER_SIZE]; 

    job_id_t head;
    job_id_t tail;

    job_id_manager() {
        head = 0;
        tail = 0;
        reset_simd();
    }

    job_id_t pop() {
        job_id_t id = avail_ids[head];
        head++;
        if (head > (MANAGER_SIZE - 2)) head = 0;
        return id;
    }

    void push(job_id_t id) {
        avail_ids[tail] = id;
        tail++;
        if (tail > (MANAGER_SIZE - 2)) tail = 0;
    }

    job_id_t assign_id(uint32_t released) {
        job_id_t id = this->pop();
        this->release_tick[id] = released;
        return id;
    }

    uint32_t retrieve_id(job_id_t id) {
        this->push(id);
        return release_tick[id];
    }

    void reset() {
        head = 0;
        tail = 0;
        for (job_id_t i = 0; i < (MANAGER_SIZE - 1); i++) {
            avail_ids[i] = i + 1;
        }
    }

    void reset_simd() {
        head = 0;
        tail = 0;

        const __m512i seq = _mm512_set_epi32(15,14,13,12,11,10,9,8, 7, 6, 5, 4, 3, 2,1,0);
        const __m512i one = _mm512_set1_epi32(1);

        const int total   = MANAGER_SIZE - 1;  
        const int full    = total / AVX512_INT32_LANES;
        const int remain  = total % AVX512_INT32_LANES;

        for (int chunk = 0; chunk < full; chunk++) {
            __m512i offset = _mm512_set1_epi32(chunk * AVX512_INT32_LANES);
            __m512i vals   = _mm512_add_epi32(_mm512_add_epi32(seq, offset), one);
            _mm512_store_si512((__m512i *)(avail_ids + chunk * AVX512_INT32_LANES), vals);
        }

        if (remain > 0) {
            __mmask16 tail_mask = (__mmask16)((1u << remain) - 1u);
            __m512i offset = _mm512_set1_epi32(full * AVX512_INT32_LANES);
            __m512i vals   = _mm512_add_epi32(_mm512_add_epi32(seq, offset), one);
            _mm512_mask_store_epi32(avail_ids + full * AVX512_INT32_LANES, tail_mask, vals);
        }
    }
} job_id_manager;

static inline void zero_output_simd(scheduler_interface_output_t *out) {
    const __m512i zero512 = _mm512_setzero_si512();

    const size_t jobs_bytes = sizeof(perf_measurement_info_t) * MANAGER_SIZE;
    uint8_t *base = reinterpret_cast<uint8_t *>(out->scheduled_jobs);
    for (size_t off = 0; off < jobs_bytes; off += 64) {
        _mm512_stream_si512((__m512i *)(base + off), zero512);
    }

    /* Update to zero out the dynamically padded MAC_PAD length */
    for (size_t off = 0; off < MAC_PAD; off += 16) {
        _mm256_storeu_si256((__m256i *)(out->num_jobs + off), _mm256_setzero_si256());
    }
    
    out->final_tick = 0;
}

#endif /* __EXTM_DATA_TYPES__ */