/**
 * @file cost_calculator.cpp
 * @brief WSPT-based cost calculator for the Stochastic Online Scheduler.
 *
 * Fully generalized for dynamic NUM_MACHINES and JOBS_PER_MACHINE.
 */

#include "data_types.hpp"
#include "top_modules.hpp"
#include <cstdio>
#include <cstdlib>

/* ── 1. stack_handler_cc (unchanged scalar) ──────────────────────────────── */
static memory_length_t stack_handler_cc(memory_length_t *stack,
                                         one_bit_t       *reset,
                                         memory_length_t *head,
                                         memory_length_t *tail,
                                         one_bit_t        operation,
                                         memory_length_t  push_address)
{
    memory_length_t new_job_address = INVALID_ADDRESS;

    if (*reset == 0) {
        *head = 0;
        *tail = JOBS_PER_MACHINE;
        for (memory_length_t i = 0; i < JOBS_PER_MACHINE; i++)
            stack[i] = i;
        *reset = 1;
    }

    if (operation == PUSH) {
        if (*tail == JOBS_PER_MACHINE) {
            if (*head == 0) new_job_address = INVALID_ADDRESS;          
            else {
                stack[*tail]  = push_address;
                *tail         = 0;
                new_job_address = stack[*head];
            }
        } else {
            if ((*tail + 1) == *head) new_job_address = INVALID_ADDRESS;          
            else {
                stack[*tail]  = push_address;
                *tail        += 1;
                new_job_address = stack[*head];
            }
        }
    } else {                                                
        if (*head == *tail) new_job_address = INVALID_ADDRESS;              
        else {
            new_job_address = stack[*head];
            *head = (*head == JOBS_PER_MACHINE) ? 0 : *head + 1;
        }
    }
    return new_job_address;
}

/* ── 2. Unpack/Pack helpers (updated boundaries) ─────────────────────────── */
SIMD_TARGET_AVX512
static inline void unpack_ept_cam(const proc_time_info_t *ept,
                                   uint32_t * __restrict__ job_ids,   
                                   uint32_t * __restrict__ proc_times)
{
    for (int j = 0; j < JOBS_PER_MACHINE; j++) {
        job_ids[j]   = ept[j].job_id;
        proc_times[j]= ept[j].proc_time;
    }
    for (int j = JOBS_PER_MACHINE; j < JPM_PAD; j++) {
        job_ids[j]   = INVALID_JOB_ID;
        proc_times[j]= 0;
    }
}

SIMD_TARGET_AVX512
static inline void unpack_weight_cam(const weight_info_t *wt,
                                      uint32_t * __restrict__ job_ids, 
                                      uint32_t * __restrict__ weights, 
                                      uint32_t * __restrict__ wspts)   
{
    for (int j = 0; j < JOBS_PER_MACHINE; j++) {
        job_ids[j] = wt[j].job_id;
        weights[j] = wt[j].weight;
        wspts[j]   = wt[j].wspt;
    }
    for (int j = JOBS_PER_MACHINE; j < JPM_PAD; j++) {
        job_ids[j] = INVALID_JOB_ID;
        weights[j] = 0;
        wspts[j]   = 0;
    }
}

SIMD_TARGET_AVX512
static inline void pack_ept_cam(proc_time_info_t       *ept,
                                 const uint32_t * __restrict__ job_ids,
                                 const uint32_t * __restrict__ proc_times)
{
    for (int j = 0; j < JOBS_PER_MACHINE; j++) {
        ept[j].job_id    = job_ids[j];
        ept[j].proc_time = (uint8_t)proc_times[j];
    }
}

SIMD_TARGET_AVX512
static inline void pack_weight_cam(weight_info_t          *wt,
                                    const uint32_t * __restrict__ job_ids,
                                    const uint32_t * __restrict__ weights,
                                    const uint32_t * __restrict__ wspts)
{
    for (int j = 0; j < JOBS_PER_MACHINE; j++) {
        wt[j].job_id = job_ids[j];
        wt[j].weight = (uint8_t)weights[j];
        wt[j].wspt   = (uint8_t)wspts[j];
    }
}

/* ── 3. cost_calculator_machine_simd (Chunked Loop) ──────────────────────── */
SIMD_TARGET_AVX512
static void cost_calculator_machine_simd(
        job_id_t                  new_job_id,
        uint8_t                   new_ept_curr,
        uint8_t                   new_weight_curr,
        job_info_update_output_t  update_job_info,
        proc_time_info_t          new_ept_prev,
        weight_info_t             new_weight_prev,
        memory_length_t          *job_address_stack,
        one_bit_t                *reset,
        memory_length_t          *head_pointer,
        memory_length_t          *tail_pointer,
        proc_time_info_t         *ept_machine,
        weight_info_t            *weight_machine,
        uint32_t                 *cost_machine_out,
        vf_index_t               *new_job_index_out)
{
    CACHE_ALIGNED uint32_t ept_job_ids  [JPM_PAD];
    CACHE_ALIGNED uint32_t proc_times   [JPM_PAD];
    CACHE_ALIGNED uint32_t wt_job_ids   [JPM_PAD];
    CACHE_ALIGNED uint32_t weights      [JPM_PAD];
    CACHE_ALIGNED uint32_t wspts        [JPM_PAD];

    if (new_ept_prev.job_id != INVALID_JOB_ID) {
        memory_length_t addr = stack_handler_cc(job_address_stack, reset,
                                                 head_pointer, tail_pointer,
                                                 POP, INVALID_ADDRESS);
        if (addr != INVALID_ADDRESS) {
            ept_machine[addr].job_id    = new_ept_prev.job_id;
            ept_machine[addr].proc_time = new_ept_prev.proc_time;
            weight_machine[addr].job_id = new_weight_prev.job_id;
            weight_machine[addr].weight = new_weight_prev.weight;
            weight_machine[addr].wspt   = new_weight_prev.wspt;
        }
    }

    uint8_t wspt_new = (new_job_id != INVALID_JOB_ID && new_ept_curr != 0)
                       ? (new_weight_curr / new_ept_curr) : 0;

    unpack_ept_cam   (ept_machine,    ept_job_ids, proc_times);
    unpack_weight_cam(weight_machine, wt_job_ids,  weights, wspts);

    uint32_t cost_high_wspt = 0;
    uint32_t cost_low_wspt  = 0;
    uint32_t n_high         = 0;

    __m512i v_target   = _mm512_set1_epi32((int)update_job_info.job_id);
    __m512i v_one      = _mm512_set1_epi32(1);
    __m512i v_wspt_new = _mm512_set1_epi32((int)wspt_new);

    for (int i = 0; i < JOBS_PER_MACHINE; i += AVX512_INT32_LANES) {
        int remain = JOBS_PER_MACHINE - i;
        __mmask16 valid_mask = (remain >= 16) ? 0xFFFF : ((1U << remain) - 1);

        /* Pass A: Job-info update */
        if (update_job_info.job_id != INVALID_JOB_ID) {
            __m512i v_ept_ids = _mm512_maskz_loadu_epi32(valid_mask, ept_job_ids + i);
            __mmask16 hit_mask = _mm512_mask_cmpeq_epi32_mask(valid_mask, v_ept_ids, v_target);

            if (hit_mask) {
                if (update_job_info.operation == JI_UPDATE) {
                    __m512i v_pt = _mm512_maskz_loadu_epi32(valid_mask, proc_times + i);
                    v_pt = _mm512_mask_sub_epi32(v_pt, hit_mask, v_pt, v_one);
                    _mm512_mask_storeu_epi32(proc_times + i, valid_mask, v_pt);

                    __m512i v_wt   = _mm512_maskz_loadu_epi32(valid_mask, weights + i);
                    __m512i v_wspt = _mm512_maskz_loadu_epi32(valid_mask, wspts + i);
                    v_wt = _mm512_mask_sub_epi32(v_wt, hit_mask, v_wt, v_wspt);
                    _mm512_mask_storeu_epi32(weights + i, valid_mask, v_wt);
                } else {
                    int hit_idx = i + __builtin_ctz((unsigned)hit_mask);
                    proc_times[hit_idx]   = 0;
                    weights   [hit_idx]   = 0;
                    wspts     [hit_idx]   = 0;
                    ept_job_ids [hit_idx] = INVALID_JOB_ID;
                    wt_job_ids  [hit_idx] = INVALID_JOB_ID;
                    stack_handler_cc(job_address_stack, reset, head_pointer, tail_pointer,
                                     PUSH, (memory_length_t)hit_idx);
                }
                pack_ept_cam   (ept_machine,    ept_job_ids, proc_times);
                pack_weight_cam(weight_machine, wt_job_ids,  weights, wspts);
            }
        }

        /* Pass B: Cost selection */
        __m512i v_pt   = _mm512_maskz_loadu_epi32(valid_mask, proc_times + i);
        __m512i v_wt   = _mm512_maskz_loadu_epi32(valid_mask, weights + i);
        __m512i v_wspt = _mm512_maskz_loadu_epi32(valid_mask, wspts + i);

        __mmask16 high_mask = _mm512_mask_cmpgt_epi32_mask(valid_mask, v_wspt, v_wspt_new);
        __mmask16 low_mask  = (~high_mask) & valid_mask;

        cost_high_wspt += _mm512_mask_reduce_add_epi32(high_mask, v_pt);
        cost_low_wspt  += _mm512_mask_reduce_add_epi32(low_mask,  v_wt);
        n_high         += __builtin_popcount((unsigned)high_mask);
    }

    *cost_machine_out  = (uint32_t)new_weight_curr *
                         ((uint32_t)new_ept_curr + cost_high_wspt)
                       + (uint32_t)new_ept_curr * cost_low_wspt;

    *new_job_index_out = (vf_index_t)n_high;
}

/* ── 4. cost_calculator_all_machines ─────────────────────────────────────── */
SIMD_TARGET_AVX512
static void cost_calculator_all_machines(
        job_id_t                  new_job_id,
        uint8_t                  *new_job_processing_time,
        uint8_t                   new_job_weight,
        job_info_update_output_t *update_job_info,
        proc_time_info_t         *new_ept_prev,
        weight_info_t            *new_weight_prev,
        memory_length_t           job_address_stack[MAC_PAD][JPM_PAD + 1],
        one_bit_t                *reset_arr,
        memory_length_t          *head_pointer,
        memory_length_t          *tail_pointer,
        proc_time_info_t          ept_cam[MAC_PAD][JPM_PAD],
        weight_info_t             weight_cam[MAC_PAD][JPM_PAD],
        uint32_t                 *cost_machine,
        vf_index_t               *new_job_index)
{
    for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
        if (m + 1 < NUM_MACHINES) {
            _mm_prefetch((const char *)ept_cam   [m + 1], _MM_HINT_T0);
            _mm_prefetch((const char *)weight_cam[m + 1], _MM_HINT_T0);
        }

        cost_calculator_machine_simd(
            new_job_id,
            new_job_processing_time[m],
            new_job_weight,
            update_job_info[m],
            new_ept_prev[m],
            new_weight_prev[m],
            job_address_stack[m],
            &reset_arr[m],
            &head_pointer[m],
            &tail_pointer[m],
            ept_cam[m],
            weight_cam[m],
            &cost_machine[m],
            &new_job_index[m]);
    }
}

/* ── 5. cost_comparator (Chunked horizontal reduction over MAC_PAD) ──────── */
SIMD_TARGET_AVX512
static void cost_comparator(job_in_t                  new_job,
                              uint32_t                 *cost,
                              vf_index_t               *index,
                              one_bit_t                *fifo_full,
                              job_info_update_input_t  *jiu_input,
                              fifo_update_input_t      *fifo_input,
                              proc_time_info_t         *new_ept,
                              weight_info_t            *new_weight)
{
    machine_id_t low_cost_machine    = 0;
    vf_index_t   low_cost_index      = JOBS_PER_MACHINE;
    uint32_t     low_cost            = UINT32_MAX;

    static machine_id_t all_cost_equal_machine_id = 0;

    /* Safe scalar zeroing, easily autovectorized by the compiler */
    for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
        jiu_input[m].new_job_id = INVALID_JOB_ID;
        jiu_input[m].alpha_j    = 0;
        new_ept[m].job_id       = INVALID_JOB_ID;
        new_ept[m].proc_time    = 0;
        new_weight[m].job_id    = INVALID_JOB_ID;
        new_weight[m].weight    = 0;
        new_weight[m].wspt      = 0;
        fifo_input[m].new_job_id  = INVALID_JOB_ID;
        fifo_input[m].fifo_index  = JOBS_PER_MACHINE;
    }

    if (new_job.job_id == INVALID_JOB_ID) {
        return;
    }

    __m512i v_global_min = _mm512_set1_epi32(UINT32_MAX);

    for (int i = 0; i < NUM_MACHINES; i += AVX512_INT32_LANES) {
        int remain = NUM_MACHINES - i;
        __mmask16 valid_mask = (remain >= 16) ? 0xFFFF : ((1U << remain) - 1);

        /* Convert 8-bit fifo_full to 32-bit mask */
        __m128i v_full_8 = _mm_maskz_loadu_epi8(valid_mask, fifo_full + i);
        __m512i v_full_32 = _mm512_cvtepu8_epi32(v_full_8);
        __mmask16 full_mask = _mm512_cmpneq_epi32_mask(v_full_32, _mm512_setzero_si512());

        __m512i v_cost = _mm512_maskz_loadu_epi32(valid_mask, cost + i);
        
        /* Machines that are invalid OR full get UINT32_MAX cost */
        __mmask16 invalid_or_full = (~valid_mask) | full_mask;
        v_cost = _mm512_mask_blend_epi32(invalid_or_full, v_cost, _mm512_set1_epi32(UINT32_MAX));

        v_global_min = _mm512_min_epu32(v_global_min, v_cost);
    }

    /* Hardware horizontal min across the running total */
    low_cost = _mm512_reduce_min_epu32(v_global_min);

    uint32_t num_equal = 0;
    for (int m = 0; m < NUM_MACHINES; m++) {
        if (!fifo_full[m] && cost[m] == low_cost) {
            if (num_equal == 0) {
                low_cost_machine = (machine_id_t)m;
                low_cost_index   = index[m];
            }
            num_equal++;
        }
    }

    if (num_equal == NUM_MACHINES) {
        low_cost_machine = all_cost_equal_machine_id;
        low_cost_index   = index[all_cost_equal_machine_id];
        all_cost_equal_machine_id++;
        if (all_cost_equal_machine_id == NUM_MACHINES)
            all_cost_equal_machine_id = 0;
    }

    jiu_input[low_cost_machine].new_job_id  = new_job.job_id;
    jiu_input[low_cost_machine].alpha_j     = new_job.alpha_j[low_cost_machine];

    new_ept[low_cost_machine].job_id        = new_job.job_id;
    new_ept[low_cost_machine].proc_time     = new_job.processing_time[low_cost_machine];

    new_weight[low_cost_machine].job_id     = new_job.job_id;
    new_weight[low_cost_machine].weight     = new_job.weight;
    new_weight[low_cost_machine].wspt       =
        (new_job.processing_time[low_cost_machine] != 0)
        ? new_job.weight / new_job.processing_time[low_cost_machine]
        : 0;

    fifo_input[low_cost_machine].fifo_index = low_cost_index;
    fifo_input[low_cost_machine].new_job_id = new_job.job_id;
}

/* ── 6. cost_calculator ──────────────────────────────────────────────────── */
SIMD_TARGET_AVX512
void cost_calculator(job_in_t                  new_job,
                     job_info_update_output_t *updated_job_info,
                     one_bit_t                *fifo_full,
                     job_info_update_input_t  *jiu_input,
                     fifo_update_input_t      *fifo_input)
{
    static CACHE_ALIGNED proc_time_info_t ept_cam   [MAC_PAD][JPM_PAD];
    static CACHE_ALIGNED weight_info_t    weight_cam[MAC_PAD][JPM_PAD];

    static CACHE_ALIGNED proc_time_info_t new_ept_prev   [MAC_PAD];
    static CACHE_ALIGNED weight_info_t    new_weight_prev[MAC_PAD];

    static CACHE_ALIGNED memory_length_t  job_address_stack_cc[MAC_PAD][JPM_PAD + 1];
    static one_bit_t       reset_arr    [MAC_PAD];
    static memory_length_t head_pointer [MAC_PAD];
    static memory_length_t tail_pointer [MAC_PAD];

    CACHE_ALIGNED uint32_t   cost_machine [MAC_PAD];
    CACHE_ALIGNED vf_index_t new_job_index[MAC_PAD];

    cost_calculator_all_machines(
        new_job.job_id,
        new_job.processing_time,
        new_job.weight,
        updated_job_info,
        new_ept_prev,
        new_weight_prev,
        job_address_stack_cc,
        reset_arr,
        head_pointer,
        tail_pointer,
        ept_cam,
        weight_cam,
        cost_machine,
        new_job_index);

    cost_comparator(
        new_job,
        cost_machine,
        new_job_index,
        fifo_full,
        jiu_input,
        fifo_input,
        new_ept_prev,
        new_weight_prev);
}