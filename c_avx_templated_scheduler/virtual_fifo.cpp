/**
 * @file virtual_fifo.cpp
 * @brief Sorted virtual FIFO implementation for the Stochastic Online Scheduler.
 *
 * Fully generalized for dynamic JOBS_PER_MACHINE using strip-mined loops.
 */

#include "virtual_fifo.hpp"
#include "top_modules.hpp"
#include <cstring>   

/* ── 1. data_selector_machine_simd (Chunked Loop) ────────────────────────── */
SIMD_TARGET_AVX512
static void data_selector_machine_simd(
        fifo_update_input_t   input,
        one_bit_t             pop_input,
        data_selector_t      *ds_out)          
{
    const __m512i base_indices = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

    for (int i = 0; i < JOBS_PER_MACHINE; i += AVX512_INT32_LANES) {
        int remain = JOBS_PER_MACHINE - i;
        __mmask16 valid_mask = (remain >= 16) ? 0xFFFF : ((1U << remain) - 1);

        __m512i slot_indices = _mm512_add_epi32(base_indices, _mm512_set1_epi32(i));

        __m512i v_ds = compute_data_select_vec(input.new_job_id,
                                                (uint32_t)input.fifo_index,
                                                pop_input,
                                                slot_indices);

        /* Safe scalar store auto-vectorized by compiler */
        CACHE_ALIGNED uint32_t ds32[16];
        _mm512_store_si512((__m512i *)ds32, v_ds);
        for (int k = 0; k < 16 && (i + k) < JOBS_PER_MACHINE; k++) {
            ds_out[i + k] = (data_selector_t)ds32[k];
        }
    }
}

SIMD_TARGET_AVX512
static void data_selector_all_machines(
        fifo_update_input_t  *input,
        one_bit_t            *pop_input,
        data_selector_t       data_select_signals[NUM_MACHINES][JPM_PAD],
        fifo_update_input_t  *input_temp,
        one_bit_t            *pop_input_temp)
{
    for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
        data_selector_machine_simd(input[m], pop_input[m], data_select_signals[m]);
        input_temp[m]    = input[m];
        pop_input_temp[m]= pop_input[m];
    }
}

/* ── 2. fifo_one_machine_simd (Chunked Loop) ─────────────────────────────── */
SIMD_TARGET_AVX512
static void fifo_one_machine_simd(
        job_id_t          new_job_id,
        one_bit_t         pop,
        data_selector_t  *ds,              
        vf_machine_state_t *state,         
        job_id_t         *top_job_id_out,
        job_id_t         *popped_job_id_out,
        one_bit_t        *full_out)
{
    uint32_t *cur  = state->fifo_regs_cur;
    uint32_t *prev = state->fifo_regs_prev;

    __m512i v_new = _mm512_set1_epi32((int)new_job_id);

    /* Initial right neighbour carry: lane 15 has INVALID_JOB_ID 
       so lane 0 of the first chunk naturally pulls INVALID_JOB_ID on shift */
    __m512i v_prev_chunk = _mm512_set_epi32(INVALID_JOB_ID, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0); 

    for (int i = 0; i < JOBS_PER_MACHINE; i += AVX512_INT32_LANES) {
        int remain = JOBS_PER_MACHINE - i;
        __mmask16 valid_mask = (remain >= 16) ? 0xFFFF : ((1U << remain) - 1);

        __m512i curr = _mm512_maskz_loadu_epi32(valid_mask, prev + i);

        /* Load next chunk for left neighbour (safe due to JPM_PAD padding) */
        __m512i next = _mm512_loadu_si512((const __m512i*)(prev + i + AVX512_INT32_LANES));
        
        /* Left neighbor: shift right by 1 element (crosses 512-bit boundaries safely) */
        __m512i v_left = _mm512_alignr_epi32(next, curr, 1);

        /* Right neighbor: shift right by 15 elements using state carried from previous chunk */
        __m512i v_right = _mm512_alignr_epi32(curr, v_prev_chunk, 15);

        __m128i xmm_ds = _mm_loadu_si128((const __m128i*)(ds + i));
        __m512i v_ds8  = _mm512_cvtepu8_epi32(xmm_ds);

        __m512i v_cur_new = fifo_reg_update(curr, v_left, v_right, v_ds8, v_new);

        /* Determine top_job_id from ds[0] on the FIRST chunk only */
        if (i == 0) {
            switch (ds[0]) {
                case W_DISABLE:    *top_job_id_out = prev[0];        break;
                case LD_SELECT:    *top_job_id_out = prev[1];        break;
                case NEW_D_SELECT: *top_job_id_out = new_job_id;     break;
                default:           *top_job_id_out = INVALID_JOB_ID; break;
            }
        }

        _mm512_mask_storeu_epi32(cur + i, valid_mask, v_cur_new);
        
        /* Save current chunk state to provide the right-neighbour boundary for the next chunk */
        v_prev_chunk = curr; 
    }

    if (new_job_id != INVALID_JOB_ID) state->entry_count += 1;

    if (pop) {
        *popped_job_id_out = prev[0];
        state->entry_count -= 1;
    } else {
        *popped_job_id_out = INVALID_JOB_ID;
    }

    *full_out = (state->entry_count == JOBS_PER_MACHINE) ? 1 : 0;

    /* fifo_read: copy cur → prev */
    for (int i = 0; i < JOBS_PER_MACHINE; i += AVX512_INT32_LANES) {
        int remain = JOBS_PER_MACHINE - i;
        __mmask16 valid_mask = (remain >= 16) ? 0xFFFF : ((1U << remain) - 1);
        __m512i v_c = _mm512_maskz_loadu_epi32(valid_mask, cur + i);
        _mm512_mask_storeu_epi32(prev + i, valid_mask, v_c);
    }
}

/* ── 3. fifo_all_machines / virtual_fifo ─────────────────────────────────── */
SIMD_TARGET_AVX512
static void fifo_all_machines(
        fifo_update_input_t *input,
        one_bit_t           *pop,
        data_selector_t      data_select_signals[NUM_MACHINES][JPM_PAD],
        job_id_t            *top_job_id,
        job_id_t            *popped_job_id,
        one_bit_t           *fifo_full)
{
    static CACHE_ALIGNED vf_machine_state_t vf_state[NUM_MACHINES];

    for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
        if (m + 1 < NUM_MACHINES)
            _mm_prefetch((const char *)&vf_state[m + 1], _MM_HINT_T0);

        fifo_one_machine_simd(
            input[m].new_job_id,
            pop[m],
            data_select_signals[m],
            &vf_state[m],
            &top_job_id[m],
            &popped_job_id[m],
            &fifo_full[m]);
    }
}

SIMD_TARGET_AVX512
void virtual_fifo(fifo_update_input_t *input,
                  one_bit_t           *pop,
                  job_id_t            *top_job_id,
                  job_id_t            *popped_job_id,
                  one_bit_t           *fifo_full)
{
    CACHE_ALIGNED data_selector_t data_selector_output[NUM_MACHINES][JPM_PAD];
    fifo_update_input_t input_temp[NUM_MACHINES];
    one_bit_t           pop_temp  [NUM_MACHINES];

    data_selector_all_machines(input, pop, data_selector_output, input_temp, pop_temp);

    fifo_all_machines(input_temp, pop_temp, data_selector_output,
                      top_job_id, popped_job_id, fifo_full);
}