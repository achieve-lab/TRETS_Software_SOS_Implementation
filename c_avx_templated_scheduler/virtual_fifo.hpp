/**
 * @file virtual_fifo.hpp
 * @brief Data structures, constants, and SIMD helpers for the Virtual FIFO.
 *
 * Fully generalized for dynamic JOBS_PER_MACHINE using AVX-512 strip-mining.
 */

#ifndef __VIRTUAL_FIFO__
#define __VIRTUAL_FIFO__

#include "data_types.hpp"   
#include "top_modules.hpp"  

#define VF_STREAM_IMPLEMENTATION 1

/* ── Data-selector encoding constants ────────────────────────────────────── */
#define W_DISABLE    (0)   
#define RD_SELECT    (1)   
#define LD_SELECT    (2)   
#define NEW_D_SELECT (3)   

/* ── Data structures ─────────────────────────────────────────────────────── */
typedef struct alignas(8) data_selector_input {
    job_id_t   new_job_id; 
    vf_index_t new_index;   
} data_selector_input_t;

/**
 * vf_machine_state_t
 * Arrays are bounded to JPM_PAD to safely accommodate ZMM load/store bounds.
 */
typedef struct alignas(64) vf_machine_state {
    CACHE_ALIGNED uint32_t fifo_regs_cur [JPM_PAD]; 
    CACHE_ALIGNED uint32_t fifo_regs_prev[JPM_PAD]; 
    uint32_t entry_count;                           
} vf_machine_state_t;

/* ── AVX-512 Helpers ─────────────────────────────────────────────────────── */

/**
 * compute_data_select_vec()
 * Upgraded to 16-lane AVX-512. Handles data selection for 16 FIFO slots per loop.
 */
SIMD_TARGET_AVX512
static inline __m512i compute_data_select_vec(uint32_t   new_job_id,
                                               uint32_t   fifo_idx_new,
                                               uint8_t    pop,
                                               __m512i    slot_indices)
{
    const __m512i v_new_idx   = _mm512_set1_epi32((int)fifo_idx_new);
    const __m512i v_new_idx_1 = _mm512_set1_epi32((int)(fifo_idx_new - 1));
    const __m512i v_zero      = _mm512_setzero_si512();

    __mmask16 lt_new   = _mm512_cmpgt_epi32_mask(v_new_idx,   slot_indices); 
    __mmask16 eq_new   = _mm512_cmpeq_epi32_mask(slot_indices, v_new_idx);   
    __mmask16 lt_new_1 = _mm512_cmpgt_epi32_mask(v_new_idx_1, slot_indices); 
    __mmask16 eq_new_1 = _mm512_cmpeq_epi32_mask(slot_indices, v_new_idx_1); 
    __mmask16 eq_zero_slot = _mm512_cmpeq_epi32_mask(slot_indices, v_zero);  

    const __m512i v_W   = _mm512_set1_epi32(W_DISABLE);
    const __m512i v_RD  = _mm512_set1_epi32(RD_SELECT);
    const __m512i v_LD  = _mm512_set1_epi32(LD_SELECT);
    const __m512i v_NEW = _mm512_set1_epi32(NEW_D_SELECT);

    __m512i result;

    if (new_job_id != INVALID_JOB_ID) {
        if (pop) {
            if (fifo_idx_new <= 1) {
                result = _mm512_mask_blend_epi32(eq_zero_slot, v_W, v_NEW);
            } else {
                result = v_W;                                              
                result = _mm512_mask_blend_epi32(lt_new_1, result, v_LD);    
                result = _mm512_mask_blend_epi32(eq_new_1, result, v_NEW);    
            }
        } else {
            result = v_RD;                                                 
            result = _mm512_mask_blend_epi32(lt_new, result, v_W);          
            result = _mm512_mask_blend_epi32(eq_new, result, v_NEW);          
        }
    } else {
        result = pop ? v_LD : v_W;
    }
    return result;
}

/**
 * fifo_reg_update()
 * Upgraded to 16-lane AVX-512.
 */
SIMD_TARGET_AVX512
static inline __m512i fifo_reg_update(
        __m512i prev,       
        __m512i left_nbr,   
        __m512i right_nbr,  
        __m512i ds,         
        __m512i new_job_vec)    
{
    __mmask16 is_LD  = _mm512_cmpeq_epi32_mask(ds, _mm512_set1_epi32(LD_SELECT));
    __mmask16 is_RD  = _mm512_cmpeq_epi32_mask(ds, _mm512_set1_epi32(RD_SELECT));
    __mmask16 is_NEW = _mm512_cmpeq_epi32_mask(ds, _mm512_set1_epi32(NEW_D_SELECT));

    __m512i result = prev;                                          
    result = _mm512_mask_blend_epi32(is_LD, result, left_nbr);         
    result = _mm512_mask_blend_epi32(is_RD, result, right_nbr);         
    result = _mm512_mask_blend_epi32(is_NEW, result, new_job_vec);        
    return result;
}

#endif /* __VIRTUAL_FIFO__ */