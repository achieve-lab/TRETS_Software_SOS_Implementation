/**
 * @file job_info_update.cpp
 * @brief Alpha-j counter management for the Stochastic Online Scheduler.
 *
 * Fully generalized for dynamic JOBS_PER_MACHINE using strip-mined loops.
 */

#include "data_types.hpp"
#include "top_modules.hpp"
#include <cstdio>

/* ── 1. stack_handler_jiu (unchanged scalar logic) ───────────────────────── */
static memory_length_t stack_handler_jiu(memory_length_t *stack,
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
            if (*head == 0) {
                new_job_address = INVALID_ADDRESS;              
            } else {
                stack[*tail]    = push_address;
                *tail           = 0;
                new_job_address = stack[*head];
            }
        } else {
            if ((*tail + 1) == *head) {
                new_job_address = INVALID_ADDRESS;              
            } else {
                stack[*tail]    = push_address;
                (*tail)        += 1;
                new_job_address = stack[*head];
            }
        }
    } else {                                                    
        if (*head == *tail) {
            new_job_address = INVALID_ADDRESS;                  
        } else {
            new_job_address = stack[*head];
            *head = (*head == JOBS_PER_MACHINE) ? 0 : *head + 1;
        }
    }
    return new_job_address;
}

/* ── 2. Unpack / pack helpers for alpha_j_cam ────────────────────────────── */
SIMD_TARGET_AVX512
static inline void unpack_alpha_j_cam(
        const alpha_j_info_t * __restrict__ cam,
        uint32_t             * __restrict__ job_ids,   /* [JPM_PAD] */
        uint32_t             * __restrict__ alpha_js)  /* [JPM_PAD] */
{
    for (int j = 0; j < JOBS_PER_MACHINE; j++) {
        job_ids [j] = cam[j].job_id;
        alpha_js[j] = (uint32_t)cam[j].alpha_j;
    }
    for (int j = JOBS_PER_MACHINE; j < JPM_PAD; j++) {
        job_ids [j] = INVALID_JOB_ID;
        alpha_js[j] = 0;
    }
}

SIMD_TARGET_AVX512
static inline void pack_alpha_j_cam(
        alpha_j_info_t       * __restrict__ cam,
        const uint32_t       * __restrict__ job_ids,
        const uint32_t       * __restrict__ alpha_js)
{
    for (int j = 0; j < JOBS_PER_MACHINE; j++) {
        cam[j].job_id  = job_ids [j];
        cam[j].alpha_j = (uint8_t)alpha_js[j];
    }
}

/* ── 3. alpha_j_update_machine_simd (Chunked Loop Version) ───────────────── */
SIMD_TARGET_AVX512
static void alpha_j_update_machine_simd(
        job_info_update_input_t   input,
        job_id_t                  top_job_id,
        alpha_j_info_t           *alpha_j_cam,
        memory_length_t          *job_address_stack,
        one_bit_t                *reset,
        memory_length_t          *head_pointer,
        memory_length_t          *tail_pointer,
        job_info_update_output_t *output,
        one_bit_t                *pop)
{
    CACHE_ALIGNED uint32_t job_ids [JPM_PAD];
    CACHE_ALIGNED uint32_t alpha_js[JPM_PAD];

    unpack_alpha_j_cam(alpha_j_cam, job_ids, alpha_js);
    *pop = 0;

    if (top_job_id != INVALID_JOB_ID) {
        __m512i zmm_target = _mm512_set1_epi32((int)top_job_id);
        __m512i zmm_one    = _mm512_set1_epi32(1);
        __m512i zmm_zero   = _mm512_setzero_si512();

        /* Strip-mined loop over JOBS_PER_MACHINE */
        for (int i = 0; i < JOBS_PER_MACHINE; i += AVX512_INT32_LANES) {
            int remain = JOBS_PER_MACHINE - i;
            __mmask16 valid_mask = (remain >= 16) ? 0xFFFF : ((1U << remain) - 1);

            __m512i zmm_ids = _mm512_maskz_loadu_epi32(valid_mask, job_ids + i);
            __mmask16 hit_mask = _mm512_mask_cmpeq_epi32_mask(valid_mask, zmm_ids, zmm_target);

            if (hit_mask) {
                __m512i zmm_aj = _mm512_maskz_loadu_epi32(valid_mask, alpha_js + i);
                zmm_aj = _mm512_mask_sub_epi32(zmm_aj, hit_mask, zmm_aj, zmm_one);

                __mmask16 zero_mask = _mm512_mask_cmpeq_epi32_mask(hit_mask, zmm_aj, zmm_zero);
                _mm512_mask_storeu_epi32(alpha_js + i, valid_mask, zmm_aj);

                output->job_id = top_job_id;

                if (zero_mask) {
                    /* Hit index offset by chunk position `i` */
                    int hit_idx = i + __builtin_ctz((unsigned)hit_mask);

                    job_ids [hit_idx] = INVALID_JOB_ID;
                    alpha_js[hit_idx] = 0;

                    stack_handler_jiu(job_address_stack, reset,
                                       head_pointer, tail_pointer,
                                       PUSH, (memory_length_t)hit_idx);

                    *pop               = 1;
                    output->operation  = JI_INVALIDATE;
                } else {
                    *pop              = 0;
                    output->operation = JI_UPDATE;
                }
            }
        }
    }

    if (input.new_job_id != INVALID_JOB_ID) {
        memory_length_t addr = stack_handler_jiu(job_address_stack, reset,
                                                  head_pointer, tail_pointer,
                                                  POP, INVALID_ADDRESS);
        if (addr != INVALID_ADDRESS) {
            job_ids [addr] = input.new_job_id;
            alpha_js[addr] = (uint32_t)input.alpha_j;
        }
    }

    pack_alpha_j_cam(alpha_j_cam, job_ids, alpha_js);
}

/* ── 4. job_info_update ──────────────────────────────────────────────────── */
SIMD_TARGET_AVX512
void job_info_update(job_info_update_input_t  *input,
                     job_id_t                 *top_job_id,
                     job_info_update_output_t *output,
                     one_bit_t                *pop)
{
    static CACHE_ALIGNED alpha_j_info_t   alpha_j_cam      [MAC_PAD][JPM_PAD];
    static CACHE_ALIGNED memory_length_t  job_address_stack[MAC_PAD][JPM_PAD + 1];
    static one_bit_t                      reset            [MAC_PAD];
    static memory_length_t                head_pointer     [MAC_PAD];
    static memory_length_t                tail_pointer     [MAC_PAD];

    /* Zero initializations relying on compiler auto-vectorization */
    for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
        pop[m] = 0;
        output[m].job_id = INVALID_JOB_ID;
        output[m].operation = JI_UPDATE;
    }

    for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
        if (m + 1 < NUM_MACHINES) {
            _mm_prefetch((const char *)alpha_j_cam[m + 1], _MM_HINT_T0);
        }

        alpha_j_update_machine_simd(
            input[m],
            top_job_id[m],
            alpha_j_cam[m],
            job_address_stack[m],
            &reset[m],
            &head_pointer[m],
            &tail_pointer[m],
            &output[m],
            &pop[m]);
    }
}