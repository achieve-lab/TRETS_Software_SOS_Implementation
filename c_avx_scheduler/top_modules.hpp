/**
 * @file top_modules.hpp
 * @brief Top-level function declarations for the Stochastic Online Scheduler.
 */

#ifndef __TOP_MODULES__
#define __TOP_MODULES__

#include "data_types.hpp"       /* alignas, SIMD helpers, job_in_t …      */
#include "extm_data_types.hpp"  /* scheduler_interface_output_t …         */

/* ═══════════════════════════════════════════════════════════════════════════
 * 1.  Compile-time SIMD capability checks
 * ═══════════════════════════════════════════════════════════════════════════ */

#if !defined(__AVX2__)
#  error "This translation unit requires AVX2.  Add -mavx2 to CXXFLAGS."
#endif

/** AVX-512 Foundation — needed for ZMM registers, masked operations. */
#if defined(__AVX512F__)
#  define HAVE_AVX512F  1
#else
#  define HAVE_AVX512F  0
#  pragma message("AVX-512F not enabled: falling back to AVX2 code paths. " \
                  "Add -mavx512f for best performance on Sapphire Rapids.")
#endif

/** AVX-512 Byte-and-Word — needed for 8/16-bit SIMD operations. */
#if defined(__AVX512BW__)
#  define HAVE_AVX512BW 1
#else
#  define HAVE_AVX512BW 0
#endif

/** AVX-512 Vector Length Extensions — enables 128/256-bit masked ops. */
#if defined(__AVX512VL__)
#  define HAVE_AVX512VL 1
#else
#  define HAVE_AVX512VL 0
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * 2.  Per-function target attribute macros
 * ═══════════════════════════════════════════════════════════════════════════ */

#if defined(__GNUC__) || defined(__clang__)
    /* * In GCC 14, using __attribute__((target(...))) overrides the global -march
     * flags and drops the implicitly required 'evex512' feature (which enables 
     * 512-bit vector intrinsics) unless explicitly specified. 
     * Since the Makefile already globally specifies `-march=sapphirerapids` or 
     * `-mavx512f`, these attributes are redundant and cause compilation failures.
     * We safely define them to empty to let the global CXXFLAGS govern the build.
     */
#  define SIMD_TARGET_AVX512
#  define SIMD_TARGET_AVX2
#else
   /* MSVC: /arch:AVX512 covers everything; attributes are no-ops. */
#  define SIMD_TARGET_AVX512
#  define SIMD_TARGET_AVX2
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * 3.  HW-flag sentinel — top-of-module constants (unchanged from scalar)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define VF_TOP               0
#define JIU_TOP              0
#define SCHEDULER_TOP        0
#define COST_CALCULATOR_TOP  0

/* ═══════════════════════════════════════════════════════════════════════════
 * 4.  Debug / verbosity flags
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifndef DEBUG
#  define DEBUG 0
#endif

#if DEBUG
#  ifndef VF_DEBUG
#    define VF_DEBUG                    1
#  endif
#  ifndef JIU_DEBUG
#    define JIU_DEBUG                   1
#  endif
#  ifndef SCHEDULER_DEBUG
#    define SCHEDULER_DEBUG             1
#  endif
#  ifndef COST_CALCULATOR_DEBUG
#    define COST_CALCULATOR_DEBUG       1
#  endif
#  ifndef EXT_MEM_INTERFACE_DEBUG_GEN
#    define EXT_MEM_INTERFACE_DEBUG_GEN    1
#  endif
#  ifndef EXT_MEM_INTERFACE_DEBUG_DETAIL
#    define EXT_MEM_INTERFACE_DEBUG_DETAIL 1
#  endif
#  ifndef DUMP_MEMORY
#    define DUMP_MEMORY                 1
#  endif
#else
#  ifndef VF_DEBUG
#    define VF_DEBUG                    0
#  endif
#  ifndef JIU_DEBUG
#    define JIU_DEBUG                   0
#  endif
#  ifndef SCHEDULER_DEBUG
#    define SCHEDULER_DEBUG             0
#  endif
#  ifndef COST_CALCULATOR_DEBUG
#    define COST_CALCULATOR_DEBUG       0
#  endif
#  ifndef EXT_MEM_INTERFACE_DEBUG_GEN
#    define EXT_MEM_INTERFACE_DEBUG_GEN    0
#  endif
#  ifndef EXT_MEM_INTERFACE_DEBUG_DETAIL
#    define EXT_MEM_INTERFACE_DEBUG_DETAIL 0
#  endif
#  ifndef DUMP_MEMORY
#    define DUMP_MEMORY                 0
#  endif
#endif /* DEBUG */

/* ═══════════════════════════════════════════════════════════════════════════
 * 5.  Runtime SIMD capability query
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct simd_capabilities {
    bool has_avx2;
    bool has_avx512f;
    bool has_avx512bw;
    bool has_avx512vl;
} simd_capabilities_t;

static inline simd_capabilities_t query_simd_capabilities(void)
{
    simd_capabilities_t caps = {false, false, false, false};

#if defined(__GNUC__) || defined(__clang__)
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
    __asm__ __volatile__ (
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0)
    );
    caps.has_avx2      = (ebx >> 5)  & 1u;  
    caps.has_avx512f   = (ebx >> 16) & 1u;  
    caps.has_avx512bw  = (ebx >> 30) & 1u;  
    caps.has_avx512vl  = (ebx >> 31) & 1u;  
#elif defined(_MSC_VER)
    int info[4] = {0};
    __cpuidex(info, 7, 0);
    caps.has_avx2      = (info[1] >> 5)  & 1;
    caps.has_avx512f   = (info[1] >> 16) & 1;
    caps.has_avx512bw  = (info[1] >> 30) & 1;
    caps.has_avx512vl  = (info[1] >> 31) & 1;
#endif
    return caps;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6.  Sub-module function declarations
 * ═══════════════════════════════════════════════════════════════════════════ */

SIMD_TARGET_AVX512
void virtual_fifo(fifo_update_input_t *input,
                  one_bit_t           *pop,
                  job_id_t            *top_job_id,
                  job_id_t            *popped_job_id,
                  one_bit_t           *fifo_full);

SIMD_TARGET_AVX512
void job_info_update(job_info_update_input_t  *input,
                     job_id_t                 *top_job_id,
                     job_info_update_output_t *output,
                     one_bit_t                *pop);

SIMD_TARGET_AVX512
void scheduler(job_in_t  new_job,
               job_id_t  popped_job_id[NUM_MACHINES],
               one_bit_t fifo_full[NUM_MACHINES]);

SIMD_TARGET_AVX512
void cost_calculator(job_in_t                  new_job,
                     job_info_update_output_t *updated_job_info,
                     one_bit_t                *fifo_full,
                     job_info_update_input_t  *jiu_input,
                     fifo_update_input_t      *fifo_input);

SIMD_TARGET_AVX512
void schedule_jobs(new_job_data_host_t          *input_stream,
                   uint32_t                      initial_tick,
                   scheduler_interface_output_t *output);

/* ═══════════════════════════════════════════════════════════════════════════
 * 7.  Inline SIMD utility
 * ═══════════════════════════════════════════════════════════════════════════ */

SIMD_TARGET_AVX2
static inline int all_bits_set_5(const uint8_t *arr)
{
    __m128i v = _mm_cvtsi32_si128(*(const uint32_t *)(const void *)arr);
    v = _mm_insert_epi8(v, (int)arr[4], 4);
    __m128i eq_zero = _mm_cmpeq_epi8(v, _mm_setzero_si128());
    int mask = _mm_movemask_epi8(eq_zero);
    return ((mask & 0x1F) == 0) ? 1 : 0;
}

SIMD_TARGET_AVX2
static inline int any_bit_set_5(const uint8_t *arr)
{
    __m128i v = _mm_cvtsi32_si128(*(const uint32_t *)(const void *)arr);
    v = _mm_insert_epi8(v, (int)arr[4], 4);
    return !_mm_testz_si128(v, v);
}

#endif /* __TOP_MODULES__ */