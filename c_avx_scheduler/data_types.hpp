/**
 * @file data_types.hpp
 * @brief Core data type definitions for the Stochastic Online Scheduler.
 *
 * AVX-512 / AVX2 adaptation notes (Intel Xeon 4th-gen "Sapphire Rapids"):
 *  - All structs that are stored in arrays iterated by SIMD loops carry
 *    alignas(64) so that 512-bit (64-byte) AVX-512 loads/stores never
 *    cross cache-line boundaries.
 *  - NUM_MACHINES = 5.  We pack 5 × uint32 values into the low 160 bits
 *    of a 256-bit __m256i register (8 lanes, 3 upper lanes zeroed/masked).
 *    For JOBS_PER_MACHINE = 10 we use two 256-bit loads (8 + 2) or a
 *    single 512-bit masked load.
 *  - Helper macros SIMD_ALIGNED / CACHE_ALIGNED annotate local arrays that
 *    feed SIMD intrinsics.
 *  - The PRINT_* colour macros and scheduler constants are unchanged.
 */

#ifndef __DATA_TYPES__
#define __DATA_TYPES__

/* ── Standard headers ────────────────────────────────────────────────────── */
#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include <string.h>
#include <fstream>
#include <iostream>

/* ── AVX / AVX-512 intrinsics ────────────────────────────────────────────── */
#include <immintrin.h>   /* AVX, AVX2, AVX-512F, AVX-512BW             */

/* ── Alignment helpers ───────────────────────────────────────────────────── */
/**
 * SIMD_ALIGNED  – aligns to a 32-byte boundary (AVX2 YMM register width).
 * CACHE_ALIGNED – aligns to a 64-byte boundary (AVX-512 ZMM / cache line).
 *
 * Use CACHE_ALIGNED for arrays that will be consumed by 512-bit loads/stores.
 * Use SIMD_ALIGNED  for arrays consumed by 256-bit loads/stores.
 *
 * Implementation note:
 *   We use __attribute__((aligned(N))) rather than alignas(N) because GCC
 *   rejects  "static alignas(N) type var"  (alignas in the middle of
 *   decl-specifiers is ill-formed in that position).
 *   __attribute__((aligned(N))) is accepted by GCC and Clang after the
 *   storage-class keyword  "static __attribute__((aligned(64))) type var".
 *
 *   Struct member alignment (e.g. "struct alignas(64) foo { … }") uses
 *   the C++11 alignas directly — that is always valid.
 */
#if defined(__GNUC__) || defined(__clang__)
#  define SIMD_ALIGNED   __attribute__((aligned(32)))
#  define CACHE_ALIGNED  __attribute__((aligned(64)))
#else
   /* MSVC */
#  define SIMD_ALIGNED   __declspec(align(32))
#  define CACHE_ALIGNED  __declspec(align(64))
#endif

/* ── Scheduler configuration ─────────────────────────────────────────────── */
#ifndef NUM_MACHINES
#define NUM_MACHINES          5    
#endif

#ifndef JOBS_PER_MACHINE
#define JOBS_PER_MACHINE      10   
#endif
/**
 * AVX lane widths used throughout the codebase:
 *   AVX2_INT32_LANES  = 8   (256-bit / 32-bit)
 *   AVX512_INT32_LANES = 16  (512-bit / 32-bit)
 *
 * For NUM_MACHINES = 5 we use an 8-lane YMM register with a 5-bit mask.
 * For JOBS_PER_MACHINE = 10 we use two 8-lane YMM passes (8 + 2) or
 * one 16-lane ZMM pass with a 10-bit mask.
 */
#define AVX2_INT32_LANES     8
#define AVX512_INT32_LANES  16

/** Mask covering only the valid NUM_MACHINES lanes (bits 0-4 set). */
#define MACHINE_LANE_MASK    ((1u << NUM_MACHINES) - 1u)   /* 0x1F = 0b00011111 */

/** Mask covering only the valid JOBS_PER_MACHINE lanes (bits 0-9 set). */
#define JOBS_LANE_MASK       ((1u << JOBS_PER_MACHINE) - 1u) /* 0x3FF */

/* ── Remaining scheduler constants (unchanged) ───────────────────────────── */
#define FIFO_POINTER_BIT_LENGTH  4   /* Update when JOBS_PER_MACHINE changes */
#define INVALID_JOB_ID           0
#define INVALID_ADDRESS          (JOBS_PER_MACHINE + 1)
#define STREAM_LENGTH            1
/** Number of jobs fed to the kernel per invocation.
 *  WARNING: Do NOT increase – larger values cause exponential design growth. */
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

/* ── Stack / JIU / Cost operation codes (unchanged) ─────────────────────── */
#define PUSH           0
#define POP            1
#define JI_UPDATE      0
#define JI_INVALIDATE  1
#define COST_HIGH      0
#define COST_LOW       1

/* ── Scalar typedefs (unchanged) ─────────────────────────────────────────── */
typedef uint32_t   job_id_t;
typedef uint8_t    memory_length_t;   /* Update if JOBS_PER_MACHINE > 255   */
typedef uint16_t   machine_id_t;
typedef memory_length_t vf_index_t;
typedef uint8_t    data_selector_t;
typedef uint8_t    one_bit_t;

/* ── Core data structures ────────────────────────────────────────────────── */

/**
 * proc_time_info_t
 * Padded to 8 bytes so that an array of NUM_MACHINES (5) entries fits neatly
 * in a 256-bit gather index vector (5 × 8B = 40B).
 */
typedef struct alignas(8) proc_time_info {
    job_id_t job_id;   /* 4 bytes */
    uint8_t  proc_time;/* 1 byte  */
    uint8_t  _pad[3];  /* 3 bytes padding → total 8 bytes */
} proc_time_info_t;

/**
 * weight_info_t
 * Padded to 8 bytes for aligned SIMD gather/scatter.
 */
typedef struct alignas(8) weight_info {
    job_id_t job_id;  /* 4 bytes */
    uint8_t  weight;  /* 1 byte  */
    uint8_t  wspt;    /* 1 byte  */
    uint8_t  _pad[2]; /* 2 bytes padding → total 8 bytes */
} weight_info_t;

/**
 * alpha_j_info_t
 * Padded to 8 bytes.
 * SIMD usage: an array of JOBS_PER_MACHINE (10) entries is loaded as
 * 10 × uint32 job_ids into a ZMM register (16-lane masked load) for
 * the parallel job-id comparison in alpha_j_update_machine().
 */
typedef struct alignas(8) alpha_j_info {
    job_id_t job_id;  /* 4 bytes */
    uint8_t  alpha_j; /* 1 byte  */
    uint8_t  _pad[3]; /* 3 bytes padding → total 8 bytes */
} alpha_j_info_t;

/**
 * job_info_update_input_t – input to the Job-Info-Update (JIU) sub-module.
 * One entry per machine; array of 5 is consumed with a 5-lane masked YMM load.
 */
typedef struct alignas(8) job_info_update_input {
    job_id_t new_job_id; /* 4 bytes */
    uint8_t  alpha_j;    /* 1 byte  */
    uint8_t  _pad[3];    /* 3 bytes padding → total 8 bytes */
} job_info_update_input_t;

/**
 * fifo_update_input_t – input to the Virtual-FIFO sub-module.
 * One entry per machine; array of 5 is loaded into two uint32 YMM vectors
 * (one for new_job_id, one for fifo_index) using a gather or interleaved load.
 */
typedef struct alignas(8) fifo_update_input {
    job_id_t  new_job_id;  /* 4 bytes */
    vf_index_t fifo_index; /* 1 byte  */
    uint8_t    _pad[3];    /* 3 bytes padding → total 8 bytes */
} fifo_update_input_t;

/**
 * job_info_update_output_t – output of the JIU sub-module.
 */
typedef struct alignas(8) job_info_update_output {
    job_id_t job_id;    /* 4 bytes */
    uint8_t  operation; /* 1 byte  */
    uint8_t  _pad[3];   /* 3 bytes → total 8 bytes */
} job_info_update_output_t;

/**
 * job_in_t – a new job arriving at the scheduler.
 *
 * Layout designed for efficient SIMD broadcast:
 *   - alpha_j[NUM_MACHINES]        : 5 × uint8  – loaded as 32-bit lanes after
 *                                    zero-extend (_mm256_cvtepu8_epi32)
 *   - processing_time[NUM_MACHINES]: 5 × uint8  – same pattern
 *
 * The struct is padded to a 32-byte boundary so that arrays of job_in_t
 * remain naturally aligned.
 */
typedef struct alignas(32) job_in {
    job_id_t job_id;                       /*  4 bytes            */
    uint8_t  weight;                       /*  1 byte             */
    uint8_t  _pad0[3];                     /*  3 bytes padding    */
    uint8_t  alpha_j[NUM_MACHINES];        /*  5 bytes            */
    uint8_t  _pad1[3];                     /*  3 bytes padding    */
    uint8_t  processing_time[NUM_MACHINES];/*  5 bytes            */
    uint8_t  _pad2[11];                    /* 11 bytes → total 32 */
} job_in_t;

/* ── SIMD helper: broadcast a uint8 array[NUM_MACHINES] → __m256i ─────────
 *
 * Loads 5 uint8 values, zero-extends each to 32-bit, and returns a YMM
 * register with the 5 values in lanes 0-4 and zeros in lanes 5-7.
 *
 * Usage:
 *   __m256i alpha = LOAD_U8_TO_EPI32(job.alpha_j);
 */
static inline __m256i load_u8_to_epi32_5(const uint8_t *arr)
{
    /* Load 4 bytes into the low 32 bits of an XMM, then the 5th byte. */
    __m128i lo = _mm_cvtsi32_si128(*(const uint32_t *)(const void *)arr);
    /* Insert byte 4 at position 4 */
    lo = _mm_insert_epi8(lo, arr[4], 4);
    /* Zero-extend 8-bit lanes → 32-bit lanes (8 output lanes, top 3 = 0) */
    return _mm256_cvtepu8_epi32(lo);
}

/**
 * load_u32_to_ymm_5 – load 5 consecutive uint32_t values into a YMM register.
 * Lanes 5-7 are zeroed via a masked load with mask = 0b00011111 = 0x1F.
 */
static inline __m256i load_u32_to_ymm_5(const uint32_t *arr)
{
    /* Masked load: only lanes 0-4 are loaded; lanes 5-7 remain zero. */
    __m256i zero = _mm256_setzero_si256();
    /* __mmask8 is available with AVX-512VL+F */
    return _mm256_mask_loadu_epi32(zero, (__mmask8)MACHINE_LANE_MASK, arr);
}

/**
 * store_ymm_5_to_u32 – store the low 5 lanes of a YMM register to memory.
 */
static inline void store_ymm_5_to_u32(uint32_t *dst, __m256i v)
{
    _mm256_mask_storeu_epi32(dst, (__mmask8)MACHINE_LANE_MASK, v);
}

#endif /* __DATA_TYPES__ */
