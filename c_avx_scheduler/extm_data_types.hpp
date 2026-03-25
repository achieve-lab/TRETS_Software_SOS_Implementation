/**
 * @file extm_data_types.hpp
 * @brief External-memory interface data types for the Stochastic Online Scheduler.
 *
 * AVX-512 / AVX2 adaptation notes (Intel Xeon 4th-gen "Sapphire Rapids"):
 *
 *  1. new_job_data_host_t
 *     - job_in_t is already aligned to 32 bytes (see data_types.hpp).
 *     - release_tick is promoted to uint64_t and padded so the whole
 *       struct is 64 bytes → one full cache line, one AVX-512 load.
 *
 *  2. scheduler_interface_output_t
 *     - num_jobs[NUM_MACHINES] is padded to 8 × uint16_t so a single
 *       128-bit SSE2 load covers all five counters.
 *     - scheduled_jobs[] uses a 64-byte aligned base for cache-friendly
 *       sequential scans.
 *
 *  3. job_id_manager
 *     - avail_ids[] and release_tick[] arrays are 64-byte aligned so
 *       bulk-reset loops can use 512-bit stores.
 *     - A new SIMD bulk-reset helper (reset_simd) is provided alongside
 *       the original scalar reset() for correctness reference.
 *
 *  4. scheduler_interface_input_t
 *     - new_job_table[] is 64-byte aligned to allow prefetch / streaming
 *       loads from ext_mem_interface.cpp.
 */

#ifndef __EXTM_DATA_TYPES__
#define __EXTM_DATA_TYPES__

#include "data_types.hpp"   /* pulls in <immintrin.h> + alignment macros */

#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include <string.h>

/* ── JOB ID MANAGER size ─────────────────────────────────────────────────── */
/**
 * MANAGER_SIZE: enough IDs for all FIFO slots plus a safety margin for jobs
 * still in the pipeline.
 * Rounded up to the next multiple of AVX512_INT32_LANES (16) so that the
 * avail_ids[] array can be reset with whole ZMM stores.
 */
#define MANAGER_SIZE_RAW   (MEM_DATA_SIZE + 10)
/* Round up to nearest multiple of 16 for clean AVX-512 store alignment */
#define MANAGER_SIZE       (((MANAGER_SIZE_RAW + AVX512_INT32_LANES - 1) \
                             / AVX512_INT32_LANES) * AVX512_INT32_LANES)

/* ── new_job_data_host_t ─────────────────────────────────────────────────── */
/**
 * One job record as seen by the host driver.
 *
 * Size decision:
 *   job_in_t  = 32 bytes  (alignas(32), see data_types.hpp)
 *   release_tick (uint32) =  4 bytes
 *   _pad             =  28 bytes
 *   ──────────────────────────────
 *   Total             = 64 bytes  → exactly one cache line.
 *
 * A MEM_DATA_SIZE (100) element array therefore spans 6 400 bytes and every
 * entry can be loaded with a single 512-bit (_mm512_load_si512) or two
 * 256-bit (_mm256_load_si256) instructions.
 */
typedef struct alignas(64) new_job_data_host {
    job_in_t  job_data;        /* 32 bytes (alignas(32) inside)           */
    uint32_t  release_tick;    /*  4 bytes                                */
    uint8_t   _pad[28];        /* 28 bytes padding → struct = 64 bytes    */
} new_job_data_host_t;

/* ── scheduler_interface_input_t ─────────────────────────────────────────── */
/**
 * Wrapper around the job table sent to the kernel.
 * The new_job_table array is explicitly aligned to 64 bytes so that
 * ext_mem_interface.cpp can use _mm_prefetch() or streaming loads.
 */
typedef struct alignas(64) scheduler_interface_input {
    CACHE_ALIGNED new_job_data_host_t new_job_table[MEM_DATA_SIZE];
    uint32_t initial_tick;
    uint8_t  _pad[60]; /* pad to next 64-byte boundary */
} scheduler_interface_input_t;

/* ── perf_measurement_info_t ─────────────────────────────────────────────── */
/**
 * Per-job scheduling result.
 * Padded to 8 bytes so sequential SIMD scans maintain natural alignment.
 */
typedef struct alignas(8) perf_measurement_info {
    uint32_t   popped_tick;        /* 4 bytes */
    machine_id_t machine_scheduled;/* 2 bytes */
    uint8_t    _pad[2];            /* 2 bytes → total 8 bytes */
} perf_measurement_info_t;

/* ── scheduler_interface_output_t ────────────────────────────────────────── */
/**
 * Output produced by schedule_jobs().
 *
 * num_jobs[]:
 *   Padded from NUM_MACHINES (5) × uint16 to 8 × uint16 (16 bytes) so that
 *   a single __m128i load covers all five counters and three zero-pad lanes:
 *
 *     __m128i counts = _mm_loadu_si128((__m128i*)output.num_jobs);
 *
 *   This allows a horizontal add across machines with _mm_hadd_epi16 chains.
 */
typedef struct alignas(64) scheduler_interface_output {
    CACHE_ALIGNED perf_measurement_info_t scheduled_jobs[MANAGER_SIZE];

    /* Padded to 8 × uint16 for SSE2/AVX2 horizontal reduction */
    alignas(16) uint16_t num_jobs[8]; /* lanes 0-4 valid; 5-7 = 0         */

    uint32_t final_tick;
    uint8_t  _pad[12]; /* pad to 16-byte boundary after final_tick */
} scheduler_interface_output_t;

/* ── job_id_manager ──────────────────────────────────────────────────────── */
/**
 * Manages recycling of job IDs in the range [1, MANAGER_SIZE].
 *
 * AVX-512 optimisations:
 *  - avail_ids[]    : CACHE_ALIGNED uint32_t array → bulk reset via
 *                     _mm512_store_si512 in groups of 16.
 *  - release_tick[] : CACHE_ALIGNED uint32_t array → same benefit.
 *  - reset_simd()   : AVX-512 bulk initialisation of avail_ids[].
 *
 * The original scalar interface (pop/push/assign_id/retrieve_id) is
 * preserved verbatim so the rest of the code requires zero changes.
 */
typedef struct alignas(64) job_id_manager {

    /* release_tick[id] stores the tick at which job <id> was released.
     * Size = MANAGER_SIZE (rounded to multiple of 16). */
    CACHE_ALIGNED uint32_t release_tick[MANAGER_SIZE];

    /* Circular buffer of available (recycled) IDs.
     * Size = MANAGER_SIZE - 1, padded to MANAGER_SIZE for SIMD stores. */
    CACHE_ALIGNED uint32_t avail_ids[MANAGER_SIZE]; /* index 0..MANAGER_SIZE-2 used */

    job_id_t head;
    job_id_t tail;

    /* ── Constructor ────────────────────────────────────────────────────── */
    job_id_manager() {
        head = 0;
        tail = 0;
        reset_simd();
    }

    /* ── Scalar pop (unchanged logic) ──────────────────────────────────── */
    job_id_t pop() {
        job_id_t id = avail_ids[head];
        head++;
        if (head > (MANAGER_SIZE - 2)) head = 0;
        return id;
    }

    /* ── Scalar push (unchanged logic) ─────────────────────────────────── */
    void push(job_id_t id) {
        avail_ids[tail] = id;
        tail++;
        if (tail > (MANAGER_SIZE - 2)) tail = 0;
    }

    /* ── Scalar assign / retrieve (unchanged) ───────────────────────────── */
    job_id_t assign_id(uint32_t released) {
        job_id_t id = this->pop();
        this->release_tick[id] = released;
        return id;
    }

    uint32_t retrieve_id(job_id_t id) {
        this->push(id);
        return release_tick[id];
    }

    /* ── Scalar reset (original, kept for reference) ────────────────────── */
    void reset() {
        head = 0;
        tail = 0;
        for (job_id_t i = 0; i < (MANAGER_SIZE - 1); i++) {
            avail_ids[i] = i + 1;
        }
    }

    /* ── AVX-512 bulk reset ─────────────────────────────────────────────── */
    /**
     * reset_simd()
     *
     * Initialises avail_ids[i] = i + 1 for i in [0, MANAGER_SIZE-2] using
     * AVX-512 stores of 16 × uint32 per iteration.
     *
     * Strategy:
     *   1. Build a base vector {0,1,2,...,15} + offset using _mm512_add_epi32.
     *   2. Add the scalar constant 1 (IDs start at 1, 0 is INVALID_JOB_ID).
     *   3. Store 16 values per iteration; final partial chunk uses a masked store.
     *
     * MANAGER_SIZE is a multiple of 16 (see macro above), so the loop always
     * processes whole chunks — no tail scalar fixup needed.
     * The last entry (index MANAGER_SIZE-1) is left uninitialised because
     * only indices [0, MANAGER_SIZE-2] are valid IDs.
     */
    void reset_simd() {
        head = 0;
        tail = 0;

        /* Base sequence: 0, 1, 2, ..., 15 */
        const __m512i seq = _mm512_set_epi32(15,14,13,12,11,10,9,8,
                                              7, 6, 5, 4, 3, 2,1,0);
        /* +1 because IDs are 1-based */
        const __m512i one = _mm512_set1_epi32(1);

        const int total   = MANAGER_SIZE - 1;  /* valid entries            */
        const int full    = total / AVX512_INT32_LANES;
        const int remain  = total % AVX512_INT32_LANES;

        for (int chunk = 0; chunk < full; chunk++) {
            /* offset = chunk * 16;  values = offset + seq + 1 */
            __m512i offset = _mm512_set1_epi32(chunk * AVX512_INT32_LANES);
            __m512i vals   = _mm512_add_epi32(_mm512_add_epi32(seq, offset), one);
            _mm512_store_si512((__m512i *)(avail_ids + chunk * AVX512_INT32_LANES),
                               vals);
        }

        /* Tail: remaining < 16 entries, use a masked store */
        if (remain > 0) {
            __mmask16 tail_mask = (__mmask16)((1u << remain) - 1u);
            __m512i offset = _mm512_set1_epi32(full * AVX512_INT32_LANES);
            __m512i vals   = _mm512_add_epi32(_mm512_add_epi32(seq, offset), one);
            _mm512_mask_store_epi32(avail_ids + full * AVX512_INT32_LANES,
                                    tail_mask, vals);
        }
    }

} job_id_manager;

/* ── Convenience: zero-initialise a scheduler_interface_output_t ─────────── */
/**
 * Uses AVX-512 non-temporal stores to flush the large scheduled_jobs[] array
 * to zero without polluting the L1/L2 caches (write-combining path).
 *
 * Callable as:  scheduler_interface_output_t out; zero_output_simd(&out);
 */
static inline void zero_output_simd(scheduler_interface_output_t *out)
{
    const __m512i zero512 = _mm512_setzero_si512();

    /* Zero scheduled_jobs[] array */
    const size_t jobs_bytes = sizeof(perf_measurement_info_t) * MANAGER_SIZE;
    uint8_t *base = reinterpret_cast<uint8_t *>(out->scheduled_jobs);
    for (size_t off = 0; off < jobs_bytes; off += 64) {
        _mm512_stream_si512((__m512i *)(base + off), zero512);
    }

    /* Zero counters */
    _mm_storeu_si128((__m128i *)out->num_jobs, _mm_setzero_si128());
    out->final_tick = 0;
}

#endif /* __EXTM_DATA_TYPES__ */
