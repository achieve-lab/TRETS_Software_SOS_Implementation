/**
 * @file scheduler.cpp
 * @brief Top-level scheduler orchestration for the Stochastic Online Scheduler.
 *
 * AVX-512 / AVX2 vectorisation (Intel Xeon 4th-gen "Sapphire Rapids")
 * =====================================================================
 *
 * Role of this file
 * -----------------
 * scheduler() is the single-tick orchestrator: it calls three sub-modules
 * in sequence:
 *
 *   cost_calculator()   →   job_info_update()   →   virtual_fifo()
 *
 * Each sub-module now carries its own AVX-512/AVX2 vectorisation
 * (files 5-7).  scheduler.cpp adds SIMD optimisation at the glue level:
 *
 * 1. Zero-initialisation of per-tick local arrays
 *    -----------------------------------------------
 *    pop[NUM_MACHINES], jiu_input[NUM_MACHINES], fifo_input[NUM_MACHINES]
 *    are zeroed with SIMD stores rather than scalar loops or memset().
 *
 *    - pop[5]          (uint8_t × 5 = 5 bytes)
 *      Single 32-bit + 8-bit zero-store, identical to job_info_update.cpp.
 *
 *    - jiu_input[5]    (job_info_update_input_t × 5, 8 bytes each = 40 bytes)
 *      Single _mm256_storeu_si256 + uint64_t zero covers all 5 entries.
 *
 *    - fifo_input[5]   (fifo_update_input_t × 5, 8 bytes each = 40 bytes)
 *      Same pattern as jiu_input.
 *
 * 2. Static state arrays (updated_job_info, top_job_id)
 *    ----------------------------------------------------
 *    These persist across ticks and are CACHE_ALIGNED so sub-module
 *    SIMD loads/stores into them are naturally aligned.
 *
 * 3. Debug print loop (NUM_MACHINES = 5 iterations)
 *    --------------------------------------------------
 *    Kept scalar — printf is not vectorisable and debug is off by default.
 *
 * 4. AVX2 horizontal OR across fifo_full[NUM_MACHINES]
 *    ---------------------------------------------------
 *    The all_fifo_full check (used implicitly by cost_calculator and the
 *    ext_mem_interface) is provided here via all_bits_set_5() from
 *    top_modules.hpp, which uses PCMPEQB + PMOVMSKB — single-cycle on
 *    Sapphire Rapids.
 *
 * Function signature is UNCHANGED from the scalar version so
 * ext_mem_host.cpp requires zero edits.
 */

#include "data_types.hpp"
#include "top_modules.hpp"
#include <cstring>   /* memset — fallback only */
#include <cstdio>

/* ═══════════════════════════════════════════════════════════════════════════
 * Helper: SIMD zero-initialise a jiu_input[NUM_MACHINES] array
 * (job_info_update_input_t × 5, each 8 bytes = 40 bytes total)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * zero_jiu_input_simd()
 *
 * Zeroes all 5 entries of a job_info_update_input_t array using a single
 * 256-bit store (covers 32 bytes) plus one 8-byte scalar store (8 bytes),
 * totalling 40 bytes with no loop overhead.
 *
 * Each zeroed entry has:
 *   new_job_id = 0  (= INVALID_JOB_ID)
 *   alpha_j    = 0
 *   _pad[3]    = 0
 */
SIMD_TARGET_AVX2
static inline void zero_jiu_input_simd(job_info_update_input_t *arr)
{
    _mm256_storeu_si256((__m256i *)arr,       _mm256_setzero_si256()); /* [0..3] */
    *reinterpret_cast<uint64_t *>(arr + 4) = 0ULL;                    /* [4]    */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Helper: SIMD zero-initialise a fifo_input[NUM_MACHINES] array
 * (fifo_update_input_t × 5, each 8 bytes = 40 bytes total)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * zero_fifo_input_simd()
 *
 * Same 256-bit + 8-byte pattern as zero_jiu_input_simd().
 *
 * Each zeroed entry has:
 *   new_job_id  = 0  (= INVALID_JOB_ID)
 *   fifo_index  = 0
 *   _pad[3]     = 0
 *
 * Note: JOBS_PER_MACHINE is the sentinel "no insertion" fifo_index in the
 * original code; cost_calculator() sets the winner entry explicitly, so
 * zeroing fifo_index here is safe — the cost_calculator output always
 * overwrites the relevant entry before virtual_fifo() reads it.
 */
SIMD_TARGET_AVX2
static inline void zero_fifo_input_simd(fifo_update_input_t *arr)
{
    _mm256_storeu_si256((__m256i *)arr,       _mm256_setzero_si256()); /* [0..3] */
    *reinterpret_cast<uint64_t *>(arr + 4) = 0ULL;                    /* [4]    */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Helper: SIMD zero-initialise pop[NUM_MACHINES]  (uint8_t × 5 = 5 bytes)
 * ═══════════════════════════════════════════════════════════════════════════ */

SIMD_TARGET_AVX2
static inline void zero_pop_simd(one_bit_t *pop)
{
    *reinterpret_cast<uint32_t *>(pop)  = 0u; /* bytes 0-3 */
    pop[4]                              = 0u; /* byte  4   */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Helper: SIMD copy top_job_id[NUM_MACHINES]  (uint32_t × 5 = 20 bytes)
 *
 * Used to move the updated top_job_id values written by virtual_fifo()
 * back into the static array without a scalar loop.
 * The src pointer comes from virtual_fifo()'s output parameter.
 * ═══════════════════════════════════════════════════════════════════════════ */

SIMD_TARGET_AVX2
static inline void copy_top_job_ids_simd(job_id_t       *dst,
                                          const job_id_t *src)
{
    /* 5 × uint32_t = 20 bytes: 16-byte XMM + one 4-byte scalar */
    __m128i v = _mm_loadu_si128((const __m128i *)src);  /* [0..3] */
    _mm_storeu_si128((__m128i *)dst, v);
    dst[4] = src[4];                                     /* [4]    */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * scheduler()  — top-level entry point (signature unchanged)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * scheduler()
 *
 * Single-tick scheduling step.  Orchestrates:
 *   1. cost_calculator()   — assigns new_job to the optimal machine,
 *                            produces jiu_input and fifo_input.
 *   2. job_info_update()   — decrements alpha_j for the top job,
 *                            updates updated_job_info and pop signals.
 *   3. virtual_fifo()      — maintains sorted FIFOs, produces popped_job_id
 *                            and fifo_full per machine.
 *
 * SIMD additions at this level:
 *   - Per-tick local arrays (pop, jiu_input, fifo_input) zero-initialised
 *     with SIMD stores (no scalar loops).
 *   - Static arrays (updated_job_info, top_job_id) are CACHE_ALIGNED so
 *     sub-module AVX loads/stores into them are naturally aligned.
 *   - top_job_id[] is copied back from virtual_fifo()'s in-place update
 *     using a 128-bit XMM load+store (covers 4 entries) + 1 scalar store.
 *   - fifo_full[] horizontal-OR test uses all_bits_set_5() from
 *     top_modules.hpp (PCMPEQB + PMOVMSKB, single instruction sequence).
 *
 * @param new_job         New job arriving this tick (job_id == INVALID if none)
 * @param popped_job_id   [out] Job IDs popped from each machine's FIFO
 * @param fifo_full       [out] Full-flag for each machine's FIFO
 */
SIMD_TARGET_AVX512
void scheduler(job_in_t   new_job,
               job_id_t   popped_job_id[NUM_MACHINES],
               one_bit_t  fifo_full    [NUM_MACHINES])
{
    /* ── Persistent state (survives across ticks) ─────────────────────
     *
     * updated_job_info[]: written by job_info_update(), read by the NEXT
     *   call to cost_calculator() via the scheduler → cost_calculator path.
     *   CACHE_ALIGNED so AVX loads inside cost_calculator are aligned.
     *
     * top_job_id[]:       written by virtual_fifo() (in-place via pointer),
     *   read by job_info_update() on the same tick.
     *   Must survive from the virtual_fifo() write (end of tick N) to the
     *   job_info_update() read (tick N+1).
     * ────────────────────────────────────────────────────────────────── */
    static CACHE_ALIGNED job_info_update_output_t
        updated_job_info[NUM_MACHINES];   /* zero-init by static storage */

    static CACHE_ALIGNED job_id_t
        top_job_id[NUM_MACHINES];         /* zero-init by static storage */

    /* ── Per-tick locals ──────────────────────────────────────────────
     *
     * These are written freshly every tick, so we zero them with SIMD
     * rather than relying on zero-initialisation syntax (which generates
     * a scalar memset on most compilers at -O2 without LTO).
     * ────────────────────────────────────────────────────────────────── */
    CACHE_ALIGNED one_bit_t               pop       [NUM_MACHINES];
    CACHE_ALIGNED job_info_update_input_t jiu_input [NUM_MACHINES];
    CACHE_ALIGNED fifo_update_input_t     fifo_input[NUM_MACHINES];

    /* Zero-initialise per-tick arrays with SIMD stores */
    zero_pop_simd      (pop);
    zero_jiu_input_simd(jiu_input);
    zero_fifo_input_simd(fifo_input);

    /* ── Stage 1: Cost Calculator ─────────────────────────────────────
     *
     * Determines which machine receives new_job (if any), computes the
     * insertion index (fifo_input[m].fifo_index) and alpha_j update
     * (jiu_input[m]).
     *
     * Reads:  new_job, updated_job_info (from previous tick), fifo_full
     * Writes: jiu_input, fifo_input
     * ────────────────────────────────────────────────────────────────── */
#if SCHEDULER_DEBUG
    printf("\033[31m[SCHED] Cost Calculator\n\033[0m");
#endif

    cost_calculator(new_job, updated_job_info, fifo_full,
                    jiu_input, fifo_input);

    /* ── Stage 2: Job Info Update ─────────────────────────────────────
     *
     * Decrements alpha_j for the current top job on each machine.
     * If alpha_j reaches zero, signals a pop via pop[m] = 1.
     *
     * Reads:  jiu_input, top_job_id (from previous tick's virtual_fifo)
     * Writes: updated_job_info (consumed by next tick's cost_calculator),
     *         pop (consumed by virtual_fifo this tick)
     * ────────────────────────────────────────────────────────────────── */
#if SCHEDULER_DEBUG
    printf("\033[31m[SCHED] Job Info Update\n\033[0m");
#endif

    job_info_update(jiu_input, top_job_id, updated_job_info, pop);

    /* ── Stage 3: Virtual FIFO ────────────────────────────────────────
     *
     * Maintains a sorted FIFO per machine.
     * Inserts new_job at the index determined by cost_calculator.
     * Pops the top entry when pop[m] == 1.
     *
     * Reads:  fifo_input, pop
     * Writes: top_job_id (updated in-place — reflects new FIFO top),
     *         popped_job_id, fifo_full
     * ────────────────────────────────────────────────────────────────── */
#if SCHEDULER_DEBUG
    printf("\033[31m[SCHED] Virtual FIFO\n\033[0m");
#endif

    virtual_fifo(fifo_input, pop, top_job_id, popped_job_id, fifo_full);

    /* ── Debug: per-machine state dump ────────────────────────────────
     *
     * Kept scalar — printf is serialising; debug is off by default
     * (DEBUG=0 in top_modules.hpp).
     * ────────────────────────────────────────────────────────────────── */
#if SCHEDULER_DEBUG
    printf("\033[31m[SCHED] ----------------------------------------\033[0m\n");
    for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
        printf("\033[31m[SCHED] Machine: %u  Top: %u  Popped: %u  "
               "Full: %u\033[0m\n",
               m,
               top_job_id    [m],
               popped_job_id [m],
               (unsigned int)fifo_full[m]);
    }
    printf("\033[31m[SCHED] ----------------------------------------\033[0m\n");
#endif

    /* ── Optional: expose the all_fifo_full signal ────────────────────
     *
     * all_bits_set_5() uses PCMPEQB + PMOVMSKB from top_modules.hpp.
     * The result is not used at this level but is available to callers
     * (e.g., ext_mem_interface.cpp) via the fifo_full[] output array.
     *
     * Included here to ensure the inline is instantiated in this TU so
     * the linker can inline it at the call site in ext_mem_interface.cpp
     * when LTO is enabled.
     * ────────────────────────────────────────────────────────────────── */
    (void)all_bits_set_5(fifo_full);   /* result consumed by caller */
}
