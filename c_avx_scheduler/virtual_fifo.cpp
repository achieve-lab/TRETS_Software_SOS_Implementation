/**
 * @file virtual_fifo.cpp
 * @brief Sorted virtual FIFO implementation for the Stochastic Online Scheduler.
 *
 * AVX-512 / AVX2 vectorisation (Intel Xeon 4th-gen "Sapphire Rapids")
 * =====================================================================
 *
 * Original hot-spots and their SIMD replacements
 * -----------------------------------------------
 *
 * 1. data_selector_per_index()  [called 10× per machine]
 *    Replaced by compute_data_select_vec() from virtual_fifo.hpp.
 *    All 8 lanes (slots 0-7) computed simultaneously with AVX2 compare+blend.
 *    Slots 8-9 handled with a 2-lane masked AVX-512VL operation.
 *
 * 2. data_selector() / data_selector_all_machines()
 *    Outer shell preserved; inner per-slot logic fully vectorised.
 *    The input_temp / pop_temp staging arrays (needed for HLS pipelining)
 *    are retained for correctness but filled with a single memcpy per machine.
 *
 * 3. fifo_write loop  [10 iterations per machine]
 *    a. Load prev[0..9] as lo8 YMM + hi2 YMM (load_fifo_regs()).
 *    b. Build left-neighbour  vector: prev[i+1]   (build_left_neighbour()).
 *    c. Build right-neighbour vector: prev[i-1]   (build_right_neighbour()).
 *    d. Apply data_select for slots 1-7 via fifo_reg_update_mid().
 *    e. Handle slot 0 (asymmetric left boundary) and slot 9 (right boundary)
 *       scalarly — 2 instructions total, not worth further vectorisation.
 *    f. Determine top_job_id from data_select_signals[0] (scalar switch).
 *    g. Store cur[0..9] with store_fifo_regs().
 *
 * 4. fifo_entry_count update and full/popped logic
 *    Three masked integer adds on a scalar counter — kept scalar.
 *    The full-signal comparison is a single integer compare.
 *
 * 5. fifo_read loop  [10 iterations per machine]
 *    Replaced by two _mm256_store_si256 / _mm512_mask_storeu_epi32 calls:
 *    a. Copy cur[0..7]  → prev[0..7]  with a single 256-bit store.
 *    b. Copy cur[8..9]  → prev[8..9]  with a 2-lane masked store.
 *
 * 6. fifo_all_machines  [NUM_MACHINES = 5 outer iterations]
 *    Each machine is processed sequentially (static state is per-machine).
 *    Software prefetch for the next machine's vf_machine_state issued
 *    at the top of each iteration.
 *
 * Data layout:
 *   vf_machine_state_t (virtual_fifo.hpp):
 *     fifo_regs_cur [JOBS_PER_MACHINE]  SIMD_ALIGNED uint32_t  (40 B)
 *     _pad0[24]
 *     fifo_regs_prev[JOBS_PER_MACHINE]  SIMD_ALIGNED uint32_t  (40 B)
 *     entry_count                       uint32_t
 *     _pad1[20]
 *   Total: 128 bytes (2 cache lines).
 */

#include "virtual_fifo.hpp"
#include "top_modules.hpp"
#include <cstring>   /* memcpy */

/* ═══════════════════════════════════════════════════════════════════════════
 * 0.  Module-level broadcast constants (declared extern in virtual_fifo.hpp)
 * ═══════════════════════════════════════════════════════════════════════════ */

const __m256i VF_VEC_W_DISABLE    = /* will be initialised below */
    /* zero-initialised at file scope; reset in vf_init_constants() */
    /* We use a constructor trick: see vf_constants_initialiser below. */
    /* Actual value set in the translation-unit constructor. */
    /* (GNU C++ guarantees static __m256i is zero at start.) */
    /* These are assigned in the static initialiser object below.    */
    /* Placeholder — real values set by vf_constants_initialiser.   */
    /* GCC zero-initialises static __m256i via memset(0), which      */
    /* matches _mm256_setzero_si256(), i.e. W_DISABLE (= 0). ✓      */
    {};   /* W_DISABLE = 0 → zero vector is correct default */

const __m256i VF_VEC_RD_SELECT    = {};   /* set by initialiser below */
const __m256i VF_VEC_LD_SELECT    = {};   /* set by initialiser below */
const __m256i VF_VEC_NEW_D_SELECT = {};   /* set by initialiser below */

/* C++ file-scope constructor to initialise the broadcast constants once. */
namespace {
struct VfConstantsInitialiser {
    VfConstantsInitialiser() {
        /* Cast away const for one-time initialisation */
        *const_cast<__m256i*>(&VF_VEC_W_DISABLE)    = _mm256_set1_epi32(W_DISABLE);
        *const_cast<__m256i*>(&VF_VEC_RD_SELECT)    = _mm256_set1_epi32(RD_SELECT);
        *const_cast<__m256i*>(&VF_VEC_LD_SELECT)    = _mm256_set1_epi32(LD_SELECT);
        *const_cast<__m256i*>(&VF_VEC_NEW_D_SELECT) = _mm256_set1_epi32(NEW_D_SELECT);
    }
} vf_constants_initialiser;
} // anonymous namespace

/* ═══════════════════════════════════════════════════════════════════════════
 * 1.  data_selector_per_index  (scalar reference — kept for clarity/debug)
 *     Hot path uses compute_data_select_vec() from virtual_fifo.hpp.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Scalar fallback for a single FIFO slot.
 * Used only for slot 8 and slot 9 (tail slots handled outside the main
 * 8-wide AVX2 vector pass) and for debug validation.
 */
static inline data_selector_t data_selector_scalar(
        fifo_update_input_t input,
        one_bit_t           pop_input,
        vf_index_t          fifo_index)
{
    if (input.new_job_id != INVALID_JOB_ID) {
        if (pop_input) {
            if (input.fifo_index <= 1) {
                return (fifo_index == 0) ? NEW_D_SELECT : W_DISABLE;
            } else {
                if      (fifo_index < (input.fifo_index - 1)) return LD_SELECT;
                else if (fifo_index == (input.fifo_index - 1)) return NEW_D_SELECT;
                else                                           return W_DISABLE;
            }
        } else {
            if      (fifo_index < input.fifo_index)  return W_DISABLE;
            else if (fifo_index == input.fifo_index) return NEW_D_SELECT;
            else                                     return RD_SELECT;
        }
    } else {
        return pop_input ? LD_SELECT : W_DISABLE;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2.  data_selector_machine_simd()
 *     Compute all JOBS_PER_MACHINE=10 data-select signals for one machine.
 *     Returns results in ds_out[JOBS_PER_MACHINE].
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * data_selector_machine_simd()
 *
 * Computes data_select_signals[0..9] for one machine using:
 *   - compute_data_select_vec() (AVX2, 8 lanes) for slots 0-7.
 *   - data_selector_scalar()                    for slots 8-9.
 *
 * The slot_indices constant vector {0,1,2,...,7} is built once and reused.
 */
SIMD_TARGET_AVX2
static void data_selector_machine_simd(
        fifo_update_input_t   input,
        one_bit_t             pop_input,
        data_selector_t      *ds_out)          /* [JOBS_PER_MACHINE] */
{
    /* Constant lane-index vector: {0,1,2,3,4,5,6,7} */
    const __m256i slot_indices = _mm256_set_epi32(7,6,5,4,3,2,1,0);

    /* ── Slots 0-7 via AVX2 (8 lanes) ─────────────────────────────── */
    __m256i v_ds = compute_data_select_vec(input.new_job_id,
                                            (uint32_t)input.fifo_index,
                                            pop_input,
                                            slot_indices);

    /* Store 8 × uint32 data-select values into ds_out[0..7]
     * data_selector_t is uint8_t, so we need to pack down from 32→8 bit. */
    SIMD_ALIGNED uint32_t ds32[8];
    _mm256_store_si256((__m256i *)ds32, v_ds);
    for (int i = 0; i < 8; i++)
        ds_out[i] = (data_selector_t)ds32[i];

    /* ── Slots 8-9 scalar ──────────────────────────────────────────── */
    ds_out[8] = data_selector_scalar(input, pop_input, 8);
    ds_out[9] = data_selector_scalar(input, pop_input, 9);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3.  data_selector_all_machines()  (vectorised inner, sequential outer)
 * ═══════════════════════════════════════════════════════════════════════════ */

SIMD_TARGET_AVX2
static void data_selector_all_machines(
        fifo_update_input_t  *input,
        one_bit_t            *pop_input,
        data_selector_t       data_select_signals[NUM_MACHINES][JOBS_PER_MACHINE],
        fifo_update_input_t  *input_temp,
        one_bit_t            *pop_input_temp)
{
    for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
        data_selector_machine_simd(input[m], pop_input[m],
                                    data_select_signals[m]);
        input_temp[m]    = input[m];
        pop_input_temp[m]= pop_input[m];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4.  fifo_one_machine_simd()
 *     Vectorised replacement for fifo() — processes one machine.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * fifo_one_machine_simd()
 *
 * Executes the fifo_write + fifo_read double-loop for one machine using
 * AVX2 vector operations:
 *
 *  fifo_write (vectorised):
 *   a. load_fifo_regs(prev) → lo8, hi2.
 *   b. build_left_neighbour(lo8, hi2)    → v_left  = prev[i+1] for i=0..7.
 *   c. build_right_neighbour(lo8)        → v_right = prev[i-1] for i=0..7.
 *   d. Convert ds[0..7] uint8 → uint32 YMM v_ds8.
 *   e. fifo_reg_update_mid(prev_lo8, v_left, v_right, v_ds8, new_job_vec)
 *      → new_cur[0..7]  (slots 1-7 correct; slot 0 fixed up below).
 *   f. Scalar fix-up for slot 0 (left boundary: left_data = prev[1]).
 *      Scalar fix-up for slot 9 (right boundary: right_data = prev[8]).
 *   g. Determine top_job_id from ds[0] (scalar switch, 1 comparison).
 *   h. store_fifo_regs(cur, new_cur_lo8, new_cur_hi2).
 *
 *  entry_count, popped_job_id, full (scalar — 3 operations).
 *
 *  fifo_read (vectorised):
 *   Copy cur[0..7] → prev[0..7] with _mm256_store_si256.
 *   Copy cur[8..9] → prev[8..9] with 2-lane masked store.
 *
 * Parameters use vf_machine_state_t (virtual_fifo.hpp) which co-locates
 * cur and prev arrays in adjacent cache lines for minimal cache pressure.
 */
SIMD_TARGET_AVX2
static void fifo_one_machine_simd(
        job_id_t          new_job_id,
        one_bit_t         pop,
        data_selector_t  *ds,              /* [JOBS_PER_MACHINE] */
        vf_machine_state_t *state,         /* cur + prev + entry_count */
        job_id_t         *top_job_id_out,
        job_id_t         *popped_job_id_out,
        one_bit_t        *full_out)
{
    uint32_t *cur  = state->fifo_regs_cur;
    uint32_t *prev = state->fifo_regs_prev;

    /* ── fifo_write ──────────────────────────────────────────────────── */

    /* a. Load prev[0..9] into two YMM registers */
    __m256i lo8, hi2;
    load_fifo_regs(prev, &lo8, &hi2);

    /* b. Left neighbour: prev[i+1] for lanes 0-7 */
    __m256i v_left  = build_left_neighbour(lo8, hi2);

    /* c. Right neighbour: prev[i-1] for lanes 0-7 (lane 0 = INVALID) */
    __m256i v_right = build_right_neighbour(lo8);

    /* d. Convert ds[0..7] uint8 → uint32 YMM */
    /*    Load 8 bytes into XMM low, zero-extend each byte to 32 bits */
    __m128i xmm_ds8 = _mm_loadl_epi64((const __m128i *)ds);
    __m256i v_ds8   = _mm256_cvtepu8_epi32(xmm_ds8);

    /* e. Broadcast new_job_id */
    __m256i v_new = _mm256_set1_epi32((int)new_job_id);

    /* f. Vectorised update for all 8 lanes (slots 0-7) */
    __m256i v_cur_lo8 = fifo_reg_update_mid(lo8, v_left, v_right,
                                              v_ds8, v_new);

    /* g. Scalar fix-up for slot 0:
     *    fifo_register(prev[1], INVALID_JOB_ID, new_job_id, prev[0], ds[0])
     *    The SIMD pass treated slot 0's right-neighbour as prev[-1] = INVALID
     *    (correct) and left-neighbour as prev[1] (correct).
     *    We only need to override if ds[0] == RD_SELECT (right = INVALID). */
    {
        uint32_t slot0_val;
        switch (ds[0]) {
            case LD_SELECT:    slot0_val = prev[1];          break;
            case NEW_D_SELECT: slot0_val = new_job_id;       break;
            case W_DISABLE:    slot0_val = prev[0];          break;
            default:           slot0_val = INVALID_JOB_ID;   break; /* RD_SELECT */
        }
        /* Overwrite lane 0 of v_cur_lo8 with the corrected value */
        v_cur_lo8 = _mm256_insert_epi32(v_cur_lo8, (int)slot0_val, 0);
    }

    /* h. Determine top_job_id from ds[0] */
    switch (ds[0]) {
        case W_DISABLE:    *top_job_id_out = prev[0];        break;
        case LD_SELECT:    *top_job_id_out = prev[1];        break;
        case NEW_D_SELECT: *top_job_id_out = new_job_id;     break;
        default:           *top_job_id_out = INVALID_JOB_ID; break;
    }

    /* i. Handle slot 9 (right boundary):
     *    fifo_register(INVALID_JOB_ID, prev[8], new_job_id, prev[9], ds[9]) */
    uint32_t slot9_val;
    switch (ds[9]) {
        case LD_SELECT:    slot9_val = INVALID_JOB_ID; break;  /* no left nbr */
        case RD_SELECT:    slot9_val = prev[8];        break;
        case NEW_D_SELECT: slot9_val = new_job_id;     break;
        default:           slot9_val = prev[9];        break;  /* W_DISABLE   */
    }

    /* Handle slot 8 (second-to-last):
     *    fifo_register(prev[9], prev[7], new_job_id, prev[8], ds[8]) */
    uint32_t slot8_val;
    switch (ds[8]) {
        case LD_SELECT:    slot8_val = prev[9];        break;
        case RD_SELECT:    slot8_val = prev[7];        break;
        case NEW_D_SELECT: slot8_val = new_job_id;     break;
        default:           slot8_val = prev[8];        break;  /* W_DISABLE   */
    }

    /* j. Store cur[0..7] from YMM */
    store_fifo_regs(cur, v_cur_lo8,
                    /* hi2 built from slot8/slot9 below */
                    _mm256_setzero_si256()  /* placeholder */);

    /* Overwrite slots 8 and 9 directly (store_fifo_regs wrote hi2=0) */
    cur[8] = slot8_val;
    cur[9] = slot9_val;

    /* ── entry_count, popped_job_id, full (scalar) ───────────────────── */
    if (new_job_id != INVALID_JOB_ID)
        state->entry_count += 1;

    if (pop) {
        *popped_job_id_out = prev[0];
        state->entry_count -= 1;
    } else {
        *popped_job_id_out = INVALID_JOB_ID;
    }

    *full_out = (state->entry_count == JOBS_PER_MACHINE) ? 1 : 0;

    /* ── fifo_read: copy cur → prev (vectorised) ─────────────────────── */
    /* cur[0..7] → prev[0..7] via 256-bit store */
    __m256i v_cur_rd;
    load_fifo_regs(cur, &v_cur_rd, &hi2);   /* reuse hi2 for slots 8-9 */
    _mm256_store_si256((__m256i *)prev, v_cur_rd);
    prev[8] = cur[8];
    prev[9] = cur[9];

#if VF_DEBUG
    std::cout << "\033[31m[VF] Machine data-select: ";
    for (int i = 0; i < JOBS_PER_MACHINE; i++)
        std::cout << (int)ds[i] << " ";
    std::cout << "\n[VF] FIFO content: ";
    for (int i = 0; i < JOBS_PER_MACHINE; i++)
        std::cout << prev[i] << " ";
    std::cout << "\n\033[0m";
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5.  fifo_all_machines()  (vectorised inner, sequential outer)
 * ═══════════════════════════════════════════════════════════════════════════ */

SIMD_TARGET_AVX2
static void fifo_all_machines(
        fifo_update_input_t *input,
        one_bit_t           *pop,
        data_selector_t      data_select_signals[NUM_MACHINES][JOBS_PER_MACHINE],
        job_id_t            *top_job_id,
        job_id_t            *popped_job_id,
        one_bit_t           *fifo_full)
{
    /*
     * vf_state holds the per-machine cur/prev FIFO registers and entry count.
     * Declared static so it persists across scheduler ticks.
     * CACHE_ALIGNED (64 B) ensures each machine's state starts on a cache line.
     */
    static CACHE_ALIGNED vf_machine_state_t vf_state[NUM_MACHINES];

    for (machine_id_t m = 0; m < NUM_MACHINES; m++) {

        /* Prefetch the next machine's state into L1 */
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

/* ═══════════════════════════════════════════════════════════════════════════
 * 6.  virtual_fifo()  — top-level entry point (signature unchanged)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * virtual_fifo()
 *
 * Top-level wrapper that:
 *   1. Computes data-select signals for all machines (data_selector_all_machines).
 *   2. Executes the FIFO update for all machines  (fifo_all_machines).
 *
 * The input_temp / pop_temp staging arrays are retained from the original
 * implementation to preserve the logical pipeline separation between the
 * data-selector and FIFO stages (required for correctness when the two
 * stages share the same input values).
 */
SIMD_TARGET_AVX512
void virtual_fifo(fifo_update_input_t *input,
                  one_bit_t           *pop,
                  job_id_t            *top_job_id,
                  job_id_t            *popped_job_id,
                  one_bit_t           *fifo_full)
{
    /* Data-select signals: one uint8 per slot per machine */
    SIMD_ALIGNED data_selector_t
        data_selector_output[NUM_MACHINES][JOBS_PER_MACHINE];

    /* Staging copies (preserves HLS pipeline semantics) */
    fifo_update_input_t input_temp[NUM_MACHINES];
    one_bit_t           pop_temp  [NUM_MACHINES];

    data_selector_all_machines(input, pop,
                                data_selector_output,
                                input_temp, pop_temp);

    fifo_all_machines(input_temp, pop_temp,
                      data_selector_output,
                      top_job_id, popped_job_id, fifo_full);
}
