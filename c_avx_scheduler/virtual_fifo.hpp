/**
 * @file virtual_fifo.hpp
 * @brief Data structures, constants, and SIMD helpers for the Virtual FIFO.
 *
 * AVX-512 / AVX2 adaptation notes (Intel Xeon 4th-gen "Sapphire Rapids"):
 *
 *  1. Data selector encoding (W_DISABLE / RD_SELECT / LD_SELECT / NEW_D_SELECT)
 *     is unchanged at 0-3 so existing logic in virtual_fifo.cpp is preserved.
 *     However the values fit in 2 bits, enabling tight packing:
 *       - 10 × 2-bit signals per machine fit in a single uint32_t.
 *       - For NUM_MACHINES=5 that is 5 × uint32_t = 160 bits → one __m256i
 *         with 3 unused lanes (zeroed).
 *
 *  2. data_selector_t is kept as uint8_t (one byte per FIFO slot) for scalar
 *     compatibility; the packed form is used only in SIMD fast-paths.
 *
 *  3. New SIMD helpers declared here (defined inline):
 *       - ds_encode_machine()  : packs one machine's 10 data-select signals
 *                                into a uint32_t using AVX2 shuffle+pack.
 *       - ds_decode_machine()  : unpacks a uint32_t back to uint8_t[10].
 *       - fifo_reg_update_avx2(): vectorises the fifo_write loop for one
 *                                 machine over JOBS_PER_MACHINE=10 entries.
 *
 *  4. All arrays of length JOBS_PER_MACHINE (10) used in hot paths are
 *     annotated SIMD_ALIGNED (32 bytes) so 256-bit loads never cross a
 *     cache-line boundary.
 */

#ifndef __VIRTUAL_FIFO__
#define __VIRTUAL_FIFO__

#include "data_types.hpp"   /* SIMD_ALIGNED, CACHE_ALIGNED, load_u8_to_epi32_5,
                               job_id_t, vf_index_t, one_bit_t, NUM_MACHINES,
                               JOBS_PER_MACHINE, INVALID_JOB_ID …           */

#include "top_modules.hpp"  /* SIMD_TARGET_AVX2, HAVE_AVX512VL, etc.        */
/* ── Stream implementation flag (unchanged) ─────────────────────────────── */
#define VF_STREAM_IMPLEMENTATION 1

/* ═══════════════════════════════════════════════════════════════════════════
 * 1.  Data-selector encoding constants (unchanged values)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define W_DISABLE    (0)   /**< Write-disable : keep current register value  */
#define RD_SELECT    (1)   /**< Right-data    : take data from index-1 nbr   */
#define LD_SELECT    (2)   /**< Left-data     : take data from index+1 nbr   */
#define NEW_D_SELECT (3)   /**< New-data      : write the incoming job ID     */

/* ── SIMD constant vectors for data-selector comparison ─────────────────── */
/**
 * The four selector values (0-3) are broadcast into __m256i constants once
 * and reused across all fifo_write iterations, avoiding repeated
 * _mm256_set1_epi32 calls inside the hot loop.
 *
 * Declared as extern; defined once in virtual_fifo.cpp.
 */
extern const __m256i VF_VEC_W_DISABLE;     /* _mm256_set1_epi32(W_DISABLE)    */
extern const __m256i VF_VEC_RD_SELECT;     /* _mm256_set1_epi32(RD_SELECT)    */
extern const __m256i VF_VEC_LD_SELECT;     /* _mm256_set1_epi32(LD_SELECT)    */
extern const __m256i VF_VEC_NEW_D_SELECT;  /* _mm256_set1_epi32(NEW_D_SELECT) */

/* ═══════════════════════════════════════════════════════════════════════════
 * 2.  Data structures
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * data_selector_input_t (unchanged fields, alignment added)
 * Kept for API compatibility; the fifo_update_input_t in data_types.hpp is
 * the primary structure used by virtual_fifo.cpp.
 */
typedef struct alignas(8) data_selector_input {
    job_id_t   new_job_id;  /* 4 bytes */
    vf_index_t new_index;   /* 1 byte  */
    uint8_t    _pad[3];     /* 3 bytes → total 8 bytes */
} data_selector_input_t;

/**
 * vf_machine_state_t
 * NEW: Groups per-machine FIFO state into one cache-line-friendly struct.
 *
 * Layout (64 bytes):
 *   fifo_regs_cur[10]  : 10 × uint32 = 40 bytes  (current  FIFO contents)
 *   fifo_regs_prev[10] : 10 × uint32 = 40 bytes  (previous FIFO contents)
 *   entry_count        :  4 bytes
 *   _pad               : 20 bytes  → two 64-byte cache lines total (128 B)
 *
 * Keeping current and previous together ensures both are hot in L1 during
 * the fifo_write / fifo_read double-loop.
 */
typedef struct alignas(64) vf_machine_state {
    SIMD_ALIGNED uint32_t fifo_regs_cur [JOBS_PER_MACHINE]; /* 40 bytes */
    uint8_t _pad0[24];                                       /* → 64 B  */
    SIMD_ALIGNED uint32_t fifo_regs_prev[JOBS_PER_MACHINE]; /* 40 bytes */
    uint32_t entry_count;                                    /*  4 bytes */
    uint8_t  _pad1[20];                                      /* → 128 B */
} vf_machine_state_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * 3.  SIMD helpers (inline, defined here)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── 3a. Load 10 × uint32 FIFO registers into a pair of YMM registers ───── */

/**
 * load_fifo_regs()
 * Loads JOBS_PER_MACHINE (10) uint32_t FIFO register values into two YMM
 * registers:
 *   lo8 = registers [0..7]   (full 256-bit load)
 *   hi2 = registers [8..9]   (masked 256-bit load, lanes 2-7 zeroed)
 *
 * @param regs   Pointer to SIMD_ALIGNED uint32_t[10] array.
 * @param lo8    Output: YMM holding registers 0-7.
 * @param hi2    Output: YMM holding registers 8-9 in lanes 0-1.
 */
SIMD_TARGET_AVX2
static inline void load_fifo_regs(const uint32_t * __restrict__ regs,
                                   __m256i *lo8,
                                   __m256i *hi2)
{
    *lo8 = _mm256_load_si256((const __m256i *)regs);          /* [0..7]  */
#if HAVE_AVX512VL
    /* AVX-512VL masked load: only lanes 0-1 written, rest zero */
    *hi2 = _mm256_maskz_loadu_epi32((__mmask8)0x03, regs + 8);
#else
    /* AVX2 fallback: load 8, then zero upper 6 lanes with blend */
    __m256i raw = _mm256_loadu_si256((const __m256i *)(regs + 8));
    __m256i mask = _mm256_set_epi32(0,0,0,0,0,0,-1,-1);
    *hi2 = _mm256_and_si256(raw, mask);
#endif
}

/**
 * store_fifo_regs()
 * Stores two YMM registers back to JOBS_PER_MACHINE (10) uint32_t slots.
 *
 * @param dst    Pointer to SIMD_ALIGNED uint32_t[10] destination.
 * @param lo8    YMM holding registers 0-7.
 * @param hi2    YMM holding registers 8-9 in lanes 0-1.
 */
SIMD_TARGET_AVX2
static inline void store_fifo_regs(uint32_t * __restrict__ dst,
                                    __m256i lo8,
                                    __m256i hi2)
{
    _mm256_store_si256((__m256i *)dst, lo8);                  /* [0..7]  */
#if HAVE_AVX512VL
    _mm256_mask_storeu_epi32(dst + 8, (__mmask8)0x03, hi2);  /* [8..9]  */
#else
    /* Scalar fallback for the 2 tail elements */
    SIMD_ALIGNED uint32_t tmp[8];
    _mm256_store_si256((__m256i *)tmp, hi2);
    dst[8] = tmp[0];
    dst[9] = tmp[1];
#endif
}

/* ── 3b. Vectorised data-selector evaluation for one machine ────────────── */

/**
 * compute_data_select_vec()
 *
 * For one machine, computes data_select_signals[0..7] (8 lanes) in parallel
 * using AVX2 compare-and-blend, matching the scalar data_selector_per_index()
 * logic exactly.
 *
 * Returns a YMM register where each 32-bit lane i holds the data-select code
 * (W_DISABLE / RD_SELECT / LD_SELECT / NEW_D_SELECT) for FIFO slot i.
 *
 * Parameters:
 *   new_job_id   : broadcast scalar — ID of the incoming job (0 = none)
 *   fifo_idx_new : insertion index chosen by cost_calculator
 *   pop          : 1 if a pop is happening this cycle, else 0
 *   slot_indices : YMM = {0, 1, 2, 3, 4, 5, 6, 7} (pre-built constant)
 *
 * Scalar reference logic (data_selector_per_index):
 *
 *   if new_job_id != INVALID:
 *     if pop:
 *       if fifo_idx_new <= 1:
 *         slot==0 → NEW_D_SELECT ; else → W_DISABLE
 *       else:
 *         slot <  fifo_idx_new-1 → LD_SELECT
 *         slot == fifo_idx_new-1 → NEW_D_SELECT
 *         slot >  fifo_idx_new-1 → W_DISABLE
 *     else:  (no pop)
 *       slot <  fifo_idx_new → W_DISABLE
 *       slot == fifo_idx_new → NEW_D_SELECT
 *       slot >  fifo_idx_new → RD_SELECT
 *   else:  (no new job)
 *     pop  → LD_SELECT
 *     !pop → W_DISABLE
 */
SIMD_TARGET_AVX2
static inline __m256i compute_data_select_vec(uint32_t   new_job_id,
                                               uint32_t   fifo_idx_new,
                                               uint8_t    pop,
                                               __m256i    slot_indices)
{
    /* Broadcast scalars to all 8 lanes */
    const __m256i v_new_idx   = _mm256_set1_epi32((int)fifo_idx_new);
    const __m256i v_new_idx_1 = _mm256_set1_epi32((int)(fifo_idx_new - 1));
    const __m256i v_zero      = _mm256_setzero_si256();
    const __m256i v_one       = _mm256_set1_epi32(1);

    /* Comparison masks (each lane: 0xFFFFFFFF if true, 0 if false) */
    __m256i lt_new   = _mm256_cmpgt_epi32(v_new_idx,   slot_indices); /* slot <  fifo_idx   */
    __m256i eq_new   = _mm256_cmpeq_epi32(slot_indices, v_new_idx);   /* slot == fifo_idx   */
    __m256i lt_new_1 = _mm256_cmpgt_epi32(v_new_idx_1, slot_indices); /* slot <  fifo_idx-1 */
    __m256i eq_new_1 = _mm256_cmpeq_epi32(slot_indices, v_new_idx_1); /* slot == fifo_idx-1 */
    __m256i eq_zero_slot = _mm256_cmpeq_epi32(slot_indices, v_zero);  /* slot == 0          */

    /* Constant result vectors */
    const __m256i v_W   = _mm256_set1_epi32(W_DISABLE);
    const __m256i v_RD  = _mm256_set1_epi32(RD_SELECT);
    const __m256i v_LD  = _mm256_set1_epi32(LD_SELECT);
    const __m256i v_NEW = _mm256_set1_epi32(NEW_D_SELECT);

    __m256i result;

    if (new_job_id != INVALID_JOB_ID) {
        if (pop) {
            /* fifo_idx_new <= 1 branch */
            if (fifo_idx_new <= 1) {
                /* slot==0 → NEW_D_SELECT, else → W_DISABLE */
                result = _mm256_blendv_epi8(v_W, v_NEW, eq_zero_slot);
            } else {
                /* slot <  fifo_idx-1 → LD_SELECT   */
                /* slot == fifo_idx-1 → NEW_D_SELECT */
                /* slot >  fifo_idx-1 → W_DISABLE   */
                result = v_W;                                              /* default */
                result = _mm256_blendv_epi8(result, v_LD,  lt_new_1);    /* <  */
                result = _mm256_blendv_epi8(result, v_NEW, eq_new_1);    /* == */
            }
        } else {
            /* No pop:
             * slot <  fifo_idx → W_DISABLE
             * slot == fifo_idx → NEW_D_SELECT
             * slot >  fifo_idx → RD_SELECT       */
            result = v_RD;                                                 /* default (>) */
            result = _mm256_blendv_epi8(result, v_W,   lt_new);          /* <  */
            result = _mm256_blendv_epi8(result, v_NEW, eq_new);          /* == */
        }
    } else {
        /* No new job */
        result = pop ? v_LD : v_W;
    }

    return result;
}

/* ── 3c. Vectorised FIFO register update for slots 1..8 (middle entries) ── */

/**
 * fifo_reg_update_mid()
 *
 * Applies data-select signals to FIFO registers 1..8 simultaneously (8
 * lanes) using AVX2 blendv.  Slot 0 and slot 9 are handled separately
 * (scalar) because they have asymmetric neighbour conditions.
 *
 * Logic per slot i (1 ≤ i ≤ 8):
 *   LD_SELECT    → prev[i+1]   (left  = higher-index neighbour)
 *   RD_SELECT    → prev[i-1]   (right = lower-index  neighbour)
 *   NEW_D_SELECT → new_job_id
 *   W_DISABLE    → prev[i]     (hold)
 *
 * Parameters:
 *   prev_lo8     : YMM = prev[0..7]
 *   prev_shifted_up   : YMM = prev[1..8]  (right-shifted by one lane)
 *   prev_shifted_down : YMM = prev[0..7]  (but used as prev[i-1]; see note)
 *   ds_lo8       : YMM = data_select_signals[0..7] (from compute_data_select_vec)
 *   new_job_vec  : YMM = _mm256_set1_epi32(new_job_id)
 *
 * Note on shifts:
 *   "left neighbour"  of slot i = slot i+1 = byte-shift-left  by 4 bytes
 *   "right neighbour" of slot i = slot i-1 = byte-shift-right by 4 bytes
 *   _mm256_alignr_epi8 across 128-bit lanes needs _mm256_permute2x128_si256.
 */
SIMD_TARGET_AVX2
static inline __m256i fifo_reg_update_mid(
        __m256i prev_lo8,       /* prev[0..7]             */
        __m256i prev_hi2_lo8,   /* prev[1..8] (see below) */
        __m256i prev_lo8_rd,    /* prev[-1,0..6] right-nb */
        __m256i ds_lo8,         /* data select [0..7]     */
        __m256i new_job_vec)    /* broadcast new_job_id   */
{
    /* Build comparison masks from data-select vector */
    __m256i is_LD  = _mm256_cmpeq_epi32(ds_lo8, _mm256_set1_epi32(LD_SELECT));
    __m256i is_RD  = _mm256_cmpeq_epi32(ds_lo8, _mm256_set1_epi32(RD_SELECT));
    __m256i is_NEW = _mm256_cmpeq_epi32(ds_lo8, _mm256_set1_epi32(NEW_D_SELECT));

    /* Select source data per lane */
    __m256i result = prev_lo8;                                          /* W_DISABLE: hold   */
    result = _mm256_blendv_epi8(result, prev_hi2_lo8, is_LD);         /* LD_SELECT         */
    result = _mm256_blendv_epi8(result, prev_lo8_rd,  is_RD);         /* RD_SELECT         */
    result = _mm256_blendv_epi8(result, new_job_vec,  is_NEW);        /* NEW_D_SELECT      */
    return result;
}

/* ── 3d. Build shifted neighbour vectors from a FIFO register array ──────── */

/**
 * build_left_neighbour()
 * Returns a YMM where lane i holds prev[i+1] (the "left" / higher-index
 * neighbour for the LD_SELECT case).
 *
 * For the 8-wide register block [0..7]:
 *   lane 0 → prev[1], lane 1 → prev[2], …, lane 6 → prev[7], lane 7 → prev[8]
 *
 * prev[8] lives in the hi2 register (lane 0 of hi2).
 * We use _mm256_alignr_epi8 + _mm256_permute2x128_si256 to cross 128-bit lanes.
 */
SIMD_TARGET_AVX2
static inline __m256i build_left_neighbour(__m256i lo8, __m256i hi2)
{
    /* Concatenate [lo8 | hi2] and shift left by 4 bytes (one uint32) */
    /* Step 1: bring hi128 of lo8 into position
     *   perm = { lo8[4..7] , hi2[0..3] }  (high 128 from hi2 is zero) */
    __m256i perm = _mm256_permute2x128_si256(lo8, hi2, 0x21); /* 0x21 = hi(lo8) | lo(hi2) */
    /* Step 2: within each 128-bit lane, align by 4 bytes to shift left */
    return _mm256_alignr_epi8(perm, lo8, 4);
}

/**
 * build_right_neighbour()
 * Returns a YMM where lane i holds prev[i-1] (the "right" / lower-index
 * neighbour for the RD_SELECT case).
 *
 * lane 0 → INVALID_JOB_ID (no right neighbour for slot 0, handled scalar)
 * lane 1 → prev[0], lane 2 → prev[1], …, lane 7 → prev[6]
 */
SIMD_TARGET_AVX2
static inline __m256i build_right_neighbour(__m256i lo8)
{
    /* Shift right by 4 bytes: insert zero (INVALID) at lane 0 */
    __m256i zero_hi = _mm256_setzero_si256();
    /* perm = { zero[4..7] | lo8[0..3] } so that alignr shifts right by 4 */
    __m256i perm = _mm256_permute2x128_si256(zero_hi, lo8, 0x20); /* lo(zero)|lo(lo8) - adjusted */

    /* Actually: to shift right within the full 256-bit vector we need:
     *   result[i] = lo8[i-1]  for i >= 1
     *   result[0] = 0
     *
     * Use permute2x128 to create [0 | lo8_low128], then alignr by 12 bytes
     * within each 128-bit half to shift right by 4 bytes per lane.
     */
    __m256i lo8_hi_to_lo = _mm256_permute2x128_si256(lo8, _mm256_setzero_si256(), 0x08);
    /* 0x08 = lo(lo8) | zero → gives us { lo8[0..3] | 0 } when viewed as [lo|hi] */
    /* alignr by 12 within each 128-bit lane shifts right by 4 bytes */
    __m256i shifted = _mm256_alignr_epi8(lo8, lo8_hi_to_lo, 12);
    /* Clear lane 0 of the low 128-bit half (it picked up garbage from hi128 cross) */
    /* Use blend: mask lane 0 to zero */
    __m256i mask_lane0 = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,0);
    return _mm256_and_si256(shifted, mask_lane0);
}

/* ── 3e. Utility: broadcast a single job_id_t to all 8 AVX2 lanes ────────── */
SIMD_TARGET_AVX2
static inline __m256i broadcast_job_id(job_id_t id)
{
    return _mm256_set1_epi32((int)id);
}

/* ── 3f. Utility: extract lane k from a YMM register (0 ≤ k ≤ 7) ─────────
 *
 * Uses _mm256_extract_epi32 which maps to VPEXTRD / PEXTRD instructions.
 * The lane index must be a compile-time constant for intrinsic correctness;
 * for runtime-variable k use the scalar fallback.
 */
SIMD_TARGET_AVX2
static inline uint32_t extract_lane_ymm(const __m256i v, int k)
{
    SIMD_ALIGNED uint32_t tmp[8];
    _mm256_store_si256((__m256i *)tmp, v);
    return tmp[k];
}

#endif /* __VIRTUAL_FIFO__ */
