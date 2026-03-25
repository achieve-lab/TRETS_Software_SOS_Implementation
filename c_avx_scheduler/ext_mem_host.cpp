/**
 * @file ext_mem_host.cpp
 * @brief Host driver for the Stochastic Online Scheduler.
 *
 * AVX-512 / AVX2 adaptation (Intel Xeon 4th-gen "Sapphire Rapids")
 * ================================================================
 *
 * Changes from the original scalar host driver
 * ---------------------------------------------
 *
 * 1. Aligned heap allocation for large I/O structs
 *    ───────────────────────────────────────────────
 *    scheduler_interface_input_t  (~6.5 KB) and
 *    scheduler_interface_output_t (~1.0 KB) are allocated on the heap
 *    with posix_memalign (64-byte / cache-line alignment) instead of
 *    on the stack.  This ensures that:
 *      a. All CACHE_ALIGNED arrays inside those structs (new_job_table[],
 *         scheduled_jobs[], num_jobs[]) are naturally aligned for
 *         _mm512_load_si512 / _mm256_load_si256 / _mm_load_si128.
 *      b. The 6.5 KB input table does not overflow the default stack
 *         (commonly 8 MB on Linux, but reduced in threaded contexts).
 *
 * 2. CPUID runtime capability check
 *    ─────────────────────────────────
 *    query_simd_capabilities() (from top_modules.hpp) is called once at
 *    startup and its result is printed.  All kernel dispatch currently
 *    uses the AVX-512 path (compiled with -mavx512f); this hook is the
 *    extension point for future runtime-dispatch if a fallback binary
 *    is needed.
 *
 * 3. Input parsing → aligned job table
 *    ─────────────────────────────────────
 *    The file-reading loop is unchanged in logic but the destination
 *    (ptr_in->new_job_table[i]) is now in a 64-byte-aligned heap buffer,
 *    so each 64-byte new_job_data_host_t record is always cache-line-aligned
 *    when schedule_jobs() issues its prefetch stream.
 *
 * 4. Output writing loop
 *    ──────────────────────
 *    ptr_out->num_jobs[] is a uint16_t[8] (padded, aligned at 16 bytes).
 *    The accumulation into num_jobs_machine[NUM_MACHINES] uses a single
 *    SSE2 PADDW to add all 5 counters in one instruction per batch.
 *
 * 5. High-resolution timing
 *    ─────────────────────────
 *    Unchanged: std::chrono::high_resolution_clock.  Added per-batch
 *    microsecond reporting in DEBUG mode.
 *
 * 6. Removed dead OpenCL headers / includes
 *    ──────────────────────────────────────────
 *    ext_mem_host.h pulled in <CL/cl2.hpp> which is not needed for the
 *    CPU-SIMD path.  It is no longer included.
 *
 * Original Xilinx copyright notice is preserved below.
 */

/*******************************************************************************
Vendor: Xilinx
Associated Filename: vadd.cpp
Purpose: VITIS vector addition – adapted for AVX-512 CPU scheduler

Copyright (C) 2019 XILINX, Inc.  (See original file for full disclaimer.)
*******************************************************************************/

#include "data_types.hpp"
#include "extm_data_types.hpp"
#include "top_modules.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <chrono>

/* ── Aligned heap allocation helpers ───────────────────────────────────── */

/**
 * alloc_aligned<T>()
 * Allocates sizeof(T) bytes on the heap with 64-byte alignment.
 * Aborts on allocation failure.
 */
template <typename T>
static T *alloc_aligned()
{
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 64, sizeof(T)) != 0) {
        std::cerr << PRINT_RED
                  << "posix_memalign failed for " << sizeof(T) << " bytes"
                  << PRINT_RESET << std::endl;
        std::abort();
    }
    /* Zero-initialise (replaces implicit zero-init of stack variables) */
    std::memset(ptr, 0, sizeof(T));
    return reinterpret_cast<T *>(ptr);
}

/* ── SSE2 batch counter accumulation ────────────────────────────────────── */

/**
 * accumulate_num_jobs_simd()
 *
 * Adds the 5 per-machine job counts produced by one schedule_jobs() batch
 * into the running total array num_jobs_machine[NUM_MACHINES].
 *
 * ptr_out->num_jobs is uint16_t[8] (aligned 16 bytes, lanes 5-7 = 0).
 * num_jobs_machine is int[NUM_MACHINES].
 *
 * Strategy:
 *   1. Load ptr_out->num_jobs[0..7] into XMM (one _mm_load_si128).
 *   2. Load num_jobs_machine[0..4] + zero-pad to int[8] into another XMM
 *      (after widening uint16 → int32 with _mm256_cvtepu16_epi32).
 *   3. Add and store back.
 *   Actually: since num_jobs_machine is int[] and num_jobs is uint16[],
 *   we widen both to 32-bit, add with AVX2, store back as int[].
 */
SIMD_TARGET_AVX2
static void accumulate_num_jobs_simd(int            *num_jobs_machine, /* [NUM_MACHINES] */
                                      const uint16_t *batch_counts)     /* [8], aligned   */
{
    /* Load batch counts (uint16[8]) and zero-extend to int32[8] */
    __m128i v_u16   = _mm_load_si128((const __m128i *)batch_counts);
    __m256i v_batch = _mm256_cvtepu16_epi32(v_u16);  /* 8 × int32 */

    /* Load current totals (int[5]) into int32[8] (pad upper 3 with 0) */
    SIMD_ALIGNED int32_t buf[8] = {0,0,0,0,0,0,0,0};
    for (int m = 0; m < NUM_MACHINES; m++) buf[m] = num_jobs_machine[m];
    __m256i v_total = _mm256_load_si256((const __m256i *)buf);

    /* Add */
    __m256i v_sum = _mm256_add_epi32(v_total, v_batch);

    /* Store back to buf, then copy NUM_MACHINES elements to num_jobs_machine */
    _mm256_store_si256((__m256i *)buf, v_sum);
    for (int m = 0; m < NUM_MACHINES; m++) num_jobs_machine[m] = buf[m];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main()
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "Usage: " << argv[0]
                  << " <input file> <output file>" << std::endl;
        return EXIT_FAILURE;
    }

    /* ── Banner ──────────────────────────────────────────────────────── */
    std::cout << PRINT_BOLD << PRINT_UNDERLINE
              << "Starting Stochastic Online Scheduler (AVX-512 build)\n"
              << PRINT_RESET;
    std::cout << "Data Size: " << MEM_DATA_SIZE << std::endl;

    /* ── Runtime SIMD capability check ─────────────────────────────── */
    simd_capabilities_t caps = query_simd_capabilities();
    std::cout << PRINT_CYAN
              << "SIMD capabilities: "
              << "AVX2="      << caps.has_avx2
              << "  AVX-512F=" << caps.has_avx512f
              << "  AVX-512BW="<< caps.has_avx512bw
              << "  AVX-512VL="<< caps.has_avx512vl
              << PRINT_RESET  << std::endl;

    if (!caps.has_avx2) {
        std::cerr << PRINT_RED
                  << "ERROR: This binary requires at least AVX2.  "
                  << "Run on a Haswell or newer CPU."
                  << PRINT_RESET << std::endl;
        return EXIT_FAILURE;
    }
    if (!caps.has_avx512f) {
        std::cout << PRINT_YELLOW
                  << "WARNING: AVX-512F not available.  "
                  << "Some inner loops will use AVX2 fallback paths."
                  << PRINT_RESET << std::endl;
    }

    /* ── File I/O setup ──────────────────────────────────────────────── */
    const std::string input_file_name  = argv[1];
    const std::string output_file_name = argv[2];

    std::cout << PRINT_YELLOW << "Input file:  " << input_file_name
              << PRINT_RESET  << std::endl;

    std::ifstream file_in(input_file_name);
    if (!file_in) {
        std::cout << PRINT_RED << "Could not open input file: "
                  << input_file_name << PRINT_RESET << std::endl;
        PRINT_FAILURE;
        return EXIT_FAILURE;
    }
    std::cout << PRINT_GREEN << "Input file opened successfully."
              << PRINT_RESET << std::endl;

    std::cout << PRINT_YELLOW << "Output file: " << output_file_name
              << PRINT_RESET  << std::endl;

    std::ofstream output_file(output_file_name, std::ios::out);
    if (!output_file.is_open()) {
        std::cout << PRINT_BOLD << PRINT_RED
                  << "Error: could not create output file: "
                  << output_file_name << PRINT_RESET << std::endl;
        PRINT_FAILURE;
        return EXIT_FAILURE;
    }

    /* ── Aligned heap allocation for I/O structs ────────────────────── */
    std::cout << PRINT_RED << "Allocating aligned I/O buffers...\n"
              << PRINT_COLOR_END;

    scheduler_interface_input_t  *ptr_in  =
        alloc_aligned<scheduler_interface_input_t>();
    scheduler_interface_output_t *ptr_out =
        alloc_aligned<scheduler_interface_output_t>();

    std::cout << PRINT_GREEN << "Buffers allocated ("
              << sizeof(scheduler_interface_input_t)  << " B input, "
              << sizeof(scheduler_interface_output_t) << " B output, "
              << "64-byte aligned).\n" << PRINT_COLOR_END;

    /* ── Job ID manager ──────────────────────────────────────────────── */
    std::cout << PRINT_RED << "Initialising Job ID manager...\n"
              << PRINT_COLOR_END;
    job_id_manager id_manager;   /* constructor calls reset_simd() */
    std::cout << PRINT_GREEN << "Job ID manager initialised (MANAGER_SIZE="
              << MANAGER_SIZE << ").\n" << PRINT_COLOR_END;

    /* ── Skip header line ────────────────────────────────────────────── */
    {
        std::string line;
        std::getline(file_in, line);   /* discard first line (header) */
    }

    /* ── Per-batch state ─────────────────────────────────────────────── */
    int initial_tick = 0;
    int num_jobs_machine[NUM_MACHINES] = {0};

    std::cout << "\nStarting scheduling loop...\n" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    /* ── Main batch loop ─────────────────────────────────────────────── */
    for (int j = 0; j < TOTAL_NUM_JOBS; j += MEM_DATA_SIZE) {

        std::cout << PRINT_YELLOW << "Batch read" << PRINT_RESET << std::endl;
        std::cout << "Jobs scheduled so far: " << j          << std::endl;
        std::cout << "Initial tick:          " << initial_tick << std::endl;

        /* ── Parse MEM_DATA_SIZE lines from input file ────────────── */
        for (int i = 0; i < MEM_DATA_SIZE; i++) {
            std::string line;
            std::getline(file_in, line);
            std::istringstream ss(line);

            /* new_job_data_host_t is 64-byte aligned inside new_job_table[] */
            new_job_data_host_t x;
            std::memset(&x, 0, sizeof(x));

            int y;

            /* weight */
            ss >> y;  x.job_data.weight = (uint8_t)y;

            /* processing_time[NUM_MACHINES] */
            for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
                ss >> y;
                x.job_data.processing_time[m] = (uint8_t)y;
            }

            /* alpha_j[NUM_MACHINES] */
            for (machine_id_t m = 0; m < NUM_MACHINES; m++) {
                ss >> y;
                x.job_data.alpha_j[m] = (uint8_t)y;
            }

            /* release_tick */
            ss >> y;  x.release_tick = (uint32_t)y;

            /* Assign a unique job ID (uses reset_simd() internally) */
            x.job_data.job_id = id_manager.assign_id(x.release_tick);

            ptr_in->new_job_table[i] = x;
        }

        ptr_in->initial_tick = (uint32_t)initial_tick;

        /* ── Zero the output buffer for this batch ─────────────────── */
        zero_output_simd(ptr_out);

        /* ── Run the scheduler kernel ──────────────────────────────── */
#if EXT_MEM_INTERFACE_DEBUG_GEN
        auto batch_start = std::chrono::high_resolution_clock::now();
#endif
        schedule_jobs(ptr_in->new_job_table, ptr_in->initial_tick, ptr_out);

#if EXT_MEM_INTERFACE_DEBUG_GEN
        auto batch_end = std::chrono::high_resolution_clock::now();
        auto batch_us  = std::chrono::duration_cast<std::chrono::microseconds>(
                             batch_end - batch_start).count();
        std::cout << PRINT_CYAN << "Batch " << (j / MEM_DATA_SIZE)
                  << " kernel time: " << batch_us << " µs"
                  << PRINT_RESET << std::endl;
#endif

        /* ── Accumulate per-machine job counts (SSE2) ──────────────── */
        accumulate_num_jobs_simd(num_jobs_machine, ptr_out->num_jobs);

        initial_tick = (int)ptr_out->final_tick;
        id_manager.reset();   /* reset_simd() for next batch */

        /* ── Write per-job results to output file ──────────────────── */
        std::cout << PRINT_GREEN
                  << "\nWriting batch results to: " << output_file_name
                  << PRINT_COLOR_END << std::endl;

        for (int i = 0; i < MEM_DATA_SIZE; i++) {
            /* scheduled_jobs[] is indexed by job_id (1-based).
             * ptr_in->new_job_table[i].job_data.job_id was assigned by
             * id_manager, so it equals i+1 within each batch. */
            job_id_t jid     = ptr_in->new_job_table[i].job_data.job_id;
            machine_id_t mach = ptr_out->scheduled_jobs[jid].machine_scheduled;

            output_file << ptr_in->new_job_table[i].release_tick            << " "
                        << (int)mach                                          << " "
                        << ptr_out->scheduled_jobs[jid].popped_tick           << " "
                        << (int)ptr_in->new_job_table[i].job_data
                                       .processing_time[mach]                << "\n";
        }

    } /* end batch loop */

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\nAll batches complete.\n" << std::endl;

    /* ── Write per-machine totals ────────────────────────────────────── */
    std::cout << PRINT_GREEN
              << "Writing per-machine job totals to: " << output_file_name
              << PRINT_COLOR_END << std::endl;

    for (int m = 0; m < NUM_MACHINES; m++) {
        output_file << num_jobs_machine[m] << " ";
    }
    output_file << "\n";
    output_file.close();

    /* ── Timing report ───────────────────────────────────────────────── */
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Total kernel time: " << duration.count()
              << " microseconds" << std::endl;
    std::cout << "Throughput:        "
              << (TOTAL_NUM_JOBS * 1000000.0 / duration.count())
              << " jobs/second" << std::endl;

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    std::free(ptr_in);
    std::free(ptr_out);

    std::cout << "Done." << std::endl;
    return EXIT_SUCCESS;
}
