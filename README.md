# TRETS Software SOS Implementation
This repository contains the software-based baseline implementations of the Stochastic Online Scheduling (SOS) algorithm, developed to benchmark the hardware-accelerated STANNIC architecture.

It includes a standard single-threaded C++ baseline alongside highly optimized AVX-512 SIMD implementations. These baselines evaluate the algorithm's scalability and iteration latency across various heterogeneous machine configurations (e.g., 5, 10, 20, 50, and 100+ machines).

## Repository Structure
* `c_scheduler/:` Contains the standard, unoptimized single-threaded C++ implementation of the SOS algorithm.

* `c_avx_scheduler/:` Contains the SIMD-accelerated C++ implementation utilizing AVX-512 intrinsics.

* `c_avx_templated_scheduler/:` Contains a templated version of the AVX-512 scheduler where the system configuration (Machines M and Jobs J) are defined at compile-time via macros for maximum optimization.

* `Example_Workload/:` Contains sample simulation input files (e.g., sim_5_machines.txt) to test the schedulers.

## Prerequisites
To compile and run the AVX-512 schedulers, your build environment and host CPU must support the targeted AVX-512 instruction set.

Compiler: GCC (g++). Version 12 or higher is recommended to natively support the '-march=sapphirerapids' flag. A fallback to standard AVX-512 flags is included for older compilers.

OS: Linux-based environment recommended. (Note: AMX instructions are not utilized in this repository to maintain compatibility with older Linux kernels, such as RHEL 8.4/Kernel 4.18).

You can verify your CPU's compatibility by running the included utility target from any of the AVX directories:

```Bash
make check_cpu
```
This will probe your machine's CPUID and report support for AVX2, AVX-512F, AVX-512BW, and AVX-512VL.

## Build Instructions
Navigate into the directory of the scheduler you wish to build and use the provided Makefile.

1. Standard Single-Threaded Baseline
To compile the standard baseline exactly as reported in our manuscript rebuttal, navigate to the c_scheduler/ directory and run the following single-line command:

```Bash
cd c_scheduler
g++ -Wall -Wno-unused-label *.cpp -o scheduler_5_10
```

3. Standard AVX-512 Scheduler
Navigate to the c_avx_scheduler/ directory and use the provided Makefile:

```Bash
cd c_avx_scheduler
make release
```
This will generate the optimized executable bin/scheduler_avx512.

3. Templated AVX-512 Scheduler
The templated scheduler requires you to define the number of Machines (M) and Jobs Per Machine (J) at compile time. If no arguments are passed, it defaults to M=5 and J=10.

```Bash
cd c_avx_templated_scheduler
make release M=10 J=20
```
This will generate the executable bin/scheduler_avx512_10_20.

Usage
Once compiled, you can run the schedulers by passing an input workload file and specifying a name for the output results file.

From within the directory where you compiled your executable, use the following format:

```Bash
./<executable> <sample input job in repo> <output txt name of choice>
```
Example Run (using the standard baseline and the provided 5-machine workload):

```Bash
# Assuming you are still in the c_scheduler/ directory:
./scheduler_5_10 ../Example_Workload/sim_5_machines.txt my_results_output.txt
```

## Verification
To verify that the compiled AVX binaries successfully incorporated the vectorized EVEX instructions, use the included disasm target. This runs objdump and searches for unique AVX-512 registers (like zmm) and vectorized operations:

```Bash
make disasm
```
Useful Makefile Commands
`make release:` Builds the highly optimized release binary (-O3, -funroll-loops, etc.).

`make debug:` Builds the binary with AddressSanitizer, UBSan, and debug symbols enabled.

`make info:` Displays the current compiler version, standard, and ISA flags being used.

`make clean:` Removes the build/ and bin/ directories.
