/*******************************************************************************
Vendor: Xilinx
Associated Filename: vadd.cpp
Purpose: VITIS vector addition

*******************************************************************************
Copyright (C) 2019 XILINX, Inc.

This file contains confidential and proprietary information of Xilinx, Inc. and
is protected under U.S. and international copyright and other intellectual
property laws.

DISCLAIMER
This disclaimer is not a license and does not grant any rights to the materials
distributed herewith. Except as otherwise provided in a valid license issued to
you by Xilinx, and to the maximum extent permitted by applicable law:
(1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX
HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR
FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
in contract or tort, including negligence, or under any other theory of
liability) for any loss or damage of any kind or nature related to, arising under
or in connection with these materials, including for any direct, or any indirect,
special, incidental, or consequential loss or damage (including loss of data,
profits, goodwill, or any type of loss or damage suffered as a result of any
action brought by a third party) even if such damage or loss was reasonably
foreseeable or Xilinx had been advised of the possibility of the same.

CRITICAL APPLICATIONS
Xilinx products are not designed or intended to be fail-safe, or for use in any
application requiring fail-safe performance, such as life-support or safety
devices or systems, Class III medical devices, nuclear facilities, applications
related to the deployment of airbags, or any other applications that could lead
to death, personal injury, or severe property or environmental damage
(individually and collectively, "Critical Applications"). Customer assumes the
sole risk and liability of any use of Xilinx products in Critical Applications,
subject only to applicable laws and regulations governing limitations on product
liability.

THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT
ALL TIMES.

*******************************************************************************/

//#include "ext_mem_host.h"
#include "data_types.hpp"
#include "extm_data_types.hpp"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <system_error>
#include "top_modules.hpp"
#include <sstream>

static const int DATA_SIZE = MEM_DATA_SIZE; //Num of jobs generated per tb //Except reduced to probing purposes

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input file>" << " <output file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << PRINT_BOLD << PRINT_UNDERLINE << "Starting Stochastic Online Scheduler\n" << PRINT_RESET;
    std::cout << "Data Size:" << DATA_SIZE << std::endl;

    std::string input_file_name = argv[1];
    std::string output_file_name = argv[2];

    std::cout << PRINT_YELLOW << "Input file used: " << input_file_name << PRINT_RESET << std::endl;

    std::cout << PRINT_RED << "Intializing Job ID manager\n" << PRINT_COLOR_END;
    // Compute the size of array in bytes
    size_t size_in  = sizeof(scheduler_interface_input_t);
    size_t size_out = sizeof(scheduler_interface_output_t); //Five machines, with 16 bit counters
    job_id_manager id_manager = job_id_manager();

    std::cout << PRINT_GREEN << "Job ID manager intialized\n" << PRINT_COLOR_END;

    // We then need to map our OpenCL buffers to get the pointers
    scheduler_interface_input_t ptr_in;
    scheduler_interface_output_t ptr_out;

    std::ifstream file_in(input_file_name);
    if (!file_in) {
        std::cout << PRINT_RED << "Could Not Find File" << std::endl; /*error*/
        PRINT_FAILURE;
    } else {
        std::cout << PRINT_GREEN << "Found Input File" << std::endl;
    }

    std::cout << PRINT_YELLOW << "Creating output file: " << output_file_name << std::endl;

    std::ofstream output_file(output_file_name, std::ios::out);

    if (!(output_file.is_open())) {
        std::cout << PRINT_BOLD << PRINT_RED << "Error in creating output file" << PRINT_RESET << std::endl;
        PRINT_FAILURE;
        exit(EXIT_FAILURE);
    }


    int initial_tick = 0;
    int num_jobs_machine[NUM_MACHINES] = {0};

    std::string line;
    std::getline(file_in, line);
    std::istringstream FIFOss(line);


    std::cout << "Starting\n" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < TOTAL_NUM_JOBS; j+=DATA_SIZE) {
        std::cout << PRINT_YELLOW << "TB Read" << std::endl << PRINT_RESET;
        std::cout << "Jobs Scheduled: " << j << std::endl;
        std::cout << "Initial Tick for this run: " << initial_tick << std::endl;
        for (int i = 0; i < DATA_SIZE; i++){
            std::getline(file_in, line); //read next line
            std::istringstream ss(line);

            //new empty job
            new_job_data_host_t x = {0};

            int y;
            ss >> y;
            //std::cout << y << " ";
            x.job_data.weight = y;
            //std::cout << static_cast<unsigned int>(x.weight) << std::endl;

            //std::cout << y << " ";
            for (machine_id_t machine = 0; machine < NUM_MACHINES; machine++) {
                ss >> y;
                x.job_data.processing_time[machine] = y;
            //std::cout << static_cast<unsigned int>(x.proc_time) << std::endl;
            }

            for (machine_id_t machine = 0; machine < NUM_MACHINES; machine++) {
                ss >> y;
                x.job_data.alpha_j[machine] = y;
            //std::cout << static_cast<unsigned int>(x.proc_time) << std::endl;
            }

            ss >> y;
            //std::cout << y << " ";
            x.release_tick = y;
            //std::cout << static_cast<unsigned int>(x.release_tick) << std::endl;

            //std::cout << std::endl;

            x.job_data.job_id = id_manager.assign_id(x.release_tick);

            ptr_in.new_job_table[i] = x;
        }

        ptr_in.initial_tick = initial_tick;

	    schedule_jobs(ptr_in.new_job_table, ptr_in.initial_tick, &ptr_out);
     
        for (int i = 0; i < NUM_MACHINES; i++) {
            num_jobs_machine[i] += ptr_out.num_jobs[i];
        }
        initial_tick = ptr_out.final_tick;
        id_manager.reset();

        std::cout << PRINT_GREEN << "\nJobs per machine Calculated, Writing output to the file: " << output_file_name << std::endl << PRINT_COLOR_END;

        for (int i = 0; i < DATA_SIZE; i++) {
            machine_id_t machine = ptr_out.scheduled_jobs[i+1].machine_scheduled;
            output_file << ptr_in.new_job_table[i].release_tick << " ";
            output_file << (int)machine << " ";
            output_file << ptr_out.scheduled_jobs[i+1].popped_tick << " ";
            output_file << (int)ptr_in.new_job_table[i].job_data.processing_time[machine] << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\nFinished\n" << std::endl;

    std::cout << PRINT_GREEN << "Writing the number of jobs per machien to output file: " << output_file_name << std::endl << PRINT_COLOR_END;
    for (int i = 0; i < NUM_MACHINES; i++) {
        output_file << num_jobs_machine[i] << " ";
    }

    output_file.close();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time taken by Kernel: " << duration.count() << " microseconds" << std::endl;

    //OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_tb_in, ptr_in));
    //OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_out, ptr_out));
    //OCL_CHECK(err, err = q.finish());

    std::cout << "Unmapped" << std::endl;

    return (EXIT_SUCCESS);
}
