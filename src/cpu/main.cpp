// main.cpp
// Entry point for the advanced FPGA-GPU HFT system.

#include <iostream>
#include "pipeline_manager.h"

int main() {
    std::cout << "[INFO] Starting Advanced FPGA-GPU HFT System..." << std::endl;

    // Create a pipeline manager for a certain batch size
    PipelineManager pipeline(1024);

    // Start threads
    pipeline.start();

    // Let the pipeline run for ~10 seconds as a demo
    std::this_thread::sleep_for(std::chrono::seconds(10));

    // Stop pipeline and exit
    pipeline.stop();

    std::cout << "[INFO] System stopped. Exiting." << std::endl;
    return 0;
}
