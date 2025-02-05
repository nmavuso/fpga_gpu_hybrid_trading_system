// pipeline_manager.cpp
// Implementation of PipelineManager.

#include "pipeline_manager.h"
#include <iostream>
#include <chrono>

// Placeholder for FPGA read/write
static float readFromFPGA() {
    // Simulate a new market data value
    static float val = 100.0f;
    val += 0.1f; // increment to simulate changing market data
    return val;
}

PipelineManager::PipelineManager(int batchSize)
    : hostBuffer_(batchSize, 0.0f),
      resultBuffer_(batchSize, 0.0f),
      gpu_(batchSize),
      running_(false)
{}

PipelineManager::~PipelineManager() {
    stop();
}

void PipelineManager::start() {
    running_ = true;
    fpgaThread_ = std::thread(&PipelineManager::fpgaDataThreadFunc, this);
    gpuThread_  = std::thread(&PipelineManager::gpuComputeThreadFunc, this);
}

void PipelineManager::stop() {
    running_ = false;
    if (fpgaThread_.joinable()) {
        fpgaThread_.join();
    }
    if (gpuThread_.joinable()) {
        gpuThread_.join();
    }
}

void PipelineManager::fpgaDataThreadFunc() {
    // Continuously read data from FPGA, fill hostBuffer
    while (running_) {
        {
            std::lock_guard<std::mutex> lock(dataMutex_);
            for (auto &val : hostBuffer_) {
                val = readFromFPGA();
            }
        }
        // Sleep briefly to simulate real-time intervals
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void PipelineManager::gpuComputeThreadFunc() {
    while (running_) {
        {
            std::lock_guard<std::mutex> lock(dataMutex_);
            // Call GPUInterface to process the current hostBuffer
            gpu_.runAllKernels(hostBuffer_.data(), resultBuffer_.data());
            // resultBuffer_ now holds updated data from advanced strategy
        }
        // Could do more things, like send results to execution engine
        // For demonstration, just log the first result
        std::cout << "[GPU Output] " << resultBuffer_[0] << std::endl;

        // Sleep or yield to simulate other tasks
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
