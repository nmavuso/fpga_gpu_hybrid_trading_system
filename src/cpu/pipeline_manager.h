// pipeline_manager.h
// A class that orchestrates the entire pipeline: reading from FPGA, managing threads, and dispatching GPU tasks.

#ifndef PIPELINE_MANAGER_H
#define PIPELINE_MANAGER_H

#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include "gpu_interface.h"

class PipelineManager {
public:
    PipelineManager(int batchSize);
    ~PipelineManager();

    // Launches threads
    void start();
    void stop();

private:
    void fpgaDataThreadFunc();
    void gpuComputeThreadFunc();

    // Data buffers
    std::vector<float> hostBuffer_;
    std::vector<float> resultBuffer_;

    GPUInterface gpu_;
    std::thread fpgaThread_;
    std::thread gpuThread_;
    std::atomic<bool> running_;
    std::mutex dataMutex_;
};

#endif // PIPELINE_MANAGER_H
