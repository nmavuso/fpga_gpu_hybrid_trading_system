// gpu_interface.h
// High-level C++ interface for launching multiple GPU kernels in parallel streams.

#ifndef GPU_INTERFACE_H
#define GPU_INTERFACE_H

#include <cuda_runtime.h>

class GPUInterface {
public:
    GPUInterface(int batchSize);
    ~GPUInterface();

    void runAllKernels(const float* h_input, float* h_output);

private:
    int batchSize_;
    float *d_input_, *d_output_, *d_positions_, *d_riskFlags_;
    cudaStream_t stream1_, stream2_, stream3_;

    // Kernel wrappers
    void launchPredictionKernel(float weight);
    void launchRiskCheckKernel(float threshold);
    void launchAdvancedStrategyKernel();
};

#endif // GPU_INTERFACE_H
