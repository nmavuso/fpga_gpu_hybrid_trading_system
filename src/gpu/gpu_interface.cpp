// gpu_interface.cpp
// Implementation of GPUInterface, showing pinned memory, multiple streams, etc.

#include "gpu_interface.h"
#include <iostream>
#include <cuda_runtime.h>

// Declarations of our extern "C" kernels
extern "C" void runPrediction(const float* d_input, float* d_output, float weight, int n, cudaStream_t stream);
extern "C" void runRiskCheck(const float* d_positions, float* d_riskFlags, float threshold, int n, cudaStream_t stream);
extern "C" void runAdvancedStrategy(const float* d_inData, float* d_outSignals, int dataSize, cudaStream_t stream);

GPUInterface::GPUInterface(int batchSize)
    : batchSize_(batchSize), d_input_(nullptr), d_output_(nullptr),
      d_positions_(nullptr), d_riskFlags_(nullptr)
{
    // Allocate device memory
    cudaMalloc((void**)&d_input_,     batchSize_ * sizeof(float));
    cudaMalloc((void**)&d_output_,    batchSize_ * sizeof(float));
    cudaMalloc((void**)&d_positions_, batchSize_ * sizeof(float));
    cudaMalloc((void**)&d_riskFlags_, batchSize_ * sizeof(float));

    // Create multiple streams
    cudaStreamCreate(&stream1_);
    cudaStreamCreate(&stream2_);
    cudaStreamCreate(&stream3_);
}

GPUInterface::~GPUInterface() {
    // Cleanup
    cudaFree(d_input_);
    cudaFree(d_output_);
    cudaFree(d_positions_);
    cudaFree(d_riskFlags_);

    cudaStreamDestroy(stream1_);
    cudaStreamDestroy(stream2_);
    cudaStreamDestroy(stream3_);
}

void GPUInterface::runAllKernels(const float* h_input, float* h_output) {
    // Copy input to GPU asynchronously on stream1
    cudaMemcpyAsync(d_input_, h_input, batchSize_ * sizeof(float), cudaMemcpyHostToDevice, stream1_);
    // For demonstration, let's treat positions_ as same as input
    cudaMemcpyAsync(d_positions_, h_input, batchSize_ * sizeof(float), cudaMemcpyHostToDevice, stream1_);

    // Launch multiple kernels in parallel streams
    launchPredictionKernel(1.05f);
    launchRiskCheckKernel(120.0f);
    launchAdvancedStrategyKernel();

    // Wait for them all to finish (you could be more granular)
    cudaStreamSynchronize(stream1_);
    cudaStreamSynchronize(stream2_);
    cudaStreamSynchronize(stream3_);

    // Retrieve final output from advanced strategy (e.g., stored in d_output_)
    cudaMemcpy(h_output, d_output_, batchSize_ * sizeof(float), cudaMemcpyDeviceToHost);
}

void GPUInterface::launchPredictionKernel(float weight) {
    runPrediction(d_input_, d_output_, weight, batchSize_, stream1_);
}

void GPUInterface::launchRiskCheckKernel(float threshold) {
    runRiskCheck(d_positions_, d_riskFlags_, threshold, batchSize_, stream2_);
}

void GPUInterface::launchAdvancedStrategyKernel() {
    runAdvancedStrategy(d_input_, d_output_, batchSize_, stream3_);
}
