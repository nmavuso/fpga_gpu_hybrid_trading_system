// risk_check.cu
// GPU kernel for real-time risk checks (margin, leverage, etc.).

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void riskCheckKernel(const float* positions, float* riskFlags, float threshold, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple risk check: if position > threshold, set a flag
        riskFlags[idx] = (positions[idx] > threshold) ? 1.0f : 0.0f;
    }
}

extern "C" void runRiskCheck(const float* d_positions, float* d_riskFlags, float threshold, int n, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    riskCheckKernel<<<gridSize, blockSize, 0, stream>>>(d_positions, d_riskFlags, threshold, n);
}
