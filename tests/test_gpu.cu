// test_gpu.cu
// Test GPU kernels in isolation.

#include <iostream>
#include <vector>
#include <cmath>

// Declarations of the GPU kernels
extern "C" void runPrediction(const float* d_input, float* d_output, float weight, int n, cudaStream_t stream);
extern "C" void runRiskCheck(const float* d_positions, float* d_riskFlags, float threshold, int n, cudaStream_t stream);
extern "C" void runAdvancedStrategy(const float* d_inData, float* d_outSignals, int dataSize, cudaStream_t stream);

int main() {
    std::cout << "[TEST] GPU Test Starting..." << std::endl;

    // Basic test data
    const int N = 10;
    std::vector<float> input(N, 100.0f);
    std::vector<float> output(N, 0.0f);
    std::vector<float> flags(N, 0.0f);

    float *dInput = nullptr, *dOutput = nullptr, *dFlags = nullptr;
    cudaMalloc(&dInput,  N * sizeof(float));
    cudaMalloc(&dOutput, N * sizeof(float));
    cudaMalloc(&dFlags,  N * sizeof(float));

    cudaMemcpy(dInput, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t s1, s2, s3;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaStreamCreate(&s3);

    runPrediction(dInput, dOutput, 1.05f, N, s1);
    runRiskCheck(dInput, dFlags, 120.0f, N, s2);
    runAdvancedStrategy(dInput, dOutput, N, s3);

    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    cudaStreamSynchronize(s3);

    cudaMemcpy(output.data(), dOutput, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(flags.data(),  dFlags,  N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dInput);
    cudaFree(dOutput);
    cudaFree(dFlags);

    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaStreamDestroy(s3);

    // Validate
    bool pass = true;
    // Just a simple check to see if we got some output
    // (In real tests, you'd compare with expected results.)
    if (std::fabs(output[0]) < 0.0001f) {
        pass = false;
    }

    if (pass) std::cout << "[TEST] GPU Test Passed!" << std::endl;
    else      std::cout << "[TEST] GPU Test Failed!" << std::endl;

    return pass ? 0 : 1;
}
