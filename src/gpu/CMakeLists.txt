enable_language(CUDA)

set(GPU_SOURCES
    kernels/prediction.cu
    kernels/risk_check.cu
    kernels/advanced_strategies.cu
    gpu_interface.cpp
)

add_library(gpu_lib STATIC ${GPU_SOURCES})
target_compile_features(gpu_lib PUBLIC cxx_std_17)

# Link necessary CUDA libraries
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(gpu_lib PRIVATE CUDA::cudart CUDA::cublas ... )
