enable_testing()

# CPU test
add_executable(test_cpu test_cpu.cpp)
target_link_libraries(test_cpu PRIVATE gpu_lib)
add_test(NAME test_cpu COMMAND test_cpu)

# GPU test
add_executable(test_gpu test_gpu.cu)
target_link_libraries(test_gpu PRIVATE gpu_lib)
add_test(NAME test_gpu COMMAND test_gpu)

# Integration test
add_executable(test_integration test_integration.cpp)
target_link_libraries(test_integration PRIVATE gpu_lib)
add_test(NAME test_integration COMMAND test_integration)
