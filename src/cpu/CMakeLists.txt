set(CPU_SOURCES
    main.cpp
    pipeline_manager.cpp
)

add_executable(hft_system ${CPU_SOURCES})
target_link_libraries(hft_system PRIVATE gpu_lib)

# If you integrate FPGA software drivers, link them here:
# target_link_libraries(hft_system PRIVATE fpga_driver_lib)
