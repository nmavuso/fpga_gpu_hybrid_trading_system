# FPGA-GPU Hybrid System for Ultra-Low Latency High-Frequency Trading (HFT)

## Overview
This **advanced** HFT system combines **FPGA** (for sub-microsecond order book updates & DMA streaming) and **GPU** (for high-throughput inference and risk checks) in a **multi-threaded CPU** pipeline.  

Features include:
- **FPGA** streaming interface for market data ingestion (DMA).
- **Multiple CUDA streams** and **page-locked (pinned) memory** for ultra-low latency GPU inference.
- **Threaded CPU pipeline** for concurrency and minimal blocking.
- **Advanced risk-check kernels** on GPU for real-time margin checks and portfolio constraints.
- **Scalable** to multi-asset, multi-exchange environments.

## Why This Is Hard
1. **FPGA ↔ CPU ↔ GPU data orchestration**: Minimizing PCIe overhead requires pinned memory, batched DMA transfers, and deep pipelining.  
2. **Shared memory contention**: Careful concurrency controls among CPU threads to avoid race conditions.  
3. **Low-latency streams**: Requires advanced CUDA usage (streams, events) plus FPGA streaming logic to feed data in microseconds.  
4. **High-frequency concurrency**: CPU must handle real-time data, FPGA partial updates, GPU calls, and risk enforcement simultaneously.

## Architecture
    ┌─────────────────┐    ┌────────────────────────┐
    │ Market Data Feed│    │ Model Weights, Risk Data│
    └─────────────────┘    └────────────────────────┘
              │                         │
              ▼                         ▼
┌────────────────────────┐      ┌─────────────────────────┐
│   FPGA (Order Book)    │ ---> │  CPU (Pipeline Manager) │
│ + DMA (Streaming Intf) │      │   Multi-Threaded Logic  │
└────────────────────────┘      └─────────────────────────┘
              │                          │
              ▼                          ▼
      ┌─────────────┐            ┌─────────────────┐
      │ GPU (CUDA)   │ <-------->│ Risk & Strategy  │
      └─────────────┘            └─────────────────┘



- **FPGA**: Receives real-time market data, updates order books, and sends incremental updates via DMA to the CPU memory.  
- **CPU**: Manages multiple threads (one for FPGA data, another for GPU requests, etc.). Orchestrates pinned buffers and concurrency.  
- **GPU**: Runs advanced inference kernels (trend prediction) and risk checks in parallel streams for high throughput.

## Installation

### Prerequisites
- **NVIDIA CUDA 11+**  
- **Xilinx or Intel FPGA Board** (UltraScale+, Stratix 10, etc.)  
- **Vivado / Quartus** for synthesis  
- **CMake 3.10+** & **GCC**  
- **Python 3.8+** (to train or test AI models)  

### Building
```bash
git clone https://github.com/yourusername/fpga-gpu-hft-advanced.git
cd fpga-gpu-hft-advanced
mkdir build && cd build
cmake ..
make -j$(nproc)
