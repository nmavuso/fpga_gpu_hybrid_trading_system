# FPGA-GPU Hybrid System for Ultra-Low Latency High-Frequency Trading (HFT)

## Overview
This project implements an **ultra-low latency high-frequency trading (HFT) system** that integrates **FPGAs for real-time order execution** and **GPUs for risk modeling and market prediction**. The system is optimized to **execute trades within nanoseconds** by leveraging FPGA parallelism and GPU-accelerated inference.

## Features
- ⚡ **Nanosecond execution latency** with FPGA-optimized order book processing.
- 🚀 **CUDA-accelerated financial model inference** for market trend predictions.
- 🔄 **PCIe optimization** for low-latency data transfer between FPGA, GPU, and CPU.
- 📉 **Multi-asset arbitrage and strategy execution** with real-time computations.
- 🔗 **Integration with market feeds** (NASDAQ, CME, Binance API, etc.).

## Why It’s Hard
- **PCIe bottleneck**: GPUs excel at compute throughput but suffer from data transfer latency.
- **Shared memory contention**: FPGA, GPU, and CPU need optimized memory allocation.
- **Low-latency optimization**: Requires **CUDA streams & persistent kernels** to maintain a real-time trading pipeline.

## System Architecture

┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ 
│ Market │ ---> │ FPGA │ ---> │ GPU │ ---> │ Execution │ 
│ Feed │        │ OrderBook│ │ ML Model │  │ Engine │ 
└──────────┘ └──────────┘ └──────────┘ └────────────┘

- **FPGA:** Processes live market data and updates order books at **sub-microsecond latencies**.
- **GPU (CUDA):** Runs deep-learning-based predictions for arbitrage and trading strategies.
- **CPU:** Handles control flow, logging, and backtesting.

## Installation
### Prerequisites
- **NVIDIA CUDA 11+**  
- **Xilinx / Intel FPGA board** (UltraScale+)  
- **Verilog / VHDL toolchain (Vivado, Quartus)**  
- **CMake & GCC**  
- **Python 3.8+ (for AI-based trading models)**  

### Clone & Build
```bash
git clone https://github.com/yourusername/fpga-gpu-hft.git
cd fpga-gpu-hft
mkdir build && cd build
cmake ..
make -j$(nproc)

