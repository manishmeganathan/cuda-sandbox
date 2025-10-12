# Vector Addition
A simple CUDA program to demonstrate parallel vector addition and compare GPU vs CPU performance.

## Overview
- Adds two large vectors (1<<20 elements) both serially on the CPU and in parallel on the GPU.
- Uses CUDA kernels with configurable block and grid sizes.
- Benchmarks execution times and verifies result accuracy.

## Build & Run
```bash
make all
```

## Example Run
```bash
# Hardware
# CPU: AMD EPYC 7352 24-Core Processor
# GPU: GeForce RTX 5060 Ti

Benchmarking CPU Execution ...
Benchmarking GPU Execution ...
Results are correct
CPU average time: 5.750289 ms
GPU average time: 0.042879 ms
Boost: 134.104686x
```
