# Vector Addition
A simple CUDA program to demonstrate parallel vector addition and compare GPU vs CPU performance.

## Overview
- Adds two large vectors (1<<20 elements) both serially on the CPU and in parallel on the GPU.
- Uses CUDA kernels with configurable block and grid sizes.
- Benchmarks execution times and verifies result accuracy.

## Build & Run
```
nvcc vector_add.cu -o vector_add && ./vector_add
```

## Example Run
