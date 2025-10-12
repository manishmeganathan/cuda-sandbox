# Matrix Addition
A CUDA program to compute **matrix multiplication** (`A[M×K] × B[K×N] = C[M×N]`) and benchmark **CPU vs GPU** performance.

## Overview
- Launch a **2D CUDA grid** of threads for matrix operations  
- Perform element-wise addition of two matrices (`A + B = C`)  
- Understand **thread indexing** for row–column mapping in GPU kernels  

## Configuration

| Parameter | Meaning | Value |
|----------|---------|-------|
| `M` | Rows of A / C | 128 |
| `K` | Cols of A & Rows of B | 256 |
| `N` | Cols of B / C | 128 |
| `TX` | Threads per block (X) | 32 |
| `TY` | Threads per block (Y) | 8 |

Grid is computed as:
- `grid.x = (N + TX - 1) / TX`
- `grid.y = (M + TY - 1) / TY`

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
CPU average time: 14.803499 ms
GPU average time: 0.033378 ms
Boost: 443.514831x
```
