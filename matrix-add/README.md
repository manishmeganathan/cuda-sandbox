# Matrix Addition
A simple CUDA program to perform **2D matrix addition** in parallel using GPU threads.

## Overview
- Launch a **2D CUDA grid** of threads for matrix operations  
- Perform element-wise addition of two matrices (`A + B = C`)  
- Understand **thread indexing** for row–column mapping in GPU kernels  

## Configuration

| Parameter | Description | Value |
|------------|--------------|--------|
| `BLOCK` | Threads per block (in each dimension) | 4 |
| `WIDTH` | Matrix width (columns) | 10 |
| `HEIGHT` | Matrix height (rows) | 8 |

## Build & Run
```bash
make all
```

## Example Run
```bash
Matrix A [8 x 10] row-major
|       0       0       3       1       7       1       8       7       6       7       |
|       0       6       3       8       8       1       7       1       8       9       |
|       5       1       9       7       7       1       2       9       1       6       |
|       7       1       8       2       3       5       6       1       5       4       |
|       0       7       0       3       5       0       5       4       4       5       |
|       3       9       9       4       8       8       7       0       9       8       |
|       8       8       0       7       1       5       4       9       8       1       |
|       3       8       8       6       3       4       6       0       0       2       |

Matrix B [8 x 10] row-major
|       6       6       2       7       2       0       7       0       3       6       |
|       8       1       4       0       0       7       7       5       6       5       |
|       6       2       3       5       8       9       1       6       9       3       |
|       9       7       9       3       4       2       5       3       4       0       |
|       9       2       2       6       5       2       3       2       9       2       |
|       8       6       6       3       3       4       2       6       2       4       |
|       9       3       1       1       8       8       5       4       1       9       |
|       4       3       3       8       9       8       3       4       3       2       |

Matrix C [8 x 10] row-major
|       6       6       5       8       9       1       15      7       9       13      |
|       8       7       7       8       8       8       14      6       14      14      |
|       11      3       12      12      15      10      3       15      10      9       |
|       16      8       17      5       7       7       11      4       9       4       |
|       9       9       2       9       10      2       8       6       13      7       |
|       11      15      15      7       11      12      9       6       11      12      |
|       17      11      1       8       9       13      9       13      9       10      |
|       7       11      11      14      12      12      9       4       3       4       |
```
