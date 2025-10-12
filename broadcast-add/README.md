# Matrix Broadcast Addition
A CUDA program that demonstrates **broadcast addition** across 3D tensors using GPU parallelism.

## Overview
- `A` is a 3D array `[W x H x D]`
- `B` is a 2D array `[W x H]`
- `C` is a 1D array `[W]`

Both `B` and `C` are **broadcast** across the depth and height dimensions respectively, to match `A`.  
Each thread computes one element of the output tensor `D`.

## Configuration
| Parameter | Description | Value |
|------------|--------------|--------|
| `BLOCK` | Threads per block (in each dimension) | 2 |
| `WIDTH` | Array width (x-dimension) | 4 |
| `HEIGHT` | Array height (y-dimension) | 3 |
| `DEPTH` | Array depth (z-dimension) | 2 |

## Build & Run
```bash
make all
```

## Example Run
```bash
Matrix A [4 x 3 x 2] col-major
||      3       7       1       5       ||      5       2       5       9       ||
||      7       4       1       0       ||      4       6       4       1       ||
||      1       1       8       2       ||      7       2       1       2       ||

Matrix B [4 x 3 x 1] col-major
||      8       7       7       0       ||
||      4       1       4       0       ||
||      0       7       7       7       ||

Matrix C [4 x 1 x 1] col-major
||      1       8       7       2       ||

Matrix D [4 x 3 x 2] col-major
||      12      22      15      7       ||      14      17      19      11      ||
||      12      13      12      2       ||      9       15      15      3       ||
||      2       16      22      11      ||      8       17      15      11      ||
```