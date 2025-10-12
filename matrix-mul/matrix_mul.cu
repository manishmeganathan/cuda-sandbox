#include <iostream>
#include <iomanip>
#include <math.h>
#include <time.h>

#define M 128  
#define K 256     
#define N 128    

// 256 threads/block
#define TX 32 
#define TY 8

// Function that performs matrix multiplication A x B = C.
// The sizes of the matrices are (M x K) * (K x N) = (M x N)
// This code runs iteratively on each element of C on the CPU.
void matmul_serial(float *A, float *B, float *C) {
    // Iterate over the rows of A
    for (int i = 0; i < M; i++) {
        // Iterate over the columns of B
        for (int j = 0; j < N; j++) {
            // Declare a dot product accumulator
            float dot = 0.0f;
            // Calculate dot product of Ai * Bj
            for (int offset = 0; offset < K; offset++) {
                dot += A[offset + i * K] * B[offset * N + j];
            }

            // Set the output of Cij to the dot product
            C[i * N + j] = dot;
        }        
    }
} 

// Kernel function for matrix multiplication A x B = C.
// The sizes of the matrices are (M x K) * (K x N) = (M x N)
// This code runs parallely with each thread computing 1 element Cij in matrix C on 
// the GPU by computing the dot product between the Ai row and Bj column matrices
__global__ void matmul_parallel(float *A, float *B, float *C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check boundary condition
    if (row < M && col < N) {
        // Declare a dot product accumulator
        float dot = 0.0f;
        // Calculate dot product of Ai * Bj
        for (int offset = 0; offset < K; offset++) {
            dot += A[offset + row * K] * B[offset * N + col];
        }

        // Set the output of Cij to the dot product
        C[row * N + col] = dot;
    }
}

// Function to initialize matrices with random values
void init_matrix(float *matrix, int elements) {
    for (int i = 0; i < elements; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// Function that prints matrices with formatting
void print_matrix(std::string label, float *matrix, int width, int height) {
    std::cout << std::fixed << std::setprecision(4) << std::showpoint;
    std::cout << label << " [" << height << " x " << width << "] row-major\n";
    
    for (int row = 0; row < height; row++) {
        std::cout << "||\t";
        for (int col = 0; col < width; col++) {
            std::cout << matrix[row * width + col] << "\t";
        }
        std::cout << "||\n";
    }
    std::cout << "\n";
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // Declare matrix variables
    float *h_A, *h_B, *hs_C, *hp_C;
    float *d_A, *d_B, *d_C;

    // Calculate array memory sizing
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate memory on the host
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    hs_C = (float*)malloc(size_C);
    hp_C = (float*)malloc(size_C);

    // Allocate memory on the device
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Initialize matrices with random values
    srand(time(NULL));
    init_matrix(h_A, M * K);
    init_matrix(h_B, K * N);

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Print matrices
    // print_matrix("Matrix A", h_A, M, K);
    // print_matrix("Matrix B", h_B, K, N);

    // Benchmark CPU Execution
    printf("Benchmarking CPU Execution ...\n");
    double cpu_total_time = 0.0;
    // Run 10 operations of the CPU matrix multiplication
    for (int i = 0; i < 10; i++) {
        double start_time = get_time();
        matmul_serial(h_A, h_B, hs_C);

        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }

    // Define computation space
    dim3 block(TX, TY);
    dim3 grid(
        (N + TX - 1) / TX, // x -> columns
        (M + TY - 1) / TY  // y -> rows
    );

    // Benchmark GPU Execution
    printf("Benchmarking GPU Execution ...\n");
    double gpu_total_time = 0.0;
    // Run 10 operations of the GPU matrix multiplication
    for (int i = 0; i < 10; i++) {
        double start_time = get_time();
        matmul_parallel<<<grid, block>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize(); // Check that all threads have finished executing

        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }

    // Verify results (sanity check)
    cudaMemcpy(hp_C, d_C, size_C, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(hs_C[i] - hp_C[i]) > 1e-3) {
            correct = false;
            break;
        }
    }

    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Print matrices
    // print_matrix("Matrix C CPU", hs_C, M, N);
    // print_matrix("Matrix C GPU", hp_C, M, N);

    // Calculate average execution times
    double cpu_avg_time = cpu_total_time / 10.0;
    double gpu_avg_time = gpu_total_time / 10.0;

    printf("CPU average time: %f ms\n", cpu_avg_time*1000);
    printf("GPU average time: %f ms\n", gpu_avg_time*1000);
    printf("Boost: %fx\n", cpu_avg_time / gpu_avg_time);

    // Free memory
    free(h_A);
    free(h_B);
    free(hs_C);
    free(hp_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}