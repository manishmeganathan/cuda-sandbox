#include <iostream>
#include <math.h>
#include <time.h>

#define SIZE 1<<20                          // 1 million vector elements
#define BLCK 256                            // 256 thread per block to launch for parallel execution
#define GRID (SIZE + BLCK - 1) / BLCK       // computation grid = # of execution blocks to launch

// Function to add the elements of two vectors serially
// i.e, one element at a time. This code runs on the CPU
void add_serial(float *a, float *b, float *c) {
 for (int i = 0; i < SIZE; i++)
    c[i] = a[i] + b[i];
}

// Kernel function to add the elements of two vectors parallely
// i.e each pair of elements on a thread. This code runs on the GPU
__global__ void add_parallel(float *a, float *b, float *c) {
    // Calculate the compute index
    // This tells the thread what output index it is responsible for calculating
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check boundary conditions
    // This check controls against overlaunched threads
    if (i < SIZE) 
        c[i] = a[i] + b[i];
}

// Function to initialize a vector with random values
void init_vector(float *vector) {
    for (int i = 0; i < SIZE; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // Declare vector variables
    float *h_A, *h_B, *hs_C, *hp_C;
    float *d_A, *d_B, *d_C;
    // Determine size of vector for memory allocation
    size_t size = SIZE * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    hs_C = (float*)malloc(size);
    hp_C = (float*)malloc(size);

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Initialize vectors with random values
    srand(time(NULL));
    init_vector(h_A);
    init_vector(h_B);

    // Copy vectors to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Benchmark CPU Execution
    printf("Benchmarking CPU Execution ...\n");
    double cpu_total_time = 0.0;
    // Run 10 operations of the CPU vector addition
    for (int i = 0; i < 10; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);

        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }

    // Benchmark GPU Execution
    printf("Benchmarking GPU Execution ...\n");
    double gpu_total_time = 0.0;
    // Run 10 operations of the GPU vector addition
    for (int i = 0; i < 10; i++) {
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize(); // Check that all threads have finished executing

        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }

    // Verify results (sanity check)
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < SIZE; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("Results are %s\n", correct ? "correct" : "incorrect");
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Boost: %fx\n", cpu_avg_time / gpu_avg_time);

    // Calculate average execution times
    double cpu_avg_time = cpu_total_time / 10.0;
    double gpu_avg_time = gpu_total_time / 10.0;

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