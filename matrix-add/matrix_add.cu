#include <iostream>
#include <string>

#define BLOCK 4
#define WIDTH 10   
#define HEIGHT 8

// Kernel function to perform 2D matrix addition
__global__ void matadd(int *A, int *B, int *C, int width, int height) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // Check boundary condition
    if (col < width && row < height) {
        // Calculate index, used thrice in the next line
        int idx = row * width + col;
        // Calculate Aij + Bij and set to Cij
        C[idx] = A[idx] + B[idx];
    }
}

// Function to initialize matrices with random values
void init_matrix(int *matrix, int elements) {
    for (int i = 0; i < elements; i++) {
        matrix[i] = rand() % 10; // constrain random values to 0-9
    }
}

// Function that prints matrices with formatting
void print_matrix(std::string label, int *matrix, int width, int height) {
    std::cout << label << " [" << height << " x " << width << "] row-major\n";
    
    for (int row = 0; row < height; row++) {
        std::cout << "|\t";

        for (int col = 0; col < width; col++) {
            std::cout << matrix[row * width + col] << "\t";
        }

        std::cout << "|\n";
    }

    std::cout << "\n";
}

int main() {
    // Declare matrix pointers
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    // Determine size of matrix for memory allocation
    size_t size = WIDTH * HEIGHT * sizeof(int);

    // Allocate host memory
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Initialize matrices with random values
    srand(time(NULL));
    init_matrix(h_A, WIDTH * HEIGHT);
    init_matrix(h_B, WIDTH * HEIGHT);

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Print matrices
    print_matrix("Matrix A", h_A, WIDTH, HEIGHT);
    print_matrix("Matrix B", h_B, WIDTH, HEIGHT);

    // Define computation space
    dim3 block(BLOCK, BLOCK);
    dim3 grid(
        (WIDTH + BLOCK - 1) / BLOCK,
        (HEIGHT + BLOCK - 1) / BLOCK
    );

    // Launch matrix addition kernel
    matadd<<<grid, block>>>(d_A, d_B, d_C, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    // Copy data from device back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    print_matrix("Matrix C", h_C, WIDTH, HEIGHT);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}