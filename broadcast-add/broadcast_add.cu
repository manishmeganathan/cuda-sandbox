// A => 3D array of size [W x H x D]
// B => 2D array of size [W x H]
// C => 1D array of size [W]
//
// Compute D = A + B + C 
// D must be a 3D array of size [W x H x D] with B & C being broadcast to 3 dimensions

#include <iostream>
#include <string>

#define BLOCK 2
#define WIDTH 4     // Width
#define HEIGHT 3    // Height
#define DEPTH 2     // Depth

__global__ void broadcast_add_matrices(
    int width, int height, int depth,
    int* A, // [W x H x D]
    int* B, // [W x H]
    int *C, // [W]
    int *D  // [W x H x D]
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Check boundary condition
    if (x < width && y < height && z < depth) {
        // Calculate index with 3-dimensional linearization
        int index = (z * width * height) + (y * width) + x;
        // Broadcast addition across 3 dimensions
        D[index] = A[index] + B[(y * width) + x] + C[x];
    }
}

// Function to initialize matrices with random values
void init_matrix(int *matrix, int elements) {
    for (int i = 0; i < elements; i++) {
        matrix[i] = rand() % 10; // constrain random values to 0-9
    }
}

// Function that prints 3D matrices with formatting
// Each slice (z = 0,1,2,..) of the matrix is printed side by side.
void print_matrix_3D(std::string label, int *mat, int width, int height, int depth) {
    std::cout << label << " [" << width << " x " << height << " x " << depth << "] col-major\n";

    for (int y = 0; y < height; y++) {
        for (int z = 0; z < depth; z++) {
            std::cout << "|| \t";
            for (int x = 0; x < width; x++) {
                std::cout << mat[(z * width * height) + (y * width) + x] << "\t";
            }
        }
        std::cout << "||\n";
    }
    std::cout << "\n";
}

int main() {
    // Declare matrix pointers
    int *h_A, *h_B, *h_C, *h_D;
    int *d_A, *d_B, *d_C, *d_D;

    // Calculate array memory sizing
    size_t size_A = WIDTH * HEIGHT * DEPTH * sizeof(int);
    size_t size_B = WIDTH * HEIGHT * sizeof(int);
    size_t size_C = WIDTH * sizeof(int);

    // Allocate memory on the host
    h_A = (int*)malloc(size_A);
    h_B = (int*)malloc(size_B);
    h_C = (int*)malloc(size_C);
    h_D = (int*)malloc(size_A); // D is the same size as A (3-dimensional)

    // Allocate memory on the device
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMalloc(&d_D, size_A); // D is the same size as A (3-dimensional)

    // Initialize matrices with random values
    srand(time(NULL));
    init_matrix(h_A, WIDTH * HEIGHT * DEPTH);
    init_matrix(h_B, WIDTH * HEIGHT);
    init_matrix(h_C, WIDTH);

    // Print matrices
    print_matrix_3D("Matrix A", h_A, WIDTH, HEIGHT, DEPTH);
    print_matrix_3D("Matrix B", h_B, WIDTH, HEIGHT, 1);
    print_matrix_3D("Matrix C", h_C, WIDTH, 1, 1);

    // Copy data for A, B & C to the device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Define compute space
    dim3 block(BLOCK, BLOCK, BLOCK);
    dim3 grid(
        (WIDTH + BLOCK - 1) / BLOCK,
        (HEIGHT + BLOCK - 1) / BLOCK,
        (DEPTH + BLOCK - 1) / BLOCK
    );

    // Launch kernel for broadcast addition
    broadcast_add_matrices<<<grid, block>>>(WIDTH, HEIGHT, DEPTH, d_A, d_B, d_C, d_D);
    cudaDeviceSynchronize();

    // Copy data from device back to host
    cudaMemcpy(h_D, d_D, size_A, cudaMemcpyDeviceToHost);
    print_matrix_3D("Matrix D", h_D, WIDTH, HEIGHT, DEPTH);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return 0;
}