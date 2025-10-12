#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

// #include <cuda_runtime.h>
#include <curand_kernel.h>

#define MNIST_IMAGE_SIZE 784
#define MNIST_TRAIN_SIZE 60000
#define MNIST_TEST_SIZE 10000

#define ASSERT(cond, msg, args...) assert((cond) || !fprintf(stderr, (msg "\n"), args))

// Kernel function to perform the forward pass on the neural network between two layers with sizes N and M.
// Takes the input activations for a batch of inputs with size B, a weight matrix between the two layers,
// the biases for each node on the input layer and computes the output layer activations.
__global__ void forward_pass(
    int B, int N, int M, 
    float* weights, // [N * M]
    float* biases,  // [N * 1]
    float* input_activations,  // [B * N]
    float* output_activations  // [B * M]
) 
{ 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check for boundary conditions
    // Output matrix is [B * M]
    if (row < B && col < M) {
        // Calculate the compute index
        int idx = row * M + col;

        // Bias values must be broadcast to the output vector
        output_activations[idx] = biases[col];
        // Perform matrix multiplication between activation matrix and weight matrix
        for (int offset = 0; offset < N; offset++) {
            output_activations[idx] += input_activations[row * N + offset] * weights[offset * M + col];
        }
    }
}

__global__ void backward_pass(
    int B, int N, int M, 
    float* weights, // [N * M]
    float* biases,  // [N * 1]
    float* inputs_lossdiff,  // [B * N]
    float* outputs_lossdiff  // [B * M]
)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check for boundary conditions
    // Output matrix is [B * M]
    if (row < B && col < M) {
         // Calculate the compute index
        int idx = row * M + col;

        // Initialize the value with 0 before aggregating the dot product
        outputs_lossdiff[idx] = 0.f;
        // Perform matrix multiplication between loss differential matrix and weight matrix
        for (int offset = 0; offset < N; offset++) {
            outputs_lossdiff[idx] += inputs_lossdiff[row * N + offset] * weights[offset * M + col];
        }
    }
}

__global__ void update_layer(
    int w, int h, int b, 
    float learn_rate,
    float* weights,
    float* biases,
    float* activations,
    float* lossdiff
)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check for boundary condition
    if (row < h && col < w) {
        float dw = 0.f, db = 0.f;

        // For each label in the batch, determine the error in the weights and biases
        for (int i = 0; i < b; i++) {
            float act = activations[i * h + row];
            float loss = lossdiff[i * w + col];

            dw += act * loss;
            db += loss;
        }

        // Adjust the weights and biases with stochastic gradient descent
        weights[row * w + col] -= learn_rate * dw / b;
        biases[col] -= learn_rate * db / b;
    }
}

// Kernel function to compute the ReLU activation for a vector [H * W] 
__global__ void relu(int w, int h, float* input, float* output) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check for boundary condition
    if (row < h && col < w) {
        // Calculate the compute index
        int idx = row * w + col;

        // Obtain input value and apply ReLU activation 
        float activation = input[idx];
        output[idx] = activation > 0.f ? activation : 0.f; // ReLU function
    }
}

__global__ void relu_backward(int w, int h, float* input, float* lossdiff, float* output)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check for boundary condition
    if (row < h && col < w) {
        // Calculate the compute index
        int idx = row * w + col;

        // Obtain the input value and apply ReLU backwards
        // Returns the loss differential if activated (>0)
        float activation = input[idx];
        output[idx] = activation > 0.f ? lossdiff[idx] : 0.f;
    }
}


// Kernel function to compute the softmax activation for a vector [H * W]
// Note: This is a very unoptimized implementation.
__global__ void softmax(int w, int h, float* input, float* output)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

     // Check for boundary condition
    if (row < h && col < w) {
        // Find the maximum value in the vector
        float maxval = input[row * w];
        for (int i = 0; i < w; i++) {
            maxval = max(maxval, input[row * w + i]);
        }

        // Aggregate the sum of exponential values
        float sum = 0.f;
        for (int i = 0; i < w; i++) {
            // Deduct the max value from each value to
            // force the exponent value between 0 and 1
            sum += exp(input[row * w + i] - maxval); 
        }

        // Determine output index, used twice in next line
        int idx = row * w + col;
        // Divide the value for index with the aggregated sum after
        // deducting the max value for the same reason as above
        output[idx] = exp(input[idx] - maxval) / sum;
    }
}

// Kernel function to compute the cross entropy loss between an expected probability and actual probability
__global__ void cross_entropy(int w, int h, float* predicted, float* actual, float* output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for boundary condition
    if (x < h) {
        float loss = 0.f;
        // Aggregate the entropy loss for each label
        for (int i = 0; i < w; i++) {
            // Determine index for probability vectors
            int idx = i + x * w;
            // We correct the log(q(x)) to prevent log(0) as it is undefined
            loss -= actual[idx] * log(max(1e-6, predicted[idx]));
        }

        // Save the aggregated loss to the output vector
        output[x] = loss;
    }
}

__global__ void cross_entropy_backward(int w, int h, float* predicted, float* actual, float* output) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check for boundary condition
    if (row < h && col < w) {
        // Calculate the compute index
        int idx = row * w + col;
        // Calculate the difference between the predicted and actual entropy
        output[idx] = predicted[idx] - actual[idx];
    }
}

// Kernel function to randomly initialize the values for a given matrix with dimensions [H * W]
__global__ void init_rand(int w, int h, float *matrix) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check for boundary condition
    if (row < h && col < w) {
        // Calculate the compute index
        int idx = row * w + col;

        // Generate the random value for the matrix
        // This code is copied from an external source and I have no idea how it works
        curandState state;
        curand_init(42, idx, 0, &state);
        matrix[idx] = curand_normal(&state) * sqrtf(2.f/h);
    }
}

// Function to read the MNIST dataset
void read_mnist(const std::string filename, int length, float* mnist, float* labels) 
{
    std::fstream fin;
    fin.open(filename);

    std::string row;
    constexpr char delim = ',';

    for(int i = 0; i < length; i++) {
        fin >> row;

        // Capture the label for the data
        int delim = row.find(delim);
        int label = std::stoi(row.substr(0, delim + 1));

        // Encode the label into the vector such that 
        // the one-hot value is set to 1 and others to 0
        for(int j = 0; j < 10; j++) {
            labels[10 * i + j] = (j == label);
        }

        // Remove the label from the row string
        row.erase(0, delim+1);

        // Parse the image pixel values into the array
        for(int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            // Capture the next pixel value
            delim = row.find(delim);
            if (delim == std::string::npos) {
                delim = row.length() - 1;
            }
            
            // Normalize values between 0-255 (grayscale) to float values between 0 and 1
            mnist[i * MNIST_IMAGE_SIZE + j] = std::stof(row.substr(0, delim + 1)) / 255;
            
            // Erase the row and loop over
            row.erase(0, delim + 1);
        }

        ASSERT(row.length() == 0, "didn't parse all values in row, %d", i);
    }
}

// matrix printing
// layer intialisation

int main() {
    return 0;
}