#include <iostream>
#include <random>
#include <vector>

#include <torch/torch.h>

// Fills a vector with deterministic Uniform(-1.0, 1.0) floats (mt19937 + seed).
// Note: same seed ⇒ same values; use different seeds for A and B to avoid structure.
void fill_vector_uniform(std::vector<float>& v, unsigned seed=1234) {
    // Create random distribution between -1.0 and 1.0 with given seed
    std::mt19937 gen(seed); 
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create random value for the vector from the distribution
    for (float& x : v) x = dist(gen); 
}

int main() {
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "yes" : "no") << "\n";

    std::vector<float> vec_data((size_t)10 * 10);
    fill_vector_uniform(vec_data, 1234); 
    
    // Create a tensor from the vector's data pointer
    torch::Tensor tensor_from_vec = torch::from_blob(
        vec_data.data(),                                // Pointer to the data
        {10, 10},                                       // Shape of the tensor
        torch::TensorOptions().dtype(torch::kFloat32)   // Data type
    );  
    std::cout << "Input Tensor:\n" << tensor_from_vec << "\n";

    tensor_from_vec = tensor_from_vec.to(torch::kCUDA);
    torch::Tensor softmax_out = torch::softmax(tensor_from_vec, /*dim=*/1); // row-wise softmax
    std::cout << "Softmax (row-wise):\n" << softmax_out.cpu() << "\n";

    return 0;
}