#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdio> 
#include <QDebug>
#include "dqn.cuh"
#include <chrono>
#include <thread>
#include <random>


// Parameterized Constructor
NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes_)
    : layerSizes(layerSizes_)
{
    if(layerSizes.size() < 2) {
        throw std::invalid_argument("NeuralNetwork must have at least two layers (input and output).");
    }

    numLayers = layerSizes.size() - 1;

    // Pre-calculate total sizes
    size_t totalWeights = 0;
    size_t totalBiases = 0;
    for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
        totalWeights += static_cast<size_t>(layerSizes[i]) * layerSizes[i+1];
        totalBiases += layerSizes[i+1];
    }

    // Reserve space for host vectors
    host_weights.reserve(totalWeights);
    host_biases.reserve(totalBiases);

    initializeHostWeightsAndBiases();

    d_weights = nullptr;
    d_biases = nullptr;

    // Allocate and copy weights and biases to device with retry mechanism
    const int maxRetries = 3;
    for (int attempt = 0; attempt < maxRetries; ++attempt) {
        try {
            allocateDeviceMemory();
            copyWeightsToDevice();
            copyBiasesToDevice();
            return; // Success, exit the constructor
        } catch (const std::runtime_error& e) {
            qDebug() << "Attempt" << attempt + 1 << "failed:" << e.what();
            if (attempt == maxRetries - 1) {
                throw; // Rethrow the exception if all attempts failed
            }
            // Wait for a short time before retrying
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}


NeuralNetwork::NeuralNetwork(const NeuralNetwork& other)
    : layerSizes(other.layerSizes),
      host_weights(other.host_weights),
      host_biases(other.host_biases) {
    allocateDeviceMemory();
    copyToDevice();
}

NeuralNetwork::NeuralNetwork(NeuralNetwork&& other) noexcept
    : layerSizes(std::move(other.layerSizes)),
      host_weights(std::move(other.host_weights)),
      host_biases(std::move(other.host_biases)),
      d_weights(std::move(other.d_weights)),
      d_biases(std::move(other.d_biases)) {}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& other) {
    if (this != &other) {
        layerSizes = other.layerSizes;
        host_weights = other.host_weights;
        host_biases = other.host_biases;
        allocateDeviceMemory();
        copyToDevice();
    }
    return *this;
}


// Destructor
NeuralNetwork::~NeuralNetwork() {
    if (d_weights || d_biases) {
        freeDeviceMemory();
    }
    // The vectors will be automatically destroyed
}


void NeuralNetwork::initializeHostWeightsAndBiases() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.05, 0.05);

    size_t totalWeights = 0;
    size_t totalBiases = 0;
    for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
        totalWeights += layerSizes[i] * layerSizes[i+1];
        totalBiases += layerSizes[i+1];
    }

    host_weights.reserve(totalWeights);
    host_biases.reserve(totalBiases);
	

	for (size_t l = 0; l < numLayers; ++l) {
	    int inputSize = layerSizes[l];
	    int outputSize = layerSizes[l+1];
	    for (int o = 0; o < outputSize; ++o) {
	        for (int i = 0; i < inputSize; ++i) {
	            host_weights.push_back(dis(gen));
	        }
	    }
	    for (int o = 0; o < outputSize; ++o) {
	        host_biases.push_back(0.0);
	    }
	}

    weightOffsets.resize(numLayers);
    biasOffsets.resize(numLayers);

    totalWeights = 0;
    totalBiases = 0;

    for (size_t l = 0; l < numLayers; ++l) {
        size_t layerWeightsSize = layerSizes[l] * layerSizes[l+1];
        size_t layerBiasesSize = layerSizes[l+1];

        weightOffsets[l] = totalWeights;
        biasOffsets[l] = totalBiases;

        totalWeights += layerWeightsSize;
        totalBiases += layerBiasesSize;
    }

    // Resize and initialize host_weights and host_biases
    host_weights.resize(totalWeights);
    host_biases.resize(totalBiases);

}



// Copy weights from host to device
void NeuralNetwork::copyWeightsToDevice() {
    if (!d_weights) {
        throw std::runtime_error("Device memory for weights is not allocated.");
    }

    size_t size = host_weights.size() * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_weights.get(), host_weights.data(), size, cudaMemcpyHostToDevice));
}

// Copy biases from host to device
void NeuralNetwork::copyBiasesToDevice() {
    if (!d_biases) {
        throw std::runtime_error("Device memory for biases is not allocated.");
    }

    size_t size = host_biases.size() * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_biases.get(), host_biases.data(), size, cudaMemcpyHostToDevice));
}
// Free device memory
void NeuralNetwork::freeDeviceMemory() {
    if(d_weights) {
        d_weights.reset();
        d_weights = nullptr;
    }
    if(d_biases) {
        d_biases.reset();
        d_biases = nullptr;
    }
}



// Example CUDA kernel for forward pass (simplified)
__global__ void forwardKernel(double* weights, double* biases, double* input, double* output, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < outputSize) {
        double sum = 0.0;
        for(int i = 0; i < inputSize; ++i) {
            size_t weight_idx = idx * inputSize + i;  // Corrected indexing
            sum += input[i] * weights[weight_idx];
        }
        sum += biases[idx];
        output[idx] = tanh(sum); // Activation function
    }
}


// Forward pass implementation using CUDA
std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    if(input.size() != static_cast<size_t>(layerSizes[0])) {
        throw std::invalid_argument("Input size does not match network input layer size.");
    }

    // Allocate device memory for activations
    auto cudaDeleter = [](double* ptr) { cudaFree(ptr); };
    std::unique_ptr<double, decltype(cudaDeleter)> d_input(nullptr, cudaDeleter);
    std::unique_ptr<double, decltype(cudaDeleter)> d_output(nullptr, cudaDeleter);

    // Copy input to device
    double* raw_d_input;
    CUDA_CHECK(cudaMalloc(&raw_d_input, input.size() * sizeof(double)));
    d_input.reset(raw_d_input);
    CUDA_CHECK(cudaMemcpy(d_input.get(), input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice));

    double* d_current_input = d_input.get();
    int current_input_size = layerSizes[0];

    for (int layer = 0; layer < numLayers; ++layer) {
        int current_output_size = layerSizes[layer + 1];

        // Allocate memory for the output of this layer
        double* raw_d_output;
        CUDA_CHECK(cudaMalloc(&raw_d_output, current_output_size * sizeof(double)));
        d_output.reset(raw_d_output);

        // Get pointers to weights and biases for this layer
        double* d_layer_weights = d_weights.get() + weightOffsets[layer];
        double* d_layer_biases = d_biases.get() + biasOffsets[layer];

        // Launch the kernel for this layer
        int threadsPerBlock = 256;
        int blocksPerGrid = (current_output_size + threadsPerBlock - 1) / threadsPerBlock;
        forwardKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_layer_weights,
            d_layer_biases,
            d_current_input,
            d_output.get(),
            current_input_size,
            current_output_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Free the input of the previous layer if it's not the original input
        if (d_current_input != d_input.get()) {
            CUDA_CHECK(cudaFree(d_current_input));
        }

        // Prepare for next layer
        d_current_input = d_output.release(); // Transfer ownership
        current_input_size = current_output_size;
    }

    // Copy output back to host
    std::vector<double> host_output(current_input_size);
    CUDA_CHECK(cudaMemcpy(host_output.data(), d_current_input, current_input_size * sizeof(double), cudaMemcpyDeviceToHost));

    // Free the last output
    CUDA_CHECK(cudaFree(d_current_input));

    return host_output;
}

// Example CUDA kernel for backpropagation (simplified)
__global__ void backpropagateKernel(double* weights, double* biases, double* input, double* target, double* output, double learningRate, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < outputSize) {
        double error = target[idx] - output[idx];
        double delta = error * (1 - output[idx] * output[idx]); // Derivative of tanh
        for(int i = 0; i < inputSize; ++i) {
            weights[i * outputSize + idx] += learningRate * delta * input[i];
        }
        biases[idx] += learningRate * delta;
    }
}

// Backpropagation implementation using CUDA
void NeuralNetwork::backpropagate(const std::vector<double>& input, const std::vector<double>& target, double learningRate) {
    if(input.size() != static_cast<size_t>(layerSizes[0])) {
        throw std::invalid_argument("Input size does not match network input layer size.");
    }
    if(target.size() != static_cast<size_t>(layerSizes.back())) {
        throw std::invalid_argument("Target size does not match network output layer size.");
    }

    // Use std::vector with custom deleter for CUDA memory
    auto cudaDeleter = [](double* ptr) { cudaFree(ptr); };
    std::unique_ptr<double, decltype(cudaDeleter)> d_input(nullptr, cudaDeleter);
    std::unique_ptr<double, decltype(cudaDeleter)> d_target(nullptr, cudaDeleter);
    std::unique_ptr<double, decltype(cudaDeleter)> d_output(nullptr, cudaDeleter);

    size_t inputSize = input.size() * sizeof(double);
    size_t outputSize = layerSizes.back() * sizeof(double);

    double* raw_d_input;
    double* raw_d_target;
    double* raw_d_output;
    CUDA_CHECK(cudaMalloc(&raw_d_input, inputSize));
    CUDA_CHECK(cudaMalloc(&raw_d_target, outputSize));
    CUDA_CHECK(cudaMalloc(&raw_d_output, outputSize));
    d_input.reset(raw_d_input);
    d_target.reset(raw_d_target);
    d_output.reset(raw_d_output);

    // Copy input and target to device
    CUDA_CHECK(cudaMemcpy(d_input.get(), input.data(), inputSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target.get(), target.data(), outputSize, cudaMemcpyHostToDevice));

    // Forward pass to get the output
    forwardKernel<<<(layerSizes.back() + 255) / 256, 256>>>(d_weights.get(), d_biases.get(), d_input.get(), d_output.get(), layerSizes[0], layerSizes.back());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch backpropagation kernel
    backpropagateKernel<<<(layerSizes.back() + 255) / 256, 256>>>(d_weights.get(), d_biases.get(), d_input.get(), d_target.get(), d_output.get(), learningRate, layerSizes[0], layerSizes.back());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free device memory
    d_input.reset();
    d_target.reset();
    d_output.reset();
}

void NeuralNetwork::allocateDeviceMemory() {
    d_weights = makeCudaUniquePtr<double>(host_weights.size());
    d_biases = makeCudaUniquePtr<double>(host_biases.size());
}



void NeuralNetwork::copyToDevice() {
    CUDA_CHECK(cudaMemcpy(d_weights.get(), host_weights.data(), 
                          host_weights.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_biases.get(), host_biases.data(), 
                          host_biases.size() * sizeof(double), cudaMemcpyHostToDevice));
}

void NeuralNetwork::copyFromDevice() {
    CUDA_CHECK(cudaMemcpy(host_weights.data(), d_weights.get(), 
                          host_weights.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_biases.data(), d_biases.get(), 
                          host_biases.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

// CPU implementations (if CUDA is not available)
std::vector<double> NeuralNetwork::cpuForward(const std::vector<double>& input) {
    // Implement CPU-based forward pass (optional)
    // ...
    return std::vector<double>(); // Placeholder
}

void NeuralNetwork::cpuBackpropagate(const std::vector<double>& input, const std::vector<double>& target, double learningRate) {
    // Implement CPU-based backpropagation (optional)
    // ...
}


void NeuralNetwork::copyWeightsAndBiasesFrom(const NeuralNetwork& other) {
    host_weights = other.host_weights;  // This will use move semantics if possible
    host_biases = other.host_biases;

    freeDeviceMemory();
    allocateDeviceMemory();
    copyWeightsToDevice();
    copyBiasesToDevice();
}



