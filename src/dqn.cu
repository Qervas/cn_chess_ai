#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdio> 
#include <QDebug>
#include "dqn.h"
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

__global__ void forwardKernel(double* weights, double* biases, double* input, double* output, double* z, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize) {
        double sum = biases[idx];
        for (int i = 0; i < inputSize; ++i) {
            size_t weight_idx = idx * inputSize + i;
            sum += input[i] * weights[weight_idx];
        }
        z[idx] = sum;
        output[idx] = tanh(sum); // Activation function
    }
}

__global__ void outputLayerDeltaKernel(double* activation, double* target, double* z, double* delta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double error = activation[idx] - target[idx];
        double derivative = 1 - tanh(z[idx]) * tanh(z[idx]); // Derivative of tanh
        delta[idx] = error * derivative;
    }
}

__global__ void hiddenLayerDeltaKernel(double* weights_next_layer, double* delta_next_layer, double* z, double* delta, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize) {
        double sum = 0.0;
        for (int i = 0; i < inputSize; ++i) {
            size_t weight_idx = i * outputSize + idx;
            sum += weights_next_layer[weight_idx] * delta_next_layer[i];
        }
        double derivative = 1 - tanh(z[idx]) * tanh(z[idx]); // Derivative of tanh
        delta[idx] = sum * derivative;
    }
}

__global__ void updateWeightsBiasesKernel(double* weights, double* biases, double* activation_prev_layer, double* delta, double learningRate, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize) {
        biases[idx] -= learningRate * delta[idx];
        for (int i = 0; i < inputSize; ++i) {
            size_t weight_idx = idx * inputSize + i;
            weights[weight_idx] -= learningRate * delta[idx] * activation_prev_layer[i];
        }
    }
}


// Backpropagation implementation using CUDA
void NeuralNetwork::backpropagate(const std::vector<double>& input, const std::vector<double>& target, double learningRate) {
    if (input.size() != static_cast<size_t>(layerSizes[0])) {
        throw std::invalid_argument("Input size does not match network input layer size.");
    }
    if (target.size() != static_cast<size_t>(layerSizes.back())) {
        throw std::invalid_argument("Target size does not match network output layer size.");
    }

    // Forward pass to store activations and z-values
    std::vector<double*> activations(numLayers + 1);
    std::vector<double*> zs(numLayers);

    // Copy input to device
    double* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice));
    activations[0] = d_input;

    // Forward pass
    for (int layer = 0; layer < numLayers; ++layer) {
        int inputSize = layerSizes[layer];
        int outputSize = layerSizes[layer + 1];

        // Allocate memory for output and z
        double* d_output;
        double* d_z;
        CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_z, outputSize * sizeof(double)));

        // Get pointers to weights and biases for this layer
        double* d_layer_weights = d_weights.get() + weightOffsets[layer];
        double* d_layer_biases = d_biases.get() + biasOffsets[layer];

        // Launch the forward kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (outputSize + threadsPerBlock - 1) / threadsPerBlock;
        forwardKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_layer_weights,
            d_layer_biases,
            activations[layer],
            d_output,
            d_z,
            inputSize,
            outputSize);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Free the input of the previous layer if it's not the original input
        if (layer > 0) {
            CUDA_CHECK(cudaFree(activations[layer]));
        }

        activations[layer + 1] = d_output;
        zs[layer] = d_z;
    }

    // Copy target to device
    double* d_target;
    CUDA_CHECK(cudaMalloc(&d_target, target.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_target, target.data(), target.size() * sizeof(double), cudaMemcpyHostToDevice));

    // Backward pass
    std::vector<double*> deltas(numLayers);

    // Compute delta for the output layer
    int outputLayer = numLayers - 1;
    int outputSize = layerSizes[outputLayer + 1];

    double* d_delta;
    CUDA_CHECK(cudaMalloc(&d_delta, outputSize * sizeof(double)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (outputSize + threadsPerBlock - 1) / threadsPerBlock;
    outputLayerDeltaKernel<<<blocksPerGrid, threadsPerBlock>>>(
        activations[outputLayer + 1],
        d_target,
        zs[outputLayer],
        d_delta,
        outputSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    deltas[outputLayer] = d_delta;

    // Backpropagate the error
    for (int layer = outputLayer - 1; layer >= 0; --layer) {
        int inputSize = layerSizes[layer + 1];
        int outputSize = layerSizes[layer];

        double* d_next_delta = deltas[layer + 1];
        double* d_delta;
        CUDA_CHECK(cudaMalloc(&d_delta, outputSize * sizeof(double)));

        double* d_layer_weights = d_weights.get() + weightOffsets[layer + 1];

        blocksPerGrid = (outputSize + threadsPerBlock - 1) / threadsPerBlock;
        hiddenLayerDeltaKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_layer_weights,
            d_next_delta,
            zs[layer],
            d_delta,
            inputSize,
            outputSize);
        CUDA_CHECK(cudaDeviceSynchronize());

        deltas[layer] = d_delta;
    }

    // Update weights and biases
    for (int layer = 0; layer < numLayers; ++layer) {
        int inputSize = layerSizes[layer];
        int outputSize = layerSizes[layer + 1];

        double* d_layer_weights = d_weights.get() + weightOffsets[layer];
        double* d_layer_biases = d_biases.get() + biasOffsets[layer];

        blocksPerGrid = (outputSize + threadsPerBlock - 1) / threadsPerBlock;
        updateWeightsBiasesKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_layer_weights,
            d_layer_biases,
            activations[layer],
            deltas[layer],
            learningRate,
            inputSize,
            outputSize);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Free allocated memory

    // Free activations that haven't been freed yet
    CUDA_CHECK(cudaFree(activations[0]));               // Free the input activation
    CUDA_CHECK(cudaFree(activations[numLayers]));       // Free the final output activation

    // Free zs (all layers)
    for (int layer = 0; layer < numLayers; ++layer) {
        CUDA_CHECK(cudaFree(zs[layer]));
    }

    // Free deltas (all layers)
    for (int layer = 0; layer < numLayers; ++layer) {
        CUDA_CHECK(cudaFree(deltas[layer]));
    }

    // Free the target
    CUDA_CHECK(cudaFree(d_target));
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



