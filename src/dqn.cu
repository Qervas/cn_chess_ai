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
// CUDA error checking macro
#define CUDA_CHECK_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constructor
NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes_)
    : layerSizes(layerSizes_)
{
    if(layerSizes.size() < 2) {
        throw std::invalid_argument("NeuralNetwork must have at least two layers (input and output).");
    }

    numLayers = layerSizes.size() - 1;

    // Initialize host weights and biases with random values
    initializeHostWeightsAndBiases();

#ifdef __CUDACC__
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
#endif
}


bool isMemoryAvailable(size_t requiredBytes) {
   void* testAlloc = std::malloc(requiredBytes);
   if (testAlloc) {
       std::free(testAlloc);
       return true;
   }
   return false;
}

void NeuralNetwork::initializeHostWeightsAndBiases() {
    // Clear the vectors before reinitializing
    host_weights.clear();
    host_biases.clear();

    // Calculate total number of weights and biases
    size_t totalWeights = 0;
    size_t totalBiases = 0;
    for(int i = 0; i < numLayers; ++i) {
        totalWeights += static_cast<size_t>(layerSizes[i]) * layerSizes[i+1];
        totalBiases += layerSizes[i+1];
    }

    host_weights.reserve(totalWeights);
    host_biases.reserve(totalBiases);

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.05, 0.05); // Adjust range as needed

    for(int i = 0; i < numLayers; ++i) {
        int inputSize = layerSizes[i];
        int outputSize = layerSizes[i+1];

        for(int j = 0; j < inputSize; ++j) {
            for(int k = 0; k < outputSize; ++k) {
                host_weights.emplace_back(dis(gen));
            }
        }

        for(int k = 0; k < outputSize; ++k) {
            host_biases.emplace_back(0.0); // Initialize biases to zero
        }
    }
}


// Copy constructor
NeuralNetwork::NeuralNetwork(const NeuralNetwork& other)
    : numLayers(other.numLayers), layerSizes(other.layerSizes),
      host_weights(other.host_weights), host_biases(other.host_biases),
      d_weights(nullptr), d_biases(nullptr)
{
#ifdef __CUDACC__
    allocateDeviceMemory();
    copyWeightsToDevice();
    copyBiasesToDevice();
#endif
}

// Copy assignment operator
NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& other) {
    if (this == &other) return *this;

#ifdef __CUDACC__
    // Free existing device memory
    freeDeviceMemory();
#endif

    // Copy layer information and weights/biases
    numLayers = other.numLayers;
    layerSizes = other.layerSizes;
    host_weights = other.host_weights;
    host_biases = other.host_biases;

#ifdef __CUDACC__
    // Allocate and copy to device
    allocateDeviceMemory();
    copyWeightsToDevice();
    copyBiasesToDevice();
#endif

    return *this;
}

// Move constructor
NeuralNetwork::NeuralNetwork(NeuralNetwork&& other) noexcept
    : numLayers(other.numLayers),
      layerSizes(std::move(other.layerSizes)),
      host_weights(std::move(other.host_weights)),
      host_biases(std::move(other.host_biases))
{
#ifdef __CUDACC__
    d_weights = other.d_weights;
    d_biases = other.d_biases;
    other.d_weights = nullptr;
    other.d_biases = nullptr;
#endif
}


// Move assignment operator
NeuralNetwork& NeuralNetwork::operator=(NeuralNetwork&& other) noexcept {
    if (this != &other) {
#ifdef __CUDACC__
        // Free existing device memory
        freeDeviceMemory();

        // Move device memory pointers
        d_weights = other.d_weights;
        d_biases = other.d_biases;
        other.d_weights = nullptr;
        other.d_biases = nullptr;
#endif

        // Move layer information and weights/biases
        numLayers = other.numLayers;
        layerSizes = std::move(other.layerSizes);
        host_weights = std::move(other.host_weights);
        host_biases = std::move(other.host_biases);
    }
    return *this;
}


// Destructor
NeuralNetwork::~NeuralNetwork() {
#ifdef __CUDACC__
    freeDeviceMemory();
#endif
}

#ifdef __CUDACC__
// Allocate device memory for weights and biases
void NeuralNetwork::allocateDeviceMemory() {
    size_t weightsSize = host_weights.size() * sizeof(double);
    size_t biasesSize = host_biases.size() * sizeof(double);

    qDebug() << "Allocating device memory for weights:" << weightsSize << "bytes";
    qDebug() << "Allocating device memory for biases:" << biasesSize << "bytes";

    cudaError_t err = cudaMalloc(&d_weights, weightsSize);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error (cudaMalloc weights): ") + cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_biases, biasesSize);
    if (err != cudaSuccess) {
        cudaFree(d_weights); // Free the previously allocated memory
        throw std::runtime_error(std::string("CUDA Error (cudaMalloc biases): ") + cudaGetErrorString(err));
    }
}

// Copy weights from host to device
void NeuralNetwork::copyWeightsToDevice() {
    if (!d_weights) {
        throw std::runtime_error("Device memory for weights is not allocated.");
    }

    size_t size = host_weights.size() * sizeof(double);
    CUDA_CHECK_ERROR(cudaMemcpy(d_weights, host_weights.data(), size, cudaMemcpyHostToDevice));
}

// Copy biases from host to device
void NeuralNetwork::copyBiasesToDevice() {
    if (!d_biases) {
        throw std::runtime_error("Device memory for biases is not allocated.");
    }

    size_t size = host_biases.size() * sizeof(double);
    CUDA_CHECK_ERROR(cudaMemcpy(d_biases, host_biases.data(), size, cudaMemcpyHostToDevice));
}
// Free device memory
void NeuralNetwork::freeDeviceMemory() {
    if(d_weights) {
        cudaFree(d_weights);
        d_weights = nullptr;
    }
    if(d_biases) {
        cudaFree(d_biases);
        d_biases = nullptr;
    }
}



// Example CUDA kernel for forward pass (simplified)
__global__ void forwardKernel(double* weights, double* biases, double* input, double* output, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < outputSize) {
        double sum = 0.0;
        for(int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[i * outputSize + idx];
        }
        sum += biases[idx];
        output[idx] = tanh(sum); // Activation function
    }
}

#ifdef __CUDACC__
// Forward pass implementation using CUDA
std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    if(input.size() != static_cast<size_t>(layerSizes[0])) {
        throw std::invalid_argument("Input size does not match network input layer size.");
    }

    // Allocate device memory for input and output
    double* d_input = nullptr;
    double* d_output = nullptr;
    size_t inputSize = input.size() * sizeof(double);
    size_t outputSize = layerSizes[1] * sizeof(double);

    CUDA_CHECK_ERROR(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK_ERROR(cudaMalloc(&d_output, outputSize));

    // Copy input to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_input, input.data(), inputSize, cudaMemcpyHostToDevice));

    // Launch forward kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (layerSizes[1] + threadsPerBlock - 1) / threadsPerBlock;
    forwardKernel<<<blocksPerGrid, threadsPerBlock>>>(d_weights, d_biases, d_input, d_output, layerSizes[0], layerSizes[1]);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Copy output back to host
    std::vector<double> host_output(layerSizes[1]);
    CUDA_CHECK_ERROR(cudaMemcpy(host_output.data(), d_output, outputSize, cudaMemcpyDeviceToHost));

    // Free device memory for input and output
    cudaFree(d_input);
    cudaFree(d_output);

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
    if(target.size() != static_cast<size_t>(layerSizes[1])) {
        throw std::invalid_argument("Target size does not match network output layer size.");
    }

    // Allocate device memory for input, target, and output
    double* d_input = nullptr;
    double* d_target = nullptr;
    double* d_output = nullptr;
    size_t inputSize = input.size() * sizeof(double);
    size_t outputSize = layerSizes[1] * sizeof(double);

    CUDA_CHECK_ERROR(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK_ERROR(cudaMalloc(&d_target, outputSize));
    CUDA_CHECK_ERROR(cudaMalloc(&d_output, outputSize));

    // Copy input and target to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_input, input.data(), inputSize, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_target, target.data(), outputSize, cudaMemcpyHostToDevice));

    // Forward pass to get the output
    forwardKernel<<<(layerSizes[1] + 255) / 256, 256>>>(d_weights, d_biases, d_input, d_output, layerSizes[0], layerSizes[1]);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Launch backpropagation kernel
    backpropagateKernel<<<(layerSizes[1] + 255) / 256, 256>>>(d_weights, d_biases, d_input, d_target, d_output, learningRate, layerSizes[0], layerSizes[1]);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_output);
}
#endif

// Copy all weights and biases to device
void NeuralNetwork::copyToDevice() {
    copyWeightsToDevice();
    copyBiasesToDevice();
}
#endif

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


