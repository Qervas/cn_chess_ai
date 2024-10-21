#ifndef DQN_H
#define DQN_H

#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <QObject>
#include <cuda_runtime.h>
#include "action.h"

// CUDA error checking macro
inline void checkCudaError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + 
                                 cudaGetErrorString(error) + 
                                 " at " + file + ":" + std::to_string(line));
    }
}

#define CUDA_CHECK(call) checkCudaError(call, __FILE__, __LINE__)

// Custom deleter for CUDA memory
struct CudaDeleter {
    void operator()(void* ptr) const {
        cudaFree(ptr);
    }
};

// Wrapper for CUDA memory
template<typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

template<typename T>
CudaUniquePtr<T> makeCudaUniquePtr(size_t size) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
    return CudaUniquePtr<T>(ptr);
}
class NeuralNetwork {
public:

    // Host-side weights and biases stored as flat vectors
    std::vector<double> host_weights; // Flattened: layer_sizes[i] * layer_sizes[i+1]
    std::vector<double> host_biases;  // Flattened: sum of layer_sizes[i+1]
    std::vector<size_t> weightOffsets;
    std::vector<size_t> biasOffsets;


	NeuralNetwork() = delete;
    NeuralNetwork(const std::vector<int>& layerSizes);
    NeuralNetwork(const NeuralNetwork& other);
	NeuralNetwork(NeuralNetwork&& other) noexcept;
    ~NeuralNetwork();
    NeuralNetwork& operator=(const NeuralNetwork& other);
	NeuralNetwork& operator=(NeuralNetwork&& other) noexcept;


	

    // Forward pass 
    std::vector<double> forward(const std::vector<double>& input);
    // Backpropagation 
    void backpropagate(const std::vector<double>& input, const std::vector<double>& target, double learningRate);

    // Method to copy weights and biases to device (if CUDA is available)
    void copyToDevice();
	void copyFromDevice();
	void initializeHostWeightsAndBiases();
    // Method to copy weights and biases from another NeuralNetwork
    void copyWeightsAndBiasesFrom(const NeuralNetwork& other);

private:
    // Layer information
    int numLayers;
public:
    std::vector<int> layerSizes;
private:
    // Device-side pointers for weights and biases (only when CUDA is available)
    CudaUniquePtr<double> d_weights;
    CudaUniquePtr<double> d_biases;

    // CUDA memory management
    void allocateDeviceMemory();
    void copyWeightsToDevice();
    void copyBiasesToDevice();
    void freeDeviceMemory();

    // CPU implementations (can be used as fallback or when CUDA is not available)
    std::vector<double> cpuForward(const std::vector<double>& input);
    void cpuBackpropagate(const std::vector<double>& input, const std::vector<double>& target, double learningRate);
};

class DQN : public QObject {
    Q_OBJECT
public:
    DQN(const std::vector<int>& layerSizes, double learningRate = 0.001, double gamma = 0.99);
    virtual ~DQN() = default;

    Action selectAction(const std::vector<double>& state, double epsilon, const std::vector<Action> validActions);
    void backpropagate(const std::vector<double>& state, const std::vector<double>& target, double learningRate);
    std::vector<double> getQValues(const std::vector<double>& state);
    void updateTargetNetwork();
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);
    void train(const std::vector<double>& state, int action, double reward, const std::vector<double>& nextState, bool done);

private:
    std::unique_ptr<NeuralNetwork> qNetwork;
    std::unique_ptr<NeuralNetwork> targetNetwork;
    double learningRate;
    double gamma;
};

#endif // DQN_H
