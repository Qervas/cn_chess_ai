#ifndef DQN_H
#define DQN_H

#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

class NeuralNetwork {
public:

    // Host-side weights and biases stored as flat vectors
    std::vector<double> host_weights; // Flattened: layer_sizes[i] * layer_sizes[i+1]
    std::vector<double> host_biases;  // Flattened: sum of layer_sizes[i+1]


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
	void initializeHostWeightsAndBiases();
    // Method to copy weights and biases from another NeuralNetwork
    void copyWeightsAndBiasesFrom(const NeuralNetwork& other);

private:
    // Layer information
    int numLayers;
public:
    std::vector<int> layerSizes;
private:
#ifdef __CUDACC__
    // Device-side pointers for weights and biases (only when CUDA is available)
    double* d_weights;
    double* d_biases;

    // CUDA memory management
    void allocateDeviceMemory();
    void copyWeightsToDevice();
    void copyBiasesToDevice();
    void freeDeviceMemory();
#endif

    // CPU implementations (can be used as fallback or when CUDA is not available)
    std::vector<double> cpuForward(const std::vector<double>& input);
    void cpuBackpropagate(const std::vector<double>& input, const std::vector<double>& target, double learningRate);
};

class DQN {
public:
    DQN(const std::vector<int>& layerSizes, double learningRate = 0.001, double gamma = 0.99);
    ~DQN();

    int selectAction(const std::vector<double>& state, double epsilon);
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
    // ReplayBuffer and other members remain unchanged
};

#endif // DQN_H
