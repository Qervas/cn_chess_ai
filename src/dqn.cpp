#include "dqn.h"
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <cstring> // For std::strcmp if needed

// Constructor
DQN::DQN(const std::vector<int>& layerSizes, double learningRate, double gamma)
    : qNetwork(std::make_unique<NeuralNetwork>(layerSizes)),
      targetNetwork(std::make_unique<NeuralNetwork>(layerSizes)),
      learningRate(learningRate),
      gamma(gamma)
{
    syncTargetNetwork();
}

// Destructor
DQN::~DQN() {}

// Select action using epsilon-greedy strategy
int DQN::selectAction(const std::vector<double>& state, double epsilon)
{
    if(static_cast<double>(rand()) / RAND_MAX < epsilon) {
        // Return a random action
        return rand() % static_cast<int>(state.size()); // Ensure state.size() is non-zero
    } else {
        // Return the action with the highest Q-value
        std::vector<double> qValues = qNetwork->forward(state);
        return static_cast<int>(std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end())));
    }
}

// Backpropagation to train the network
void DQN::backpropagate(const std::vector<double>& state, const std::vector<double>& target, double learningRate)
{
    qNetwork->backpropagate(state, target, learningRate);
}

// Get Q-values for a given state
std::vector<double> DQN::getQValues(const std::vector<double>& state)
{
    return qNetwork->forward(state);
}

// Update target network to match Q-network
void DQN::updateTargetNetwork()
{
    *targetNetwork = *qNetwork;
}

// Save model weights and biases
void DQN::saveModel(const std::string& filename)
{
    // Open the file in binary mode
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        throw std::runtime_error("Unable to open file for saving model.");
    }

    // Serialize weights
    size_t weightsSize = qNetwork->host_weights.size();
    outFile.write(reinterpret_cast<const char*>(qNetwork->host_weights.data()), weightsSize * sizeof(double));
    if (!outFile) {
        throw std::runtime_error("Error writing weights to model file.");
    }

    // Serialize biases
    size_t biasesSize = qNetwork->host_biases.size();
    outFile.write(reinterpret_cast<const char*>(qNetwork->host_biases.data()), biasesSize * sizeof(double));
    if (!outFile) {
        throw std::runtime_error("Error writing biases to model file.");
    }

    // Optionally, serialize layer sizes for validation during loading
    size_t layerSizesSize = qNetwork->layerSizes.size();
    outFile.write(reinterpret_cast<const char*>(&layerSizesSize), sizeof(size_t));
    if (!outFile) {
        throw std::runtime_error("Error writing layer sizes to model file.");
    }

    outFile.write(reinterpret_cast<const char*>(qNetwork->layerSizes.data()), layerSizesSize * sizeof(int));
    if (!outFile) {
        throw std::runtime_error("Error writing layer sizes to model file.");
    }

    outFile.close();
}

// Load model weights and biases
void DQN::loadModel(const std::string& filename)
{
    // Open the file in binary mode
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open()) {
        throw std::runtime_error("Unable to open file for loading model.");
    }

    // Deserialize weights
    size_t weightsSize = qNetwork->host_weights.size();
    inFile.read(reinterpret_cast<char*>(qNetwork->host_weights.data()), weightsSize * sizeof(double));
    if (inFile.gcount() != static_cast<std::streamsize>(weightsSize * sizeof(double))) {
        throw std::runtime_error("Error reading weights from model file.");
    }

    // Deserialize biases
    size_t biasesSize = qNetwork->host_biases.size();
    inFile.read(reinterpret_cast<char*>(qNetwork->host_biases.data()), biasesSize * sizeof(double));
    if (inFile.gcount() != static_cast<std::streamsize>(biasesSize * sizeof(double))) {
        throw std::runtime_error("Error reading biases from model file.");
    }

    // Optionally, deserialize layer sizes for validation
    size_t layerSizesSize = 0;
    inFile.read(reinterpret_cast<char*>(&layerSizesSize), sizeof(size_t));
    if (inFile.gcount() != sizeof(size_t)) {
        throw std::runtime_error("Error reading layer sizes from model file.");
    }

    std::vector<int> loadedLayerSizes(layerSizesSize);
    inFile.read(reinterpret_cast<char*>(loadedLayerSizes.data()), layerSizesSize * sizeof(int));
    if (inFile.gcount() != static_cast<std::streamsize>(layerSizesSize * sizeof(int))) {
        throw std::runtime_error("Error reading layer sizes from model file.");
    }

    // Validate layer sizes
    if (loadedLayerSizes != qNetwork->layerSizes) {
        throw std::runtime_error("Layer sizes in the model file do not match the current network architecture.");
    }

    inFile.close();

    // Copy weights and biases to device
    qNetwork->copyToDevice();
}

// Implement the train method
void DQN::train(const std::vector<double>& state, int action, double reward, const std::vector<double>& nextState, bool done)
{
    // Get current Q-values
    std::vector<double> currentQ = qNetwork->forward(state);

    // Compute target Q-value
    if (done) {
        currentQ[action] = reward;
    } else {
        std::vector<double> nextQ = targetNetwork->forward(nextState);
        currentQ[action] = reward + gamma * *std::max_element(nextQ.begin(), nextQ.end());
    }

    // Perform backpropagation to update the network
    qNetwork->backpropagate(state, currentQ, learningRate);
}

// Sync target network with Q-network
void DQN::syncTargetNetwork() {
    // Directly copy the flattened weights and biases
    qNetwork->host_weights = qNetwork->host_weights; // This line appears redundant
    qNetwork->host_biases = qNetwork->host_biases;   // This line appears redundant

    // Instead, you should copy from qNetwork to targetNetwork
    targetNetwork->host_weights = qNetwork->host_weights;
    targetNetwork->host_biases = qNetwork->host_biases;

    // Copy to device
    targetNetwork->copyToDevice();
}
