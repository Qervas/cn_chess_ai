#include "dqn.h"
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <ctime>
#include <limits>
#include <QDebug>
#include <QFile>

// Constructor
DQN::DQN(const std::vector<int>& layerSizes, double learningRate, double gamma)
    : qNetwork(std::make_unique<NeuralNetwork>(layerSizes)),
      targetNetwork(std::make_unique<NeuralNetwork>(layerSizes)),
      learningRate(learningRate),
      gamma(gamma)
{
    updateTargetNetwork();
    std::srand(static_cast<unsigned int>(std::time(nullptr))); // Seed RNG
}


// Select action using epsilon-greedy strategy
Action DQN::selectAction(const std::vector<double>& state, double epsilon, const std::vector<Action> validActions)
{
    if(validActions.empty()) {
        throw std::runtime_error("No valid actions available.");
    }

    double randValue = static_cast<double>(rand()) / RAND_MAX;
    if(randValue < epsilon) {
        // Exploration: Return a random valid action
        int randomIndex = rand() % validActions.size();
        return validActions[randomIndex];
    } else {
        // Exploitation: Choose the best valid action based on Q-values
        std::vector<double> qValues = qNetwork->forward(state);

        double maxQ = -std::numeric_limits<double>::infinity();
        Action bestAction = validActions[0];

        for(const auto& action : validActions) {
            if(action.to >= qValues.size()) {
                qDebug() << "Warning: Action.to index out of bounds.";
                continue; // Skip invalid indices
            }
            double q = qValues[action.to]; // Assuming action.to uniquely identifies the action
            if(q > maxQ) {
                maxQ = q;
                bestAction = action;
            }
        }

        return bestAction;
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
void DQN::updateTargetNetwork() {
    targetNetwork->copyWeightsAndBiasesFrom(*qNetwork);
}

// Save model weights and biases
void DQN::saveModel(const std::string& filename)
{
    QFile outFile(QString::fromStdString(filename));
    if (!outFile.open(QIODevice::WriteOnly)) {
        throw std::runtime_error("Unable to open file for saving model.");
    }

    QDataStream out(&outFile);
    out.setVersion(QDataStream::Qt_6_6);
	
    // Serialize weights
    size_t weightsSize = qNetwork->host_weights.size();
    out.writeRawData(reinterpret_cast<const char*>(qNetwork->host_weights.data()), weightsSize * sizeof(double));
    if (out.status() != QDataStream::Ok) {
        throw std::runtime_error("Error writing weights to model file.");
    }

    // Serialize biases
    size_t biasesSize = qNetwork->host_biases.size();
    out.writeRawData(reinterpret_cast<const char*>(qNetwork->host_biases.data()), biasesSize * sizeof(double));
    if (out.status() != QDataStream::Ok) {
        throw std::runtime_error("Error writing biases to model file.");
    }

    // Serialize layer sizes
    size_t layerSizesSize = qNetwork->layerSizes.size();
    out << static_cast<quint64>(layerSizesSize);
    for (int size : qNetwork->layerSizes) {
        out << size;
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

