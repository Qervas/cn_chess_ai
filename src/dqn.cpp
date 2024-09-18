
#include "dqn.h"

// Neural Network implementation
NeuralNetwork::NeuralNetwork(const QVector<int>& layerSizes) {
    for (int i = 1; i < layerSizes.size(); ++i) {
        weights.push_back(QVector<QVector<double>>(layerSizes[i], QVector<double>(layerSizes[i-1])));
        biases.push_back(QVector<double>(layerSizes[i]));
        activations.push_back(QVector<double>(layerSizes[i]));
    }

    QRandomGenerator gen;

    for (auto& layer : weights) {
        for (auto& neuron : layer) {
            for (auto& weight : neuron) {
                weight = gen.generateDouble() * 2 - 1; // Generate values between -1 and 1
            }
        }
    }

    for (auto& layer : biases) {
        for (auto& bias : layer) {
            bias = gen.generateDouble() * 2 - 1; // Generate values between -1 and 1
        }
    }
}

QVector<double> NeuralNetwork::forward(const QVector<double>& input) {
    QVector<double> current = input;
    for (int i = 0; i < weights.size(); ++i) {
        QVector<double> next(weights[i].size());
        for (int j = 0; j < weights[i].size(); ++j) {
            double sum = 0;
            for (int k = 0; k < current.size(); ++k) {
                sum += current[k] * weights[i][j][k];
            }
            next[j] = std::tanh(sum + biases[i][j]);
        }
        current = next;
        activations[i] = current;
    }
    return current;
}

void NeuralNetwork::backpropagate(const QVector<double>& input, const QVector<double>& target, double learningRate) {
    QVector<QVector<double>> deltas(weights.size());

    // Compute deltas for output layer
    deltas.back().resize(activations.back().size());
    for (int i = 0; i < activations.back().size(); ++i) {
        double output = activations.back()[i];
        deltas.back()[i] = (target[i] - output) * (1 - output * output);
    }

    // Compute deltas for hidden layers
    for (int i = weights.size() - 2; i >= 0; --i) {
        deltas[i].resize(weights[i].size());
        for (int j = 0; j < weights[i].size(); ++j) {
            double sum = 0.0;
            for (int k = 0; k < weights[i+1].size(); ++k) {
                sum += weights[i+1][k][j] * deltas[i+1][k];
            }
            deltas[i][j] = sum * (1 - activations[i][j] * activations[i][j]);
        }
    }

    // Update weights and biases
    QVector<double> prevActivation = input;
    for (int i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < weights[i].size(); ++j) {
            for (int k = 0; k < weights[i][j].size(); ++k) {
                weights[i][j][k] += learningRate * deltas[i][j] * prevActivation[k];
            }
            biases[i][j] += learningRate * deltas[i][j];
        }
        prevActivation = activations[i];
    }
}

// Replay Buffer implementation
ReplayBuffer::ReplayBuffer(int capacity) : capacity(capacity) {}

void ReplayBuffer::add(const QVector<double>& state, int action, double reward, const QVector<double>& nextState, bool done) {
    if (buffer.size() >= capacity) {
        buffer.removeFirst();
    }
    QVariantList experience;
    experience << QVariant::fromValue(state)
               << QVariant::fromValue(action)
               << QVariant::fromValue(reward)
               << QVariant::fromValue(nextState)
               << QVariant::fromValue(done);
    buffer.append(QVariant::fromValue(experience));
}

QVector<QVariantList> ReplayBuffer::sample(int batchSize) {
    QVector<QVariantList> batch;
    for (int i = 0; i < batchSize; ++i) {
        int index = QRandomGenerator::global()->bounded(buffer.size());
        batch.append(buffer[index].value<QVariantList>());
    }
    return batch;
}

// DQN implementation
DQN::DQN(int stateSize, int actionSize, int hiddenSize, double learningRate, double gamma, int batchSize)
    : stateSize(stateSize), actionSize(actionSize), learningRate(learningRate), gamma(gamma), batchSize(batchSize) {
    qNetwork = std::make_unique<NeuralNetwork>(QVector<int>{stateSize, hiddenSize, actionSize});
    targetNetwork = std::make_unique<NeuralNetwork>(QVector<int>{stateSize, hiddenSize, actionSize});
    replayBuffer = std::make_unique<ReplayBuffer>(10000);
}

int DQN::selectAction(const QVector<double>& state, double epsilon) {
    if (QRandomGenerator::global()->generateDouble() < epsilon) {
        return QRandomGenerator::global()->bounded(actionSize);
    } else {
        QVector<double> qValues = qNetwork->forward(state);
        return std::max_element(qValues.begin(), qValues.end()) - qValues.begin();
    }
}

void DQN::train(const QVector<double>& state, int action, double reward, const QVector<double>& nextState, bool done) {
    replayBuffer->add(state, action, reward, nextState, done);

    if (replayBuffer->sample(1).size() < batchSize) return;

    auto batch = replayBuffer->sample(batchSize);
    for (const auto& experience : batch) {
        QVector<double> s = experience[0].value<QVector<double>>();
        int a = experience[1].toInt();
        double r = experience[2].toDouble();
        QVector<double> ns = experience[3].value<QVector<double>>();
        bool d = experience[4].toBool();

        QVector<double> targetQ = qNetwork->forward(s);
        if (d) {
            targetQ[a] = r;
        } else {
            QVector<double> nextQ = targetNetwork->forward(ns);
            targetQ[a] = r + gamma * *std::max_element(nextQ.begin(), nextQ.end());
        }
        qNetwork->backpropagate(s, targetQ, learningRate);
    }
}

void DQN::updateTargetNetwork() {
    *targetNetwork = *qNetwork;
}

QVector<double> DQN::getQValues(const QVector<double>& state) {
    return qNetwork->forward(state);
}

void DQN::saveModel(const QString& filename) {
    QFile file(filename);
    if (file.open(QIODevice::WriteOnly)) {
        QDataStream out(&file);
        // save Q network weights and biases
        out << qNetwork->weights << qNetwork->biases;
        file.close();
    }
}

void DQN::loadModel(const QString& filename) {
    QFile file(filename);
    if (file.open(QIODevice::ReadOnly)) {
        QDataStream in(&file);
        // load Q network weights and biases
        in >> qNetwork->weights >> qNetwork->biases;
        file.close();
        updateTargetNetwork();
    }
}
