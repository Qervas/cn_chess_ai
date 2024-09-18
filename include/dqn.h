#ifndef DQN_H
#define DQN_H

#include <QVector>
#include <QPair>
#include <QRandomGenerator>
#include <memory>
#include <QVariant>
#include <QtAlgorithms>
#include <QtMath>
#include <QFile>
#include <QDataStream>

class NeuralNetwork {
public:
    NeuralNetwork(const QVector<int>& layerSizes);
    QVector<double> forward(const QVector<double>& input);
    void backpropagate(const QVector<double>& input, const QVector<double>& target, double learningRate);
    QVector<QVector<QVector<double>>> weights;
    QVector<QVector<double>> biases;

private:
    QVector<QVector<double>> activations;
};

class ReplayBuffer {
public:
    ReplayBuffer(int capacity);
    void add(const QVector<double>& state, int action, double reward, const QVector<double>& nextState, bool done);
    QVector<QVariantList> sample(int batchSize);

private:
    int capacity;
    QVector<QVariant> buffer;
    QRandomGenerator rng;
};

class DQN {
public:
    DQN(int stateSize, int actionSize, int hiddenSize, double learningRate, double gamma, int batchSize);
    int selectAction(const QVector<double>& state, double epsilon);
    void train(const QVector<double>& state, int action, double reward, const QVector<double>& nextState, bool done);
    void updateTargetNetwork();
    QVector<double> getQValues(const QVector<double>& state);
    void saveModel(const QString& filename);
    void loadModel(const QString& filename);

private:
    std::unique_ptr<NeuralNetwork> qNetwork;
    std::unique_ptr<NeuralNetwork> targetNetwork;
    std::unique_ptr<ReplayBuffer> replayBuffer;
    int stateSize;
    int actionSize;
    double learningRate;
    double gamma;
    int batchSize;
    QRandomGenerator rng;

    friend class ChessAI;
};

#endif // DQN_H
