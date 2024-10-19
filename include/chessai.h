#ifndef CHESSAI_H
#define CHESSAI_H

#include <QObject>
#include <QRandomGenerator>
#include <QVariant>
#include <QString>
#include <QPair>
#include <vector>
#include "chessboard.h"
#include "dqn.h" // Include DQN definition

class ChessAI : public QObject
{
    Q_OBJECT

public:
    explicit ChessAI(ChessBoard* board, QObject *parent = nullptr);
    QPair<QPair<int, int>, QPair<int, int>> getAIMove(PieceColor color);
    void train(int numEpisodes);
    void saveModel(const QString& filename);
    void loadModel(const QString& filename);
    void startSelfPlay(int numGames);

signals:
    void gameCompleted(int gameNumber, int redScore, int blackScore);

private:
    ChessBoard* board;
    std::unique_ptr<DQN> dqn; // Ensure DQN uses the CUDA-enabled NeuralNetwork
    std::vector<double> getStateRepresentation(); // Changed to std::vector<double>
    int actionToMove(int action, int& fromRow, int& fromCol, int& toRow, int& toCol);
    int moveToAction(int fromRow, int fromCol, int toRow, int toCol);
    int evaluateBoard(PieceColor color);
    
    // Learning parameters
    double learningRate = 0.001;
    double gamma = 0.99;
};

#endif // CHESSAI_H
