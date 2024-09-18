#ifndef CHESSAI_H
#define CHESSAI_H

#include <QObject>
#include <QRandomGenerator>
#include <QVariant>
#include <QString>
#include <QVector>
#include <QPair>
#include "chessboard.h"
#include "dqn.h"

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
    std::unique_ptr<DQN> dqn;
    QVector<double> getStateRepresentation();
    int actionToMove(int action, int& fromRow, int& fromCol, int& toRow, int& toCol);
    int moveToAction(int fromRow, int fromCol, int toRow, int toCol);
    int evaluateBoard(PieceColor color);

};

#endif // CHESSAI_H
