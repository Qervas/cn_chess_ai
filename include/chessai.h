#ifndef CHESSAI_H
#define CHESSAI_H

#include <QObject>
#include <QRandomGenerator>
#include <QVariant>
#include <QString>
#include <QPair>
#include <vector>
#include <QFile>
#include <QTextStream>
#include "chessboard.h"
#include "dqn.h"
#include "action.h"

class ChessAI : public QObject
{
    Q_OBJECT

public:
    explicit ChessAI(ChessBoard* board);
    ~ChessAI();
    QPair<QPair<int, int>, QPair<int, int>> getAIMove(PieceColor color);
    void saveModel(const QString& filename);
    void loadModel(const QString& filename);
	void onGameCompleted(int gameNumber, int redScore, int blackScore);
	void initializeDQN();
	bool isDQNInitialized() const{ return dqn != nullptr;}

public slots:
    void train(int numEpisodes);
    void startSelfPlay(int numGames);

signals:
    void gameCompleted(int gameNumber, int redScore, int blackScore);
    void trainingFinished();
    void selfPlayFinished();

private:
    ChessBoard* board;
    std::unique_ptr<DQN> dqn; 
    std::vector<double> getStateRepresentation(); 
    Action actionToMove(const Action& action);
    Action moveToAction(int fromRow, int fromCol, int toRow, int toCol);
    int evaluateBoard(PieceColor color, int moveCount);
    
    // Learning parameters
    double learningRate = 0.001;
    double gamma = 0.99;

    // New method to gather all valid actions for the current player
    std::vector<Action> getAllValidActions(PieceColor player) const;

    QFile logFile;
    int numGames; 
};

#endif // CHESSAI_H
