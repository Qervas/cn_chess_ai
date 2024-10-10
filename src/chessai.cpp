#include "chessai.h"
#include <QDebug>

ChessAI::ChessAI(ChessBoard* board, QObject *parent)
    : QObject(parent), board(board)
{
    int stateSize = 90 * 14; // 90 squares, 14 possible pieces (7 types * 2 colors)
    int actionSize = 90 * 89; // From any square to any other square
    int hiddenSize = 256;
    double learningRate = 0.001;
    double gamma = 0.99;
    int batchSize = 32;
    dqn = std::make_unique<DQN>(stateSize, actionSize, hiddenSize, learningRate, gamma, batchSize);
}

QPair<QPair<int, int>, QPair<int, int>> ChessAI::getAIMove(PieceColor color)
{
    QVector<double> state = getStateRepresentation();
    int maxAttempts = 1; // Maximum number of attempts to find a valid move

    for (int attempt = 0; attempt < maxAttempts; ++attempt) {
        int action = dqn->selectAction(state, 0.1); // 0.1 is the exploration rate, you can adjust this
        int fromRow, fromCol, toRow, toCol;
        actionToMove(action, fromRow, fromCol, toRow, toCol);
        auto validMoves = board->getValidMoves(fromRow, fromCol); //this line takes HALF A SECOND to complete

        // check if the move is valid
        if (board->getPieceAt(fromRow, fromCol).color == color &&
            !validMoves.isEmpty()) {

            bool isValidMove = false;
            for (const auto& move : validMoves) {
                if (move.first == toRow && move.second == toCol) {
                    isValidMove = true;
                    break;
                }
            }

            if (isValidMove) {
                return qMakePair(qMakePair(fromRow, fromCol), qMakePair(toRow, toCol));
            }
        }

        //  If the move is invalid, continue to the next attempt
    }

    // If no legal move is found after multiple attempts, randomly select a legal move
    QVector<QPair<int, int>> allPieces;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (board->getPieceAt(i, j).color == color) {
                allPieces.append(qMakePair(i, j));
            }
        }
    }

    while (!allPieces.isEmpty()) {
        int randomIndex = QRandomGenerator::global()->bounded(allPieces.size());
        QPair<int, int> piece = allPieces[randomIndex];
        QVector<QPair<int, int>> validMoves = board->getValidMoves(piece.first, piece.second);

        if (!validMoves.isEmpty()) {
            int moveIndex = QRandomGenerator::global()->bounded(validMoves.size());
            QPair<int, int> move = validMoves[moveIndex];
            return qMakePair(piece, move);
        }

        allPieces.removeAt(randomIndex);
    }

    // If there are no legal moves, return an invalid move
    return qMakePair(qMakePair(-1, -1), qMakePair(-1, -1));
}

void ChessAI::train(int numEpisodes)
{
    for (int episode = 0; episode < numEpisodes; ++episode) {
        board->reset();
        PieceColor currentPlayer = PieceColor::Red;
        QVector<double> state = getStateRepresentation();

        while (!board->checkGameOver()) {
            int action = dqn->selectAction(state, 0.1);
            int fromRow, fromCol, toRow, toCol;
            actionToMove(action, fromRow, fromCol, toRow, toCol);

            ChessPiece capturedPiece = board->movePiece(fromRow, fromCol, toRow, toCol);
            double reward = evaluateBoard(currentPlayer);

            QVector<double> nextState = getStateRepresentation();
            bool done = board->checkGameOver();

            dqn->train(state, action, reward, nextState, done);

            state = nextState;
            currentPlayer = (currentPlayer == PieceColor::Red) ? PieceColor::Black : PieceColor::Red;
        }

        if (episode % 10 == 0) {
            dqn->updateTargetNetwork();
        }

        qDebug() << "Episode" << episode + 1 << "completed";
    }
}

void ChessAI::saveModel(const QString& filename)
{
    dqn->saveModel(filename);
}

void ChessAI::loadModel(const QString& filename)
{
    dqn->loadModel(filename);
}

QVector<double> ChessAI::getStateRepresentation()
{
    QVector<double> state(90 * 14, 0.0);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 9; ++j) {
            ChessPiece piece = board->getPieceAt(i, j);
            int index = i * 9 + j;
            if (piece.type != PieceType::Empty) {
                int pieceIndex = static_cast<int>(piece.type) - 1;
                if (piece.color == PieceColor::Black) {
                    pieceIndex += 7;
                }
                state[index * 14 + pieceIndex] = 1.0;
            }
        }
    }
    return state;
}

int ChessAI::actionToMove(int action, int& fromRow, int& fromCol, int& toRow, int& toCol)
{
    int from = action / 89;
    int to = action % 89;
    if (to >= from) ++to;
    fromRow = from / 9;
    fromCol = from % 9;
    toRow = to / 9;
    toCol = to % 9;
    return 0;
}

int ChessAI::moveToAction(int fromRow, int fromCol, int toRow, int toCol)
{
    int from = fromRow * 9 + fromCol;
    int to = toRow * 9 + toCol;
    if (to > from) --to;
    return from * 89 + to;
}

int ChessAI::evaluateBoard(PieceColor color)
{
    // todo: simple evaluation function, need to improve
    int score = 0;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 9; ++j) {
            ChessPiece piece = board->getPieceAt(i, j);
            if (piece.color == color) {
                score += static_cast<int>(piece.type);
            } else if (piece.color != PieceColor::None) {
                score -= static_cast<int>(piece.type);
            }
        }
    }
    return score;
}

void ChessAI::startSelfPlay(int numGames)
{
    for (int i = 0; i < numGames; ++i) {
        board->reset();
        QVector<double> state = getStateRepresentation();

        while (!board->checkGameOver()) {
            PieceColor currentPlayer = board->getCurrentPlayer();

            // Select action
            int action = dqn->selectAction(state, 0.1);
            int fromRow, fromCol, toRow, toCol;
            actionToMove(action, fromRow, fromCol, toRow, toCol);

            // Execute move
            ChessPiece capturedPiece = board->movePiece(fromRow, fromCol, toRow, toCol);

            // Calculate reward
            double reward = evaluateBoard(currentPlayer);

            // Get new state
            QVector<double> nextState = getStateRepresentation();

            // Check if game is over
            bool done = board->checkGameOver();

            // Train DQN
            dqn->train(state, action, reward, nextState, done);

            // Update state
            state = nextState;

            // Update target network every certain number of steps
            if (board->getMoveCount() % 100 == 0) {
                dqn->updateTargetNetwork();
            }
        }

        // Game over, emit signal
        PieceColor winner = board->getWinner();
        emit gameCompleted(i + 1, board->getRedScore(), board->getBlackScore());

        // Save model every certain number of games
        if ((i + 1) % 100 == 0) {
            saveModel(QString("model_after_%1_games.bin").arg(i + 1));
        }

        qDebug() << "Game" << i + 1 << "completed. Winner:" << (winner == PieceColor::Red ? "Red" : "Black")
                 << "Red score:" << board->getRedScore() << "Black score:" << board->getBlackScore();
    }
}
