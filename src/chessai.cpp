#include "chessai.h"
#include <QDebug>
#include <cstdlib>

ChessAI::ChessAI(ChessBoard* board, QObject *parent)
    : QObject(parent), board(board)
{
    // Reduce state size by using a more compact representation
    int stateSize = 90 * 7; // 90 squares, 7 piece types (ignore color for now)

    // Reduce action size by encoding moves more efficiently
    int actionSize = 90 * 4; // 90 source squares, 4 directions (up, down, left, right)

    // Reduce hidden layer size
    int hiddenSize = 128;

    // Create the network with smaller dimensions
    dqn = std::make_unique<DQN>(std::vector<int>{stateSize, hiddenSize, actionSize});
}

QPair<QPair<int, int>, QPair<int, int>> ChessAI::getAIMove(PieceColor color)
{
    std::vector<double> state = getStateRepresentation();
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
        std::vector<double> state = getStateRepresentation();

        while (!board->checkGameOver()) {
            int action = dqn->selectAction(state, 0.1);
            int fromRow, fromCol, toRow, toCol;
            actionToMove(action, fromRow, fromCol, toRow, toCol);

            ChessPiece capturedPiece = board->movePiece(fromRow, fromCol, toRow, toCol);
            double reward = evaluateBoard(currentPlayer);

            std::vector<double> targetQ = dqn->getQValues(state);
            if (capturedPiece.type != PieceType::Empty) {
                targetQ[action] = reward;
            } else {
                std::vector<double> nextQ = dqn->getQValues(getStateRepresentation());
                targetQ[action] = reward + gamma * *std::max_element(nextQ.begin(), nextQ.end());
            }

            dqn->backpropagate(state, targetQ, learningRate);

            state = getStateRepresentation();
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
    dqn->saveModel(filename.toStdString());
}

void ChessAI::loadModel(const QString& filename)
{
    dqn->loadModel(filename.toStdString());
}

void ChessAI::startSelfPlay(int numGames)
{
    for (int i = 0; i < numGames; ++i) {
        board->reset();
        std::vector<double> state = getStateRepresentation();

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
            std::vector<double> nextState = getStateRepresentation();

            // Check if game is over
            bool done = board->checkGameOver();

            // Train DQN
            dqn->backpropagate(state, std::vector<double>{}, learningRate); // Adjust as needed

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

std::vector<double> ChessAI::getStateRepresentation()
{
    std::vector<double> state;
    state.reserve(90 * 14); // 90 squares, 14 possible pieces (7 types * 2 colors)

    for (int row = 0; row < 10; ++row) {
        for (int col = 0; col < 9; ++col) {
            ChessPiece piece = board->getPieceAt(row, col);
            std::vector<double> pieceEncoding(14, 0.0);
            if (piece.type != PieceType::Empty) {
                int index = static_cast<int>(piece.type) - 1;
                if (piece.color == PieceColor::Black) {
                    index += 7;
                }
                pieceEncoding[index] = 1.0;
            }
            state.insert(state.end(), pieceEncoding.begin(), pieceEncoding.end());
        }
    }

    return state;
}

int ChessAI::actionToMove(int action, int& fromRow, int& fromCol, int& toRow, int& toCol)
{
    int fromSquare = action / 89;
    int toSquare = action % 89;

    fromRow = fromSquare / 9;
    fromCol = fromSquare % 9;
    toRow = toSquare / 9;
    toCol = toSquare % 9;

    return action;
}

int ChessAI::moveToAction(int fromRow, int fromCol, int toRow, int toCol)
{
    int fromSquare = fromRow * 9 + fromCol;
    int toSquare = toRow * 9 + toCol;
    return fromSquare * 89 + toSquare;
}

int ChessAI::evaluateBoard(PieceColor color)
{
    int score = 0;
    for (int row = 0; row < 10; ++row) {
        for (int col = 0; col < 9; ++col) {
            ChessPiece piece = board->getPieceAt(row, col);
            if (piece.color == color) {
                switch (piece.type) {
                    case PieceType::General: score += 1000; break;
                    case PieceType::Advisor: score += 20; break;
                    case PieceType::Elephant: score += 20; break;
                    case PieceType::Horse: score += 40; break;
                    case PieceType::Chariot: score += 90; break;
                    case PieceType::Cannon: score += 45; break;
                    case PieceType::Soldier: score += 10; break;
                    default: break;
                }
            } else if (piece.color != PieceColor::None) {
                switch (piece.type) {
                    case PieceType::General: score -= 1000; break;
                    case PieceType::Advisor: score -= 20; break;
                    case PieceType::Elephant: score -= 20; break;
                    case PieceType::Horse: score -= 40; break;
                    case PieceType::Chariot: score -= 90; break;
                    case PieceType::Cannon: score -= 45; break;
                    case PieceType::Soldier: score -= 10; break;
                    default: break;
                }
            }
        }
    }
    return score;
}
