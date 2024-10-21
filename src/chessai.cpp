#include "chessai.h"
#include "action.h"
#include <QDebug>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <QFile>
#include <QTextStream>

ChessAI::ChessAI(ChessBoard* board)
    : board(board)
{
     std::srand(static_cast<unsigned int>(std::time(nullptr)));


    // Open the log file
    logFile.setFileName("game_log.txt");
    if (!logFile.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
        qWarning() << "Failed to open log file:" << logFile.errorString();
    }
}

ChessAI::~ChessAI() {
    if (logFile.isOpen()) {
        logFile.close();
    }
}

QPair<QPair<int, int>, QPair<int, int>> ChessAI::getAIMove(PieceColor color)
{
    std::vector<double> state = getStateRepresentation();
    int maxAttempts = 10; // Increased attempts for better move selection

    for (int attempt = 0; attempt < maxAttempts; ++attempt) {
        // Gather all valid actions for the current player
        std::vector<Action> validActions = getAllValidActions(color);

        if(validActions.empty()) {
            continue; // No valid actions, try again
        }

        // Select an action using DQN with the list of valid actions
        Action selectedAction = dqn->selectAction(state, 0.1, validActions);

        int fromRow = selectedAction.from / 9;
        int fromCol = selectedAction.from % 9;
        int toRow = selectedAction.to / 9;
        int toCol = selectedAction.to % 9;

        QVector<QPair<int, int>> validMoves = board->getValidMoves(fromRow, fromCol);

        // Check if the move is valid
        if (board->getPieceAt(fromRow, fromCol).color == color && !validMoves.isEmpty()) {
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
    }

    // If no valid move found after maxAttempts, select a random valid move
    std::vector<Action> validActions = getAllValidActions(color);
    if(validActions.empty()) {
        // No valid actions available
        return qMakePair(qMakePair(-1, -1), qMakePair(-1, -1));
    }

    int randomIndex = QRandomGenerator::global()->bounded(static_cast<int>(validActions.size()));
    Action randomAction = validActions[randomIndex];
    int fromRow = randomAction.from / 9;
    int fromCol = randomAction.from % 9;
    int toRow = randomAction.to / 9;
    int toCol = randomAction.to % 9;

    return qMakePair(qMakePair(fromRow, fromCol), qMakePair(toRow, toCol));
}

void ChessAI::train(int numEpisodes)
{
	initializeDQN();
    for (int episode = 0; episode < numEpisodes; ++episode) {
        board->reset();
        PieceColor currentPlayer = PieceColor::Red;
        std::vector<double> state = getStateRepresentation();

        while (!board->checkGameOver()) {
            // Gather all valid actions for the current state and player
            std::vector<Action> validActions = getAllValidActions(currentPlayer);

            if (validActions.empty()) {
                // No valid actions available, skip to game over
                break;
            }

            // Select an action using DQN with the list of valid actions
            Action selectedAction = dqn->selectAction(state, 0.1, validActions);

            int fromRow = selectedAction.from / 9;
            int fromCol = selectedAction.from % 9;
            int toRow = selectedAction.to / 9;
            int toCol = selectedAction.to % 9;

            board->movePiece(fromRow, fromCol, toRow, toCol);

            double reward = evaluateBoard(currentPlayer);

            std::vector<double> nextState = getStateRepresentation();
            bool done = board->checkGameOver();
           
            // Compute target Q-values
            std::vector<double> targetQ = dqn->getQValues(state);
            if (done) {
                targetQ[selectedAction.to] = reward;
            } else {
                std::vector<double> nextQ = dqn->getQValues(nextState);
                targetQ[selectedAction.to] = reward + gamma * *std::max_element(nextQ.begin(), nextQ.end());
            }

            // Perform backpropagation
            dqn->backpropagate(state, targetQ, learningRate);

            // Update state
            state = nextState;

            // Switch player
            currentPlayer = (currentPlayer == PieceColor::Red) ? PieceColor::Black : PieceColor::Red;

            // Update target network every certain number of steps
            if (board->getMoveCount() % 100 == 0) {
                dqn->updateTargetNetwork();
            }
        }

        // Game over, emit signal
        PieceColor winner = board->getWinner();
        emit gameCompleted(episode + 1, board->getRedScore(), board->getBlackScore());

        // Save model every certain number of games
        if ((episode + 1) % 100 == 0) {
            saveModel(QString("model_after_%1_games.bin").arg(episode + 1));
        }

        qDebug() << "Game" << episode + 1 << "completed. Winner:" 
                 << (winner == PieceColor::Red ? "Red" : "Black")
                 << "Red score:" << board->getRedScore() 
                 << "Black score:" << board->getBlackScore();
    }
    emit trainingFinished();
}

void ChessAI::saveModel(const QString& filename)
{
    if (dqn) {
        dqn->saveModel(filename.toStdString());
    } else {
        qWarning() << "DQN is not initialized. Cannot save model.";
    }
}

void ChessAI::loadModel(const QString& filename)
{
    if (dqn) {
        dqn->loadModel(filename.toStdString());
    } else {
        qWarning() << "DQN is not initialized. Cannot load model.";
    }
}

void ChessAI::startSelfPlay(int numGames)
{
    for (int i = 0; i < numGames; ++i) {
        board->reset();
        std::vector<double> state = getStateRepresentation();

        while (!board->checkGameOver()) {
            PieceColor currentPlayer = board->getCurrentPlayer();

            // Gather all valid actions for the current player
            std::vector<Action> validActions = getAllValidActions(currentPlayer);

            if (validActions.empty()) {
                // No valid actions available, skip to game over
                break;
            }

            // Select an action using DQN with the list of valid actions
            Action selectedAction = dqn->selectAction(state, 0.1, validActions);

            int fromRow = selectedAction.from / 9;
            int fromCol = selectedAction.from % 9;
            int toRow = selectedAction.to / 9;
            int toCol = selectedAction.to % 9;

            // Execute move
            ChessPiece capturedPiece = board->movePiece(fromRow, fromCol, toRow, toCol);

            // Calculate reward
            double reward = evaluateBoard(currentPlayer);

            // Get new state
            std::vector<double> nextState = getStateRepresentation();

            // Check if game is over
            bool done = board->checkGameOver();

            // Compute target Q-values
            std::vector<double> targetQ = dqn->getQValues(state);
            if (done) {
                targetQ[selectedAction.to] = reward;
            } else {
                std::vector<double> nextQ = dqn->getQValues(nextState);
                targetQ[selectedAction.to] = reward + gamma * *std::max_element(nextQ.begin(), nextQ.end());
            }

            // Train DQN
            dqn->backpropagate(state, targetQ, learningRate);

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

        qDebug() << "Game" << i + 1 << "completed. Winner:" 
                 << (winner == PieceColor::Red ? "Red" : "Black")
                 << "Red score:" << board->getRedScore() 
                 << "Black score:" << board->getBlackScore();
    }

    emit selfPlayFinished();
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

Action ChessAI::actionToMove(const Action& action)
{
    int fromSquare = action.from;
    int toSquare = action.to;

    int fromRow = fromSquare / 9;
    int fromCol = fromSquare % 9;
    int toRow = toSquare / 9;
    int toCol = toSquare % 9;

    return Action{fromRow * 9 + fromCol, toRow * 9 + toCol};
}

Action ChessAI::moveToAction(int fromRow, int fromCol, int toRow, int toCol)
{
    int fromSquare = fromRow * 9 + fromCol;
    int toSquare = toRow * 9 + toCol;
    return Action{fromSquare, toSquare};
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

std::vector<Action> ChessAI::getAllValidActions(PieceColor player) const
{
    std::vector<Action> validActions;

    for(int row = 0; row < 10; ++row) {
        for(int col = 0; col < 9; ++col) {
            ChessPiece piece = board->getPieceAt(row, col);
            if(piece.color == player) {
                QVector<QPair<int, int>> moves = board->getValidMoves(row, col);
                for(const auto& move : moves) {
                    int toRow = move.first;
                    int toCol = move.second;
                    int fromSquare = row * 9 + col;
                    int toSquare = toRow * 9 + toCol;
                    validActions.emplace_back(Action{fromSquare, toSquare});
                }
            }
        }
    }

    return validActions;
}

void ChessAI::onGameCompleted(int gameNumber, int redScore, int blackScore) {
    if (!logFile.isOpen()) {
        qWarning() << "Log file is not open";
        return;
    }

    QString result = (redScore > blackScore) ? "Red wins!" : (blackScore > redScore) ? "Black wins!" : "It's a draw!";

    QTextStream out(&logFile);
    out << QString("Game %1 completed. Red Score: %2, Black Score: %3. %4\n")
           .arg(gameNumber).arg(redScore).arg(blackScore).arg(result);

    // If it is the last game, you can add summary information here
    if (gameNumber == numGames) {
        out << QString("AI self-play session completed. Total games: %1\n\n").arg(numGames);
    }

    // Ensure the data is written to the file
    logFile.flush();

    // Optional: Output the result to the console
    qDebug() << QString("Game %1 completed. Red Score: %2, Black Score: %3. %4")
                .arg(gameNumber).arg(redScore).arg(blackScore).arg(result);
}

void ChessAI::initializeDQN()
{
    if (!dqn) {
        // Define state and action sizes based on the new representation
        int stateSize = 90 * 14; // 90 squares, 14 possible pieces (7 types * 2 colors)
        int actionSize = 90 * 90; // From any square to any square

        dqn = std::make_unique<DQN>(std::vector<int>{stateSize, 128, actionSize});
    }
}