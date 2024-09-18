#include "chessboard.h"
#include <QDebug>

ChessBoard::ChessBoard() : redScore(0), blackScore(0) {
    initializeBoard();
}

void ChessBoard::initializeBoard() {
    board.clear();

    // initialize red pieces
    board[Position(0, 0)] = board[Position(0, 8)] = ChessPiece(PieceType::Chariot, PieceColor::Red);
    board[Position(0, 1)] = board[Position(0, 7)] = ChessPiece(PieceType::Horse, PieceColor::Red);
    board[Position(0, 2)] = board[Position(0, 6)] = ChessPiece(PieceType::Elephant, PieceColor::Red);
    board[Position(0, 3)] = board[Position(0, 5)] = ChessPiece(PieceType::Advisor, PieceColor::Red);
    board[Position(0, 4)] = ChessPiece(PieceType::General, PieceColor::Red);
    board[Position(2, 1)] = board[Position(2, 7)] = ChessPiece(PieceType::Cannon, PieceColor::Red);
    board[Position(3, 0)] = board[Position(3, 2)] = board[Position(3, 4)] = board[Position(3, 6)] = board[Position(3, 8)] = ChessPiece(PieceType::Soldier, PieceColor::Red);

    // initialize black pieces
    board[Position(9, 0)] = board[Position(9, 8)] = ChessPiece(PieceType::Chariot, PieceColor::Black);
    board[Position(9, 1)] = board[Position(9, 7)] = ChessPiece(PieceType::Horse, PieceColor::Black);
    board[Position(9, 2)] = board[Position(9, 6)] = ChessPiece(PieceType::Elephant, PieceColor::Black);
    board[Position(9, 3)] = board[Position(9, 5)] = ChessPiece(PieceType::Advisor, PieceColor::Black);
    board[Position(9, 4)] = ChessPiece(PieceType::General, PieceColor::Black);
    board[Position(7, 1)] = board[Position(7, 7)] = ChessPiece(PieceType::Cannon, PieceColor::Black);
    board[Position(6, 0)] = board[Position(6, 2)] = board[Position(6, 4)] = board[Position(6, 6)] = board[Position(6, 8)] = ChessPiece(PieceType::Soldier, PieceColor::Black);
}

ChessPiece ChessBoard::getPieceAt(int row, int col) const {
    Position pos(row, col);
    return board.contains(pos) ? board[pos] : ChessPiece();
}

ChessPiece ChessBoard::movePiece(int fromRow, int fromCol, int toRow, int toCol) {
    Position fromPos(fromRow, fromCol);
    Position toPos(toRow, toCol);

    // check if the move is valid
    if (!isValidMove(fromRow, fromCol, toRow, toCol)) {
        return ChessPiece(); // return an empty piece to indicate an invalid move
    }

    ChessPiece capturedPiece = getPieceAt(toRow, toCol);
    ChessPiece movingPiece = board[fromPos];

    // execute the move
    board[toPos] = movingPiece;
    board.remove(fromPos);

    if (capturedPiece.type != PieceType::Empty) {
        int score = getPieceScore(capturedPiece.type);
        if (capturedPiece.color == PieceColor::Red) {
            blackScore += score;
        } else {
            redScore += score;
        }
    }

    return capturedPiece;
}

bool ChessBoard::isValidMove(int fromRow, int fromCol, int toRow, int toCol) const {
    if (!isInsideBoard(fromRow, fromCol) || !isInsideBoard(toRow, toCol)) {
        return false;
    }

    Position fromPos(fromRow, fromCol);
    Position toPos(toRow, toCol);

    if (!board.contains(fromPos)) {
        return false;
    }

    const ChessPiece& fromPiece = board[fromPos];
    const ChessPiece& toPiece = board.contains(toPos) ? board[toPos] : ChessPiece();

    if (fromPiece.type == PieceType::Empty) {
        return false;
    }

    if (fromPiece.color == toPiece.color && toPiece.type != PieceType::Empty) {
        return false;
    }

    //  check if the move is valid for the specific piece type
    switch (fromPiece.type) {
        case PieceType::General: return isValidGeneralMove(fromRow, fromCol, toRow, toCol);
        case PieceType::Advisor: return isValidAdvisorMove(fromRow, fromCol, toRow, toCol);
        case PieceType::Elephant: return isValidElephantMove(fromRow, fromCol, toRow, toCol);
        case PieceType::Horse: return isValidHorseMove(fromRow, fromCol, toRow, toCol);
        case PieceType::Chariot: return isValidChariotMove(fromRow, fromCol, toRow, toCol);
        case PieceType::Cannon: return isValidCannonMove(fromRow, fromCol, toRow, toCol);
        case PieceType::Soldier: return isValidSoldierMove(fromRow, fromCol, toRow, toCol);
        default: return false;
    }
}

bool ChessBoard::isInsideBoard(int row, int col) const {
    return row >= 0 && row < 10 && col >= 0 && col < 9;
}

bool ChessBoard::isValidGeneralMove(int fromRow, int fromCol, int toRow, int toCol) const {
    const ChessPiece& fromPiece = board[Position(fromRow, fromCol)];
    int rowDiff = abs(toRow - fromRow);
    int colDiff = abs(toCol - fromCol);

    bool inPalace = (fromPiece.color == PieceColor::Red)
        ? isInRedPalace(toRow, toCol)
        : isInBlackPalace(toRow, toCol);

    return inPalace && ((rowDiff == 1 && colDiff == 0) || (rowDiff == 0 && colDiff == 1));
}

bool ChessBoard::isValidAdvisorMove(int fromRow, int fromCol, int toRow, int toCol) const {
    const ChessPiece& fromPiece = board[Position(fromRow, fromCol)];
    int rowDiff = abs(toRow - fromRow);
    int colDiff = abs(toCol - fromCol);

    // The Advisor (or Minister) can only move diagonally within the palace.
    bool inPalace = (fromPiece.color == PieceColor::Red)
        ? (toRow >= 0 && toRow <= 2 && toCol >= 3 && toCol <= 5)
        : (toRow >= 7 && toRow <= 9 && toCol >= 3 && toCol <= 5);

    return inPalace && rowDiff == 1 && colDiff == 1;
}

bool ChessBoard::isValidElephantMove(int fromRow, int fromCol, int toRow, int toCol) const {
    const ChessPiece& fromPiece = board[Position(fromRow, fromCol)];
    int rowDiff = abs(toRow - fromRow);
    int colDiff = abs(toCol - fromCol);

    // The Elephant/Bishop can only move within its own territory.
    bool inOwnSide = (fromPiece.color == PieceColor::Red)
        ? (toRow >= 0 && toRow <= 4)
        : (toRow >= 5 && toRow <= 9);

    // Check if the Elephant's eye is blocked.
    int eyeRow = (fromRow + toRow) / 2;
    int eyeCol = (fromCol + toCol) / 2;

    return inOwnSide && rowDiff == 2 && colDiff == 2 && board[Position(eyeRow, eyeCol)].type == PieceType::Empty;
}

bool ChessBoard::isValidHorseMove(int fromRow, int fromCol, int toRow, int toCol) const {
    int rowDiff = abs(toRow - fromRow);
    int colDiff = abs(toCol - fromCol);

    if ((rowDiff == 2 && colDiff == 1) || (rowDiff == 1 && colDiff == 2)) {
        // check the horse's leg is not blocked
        int legRow = (rowDiff == 2) ? (fromRow + (toRow - fromRow) / 2) : fromRow;
        int legCol = (colDiff == 2) ? (fromCol + (toCol - fromCol) / 2) : fromCol;
        return board[Position(legRow, legCol)].type == PieceType::Empty;
    }
    return false;
}

bool ChessBoard::isValidChariotMove(int fromRow, int fromCol, int toRow, int toCol) const {
    if (fromRow != toRow && fromCol != toCol) return false;

    int start, end;
    if (fromRow == toRow) {
        start = std::min(fromCol, toCol) + 1;
        end = std::max(fromCol, toCol);
        for (int col = start; col < end; ++col) {
            if (board[Position(fromRow, col)].type != PieceType::Empty) return false;
        }
    } else {
        start = std::min(fromRow, toRow) + 1;
        end = std::max(fromRow, toRow);
        for (int row = start; row < end; ++row) {
            if (board[Position(row, fromCol)].type != PieceType::Empty) return false;
        }
    }
    return true;
}

bool ChessBoard::isValidCannonMove(int fromRow, int fromCol, int toRow, int toCol) const {
    if (fromRow != toRow && fromCol != toCol) return false;

    int count = 0;
    int start, end;
    if (fromRow == toRow) {
        start = std::min(fromCol, toCol) + 1;
        end = std::max(fromCol, toCol);
        for (int col = start; col < end; ++col) {
            if (board[Position(fromRow, col)].type != PieceType::Empty) ++count;
        }
    } else {
        start = std::min(fromRow, toRow) + 1;
        end = std::max(fromRow, toRow);
        for (int row = start; row < end; ++row) {
            if (board[Position(row, fromCol)].type != PieceType::Empty) ++count;
        }
    }

    if (board[Position(toRow, toCol)].type == PieceType::Empty) {
        return count == 0;
    } else {
        return count == 1;
    }
}

bool ChessBoard::isValidSoldierMove(int fromRow, int fromCol, int toRow, int toCol) const {
    const ChessPiece& fromPiece = board[Position(fromRow, fromCol)];
    int rowDiff = toRow - fromRow;
    int colDiff = abs(toCol - fromCol);

    if (fromPiece.color == PieceColor::Red) {
        if (fromRow < 5) { // Not crossed the river
            return rowDiff == 1 && colDiff == 0;
        } else { // Crossed the river
            return (rowDiff == 1 && colDiff == 0) || (rowDiff == 0 && colDiff == 1);
        }
    } else { // Black
        if (fromRow > 4) { // Not crossed the river
            return rowDiff == -1 && colDiff == 0;
        } else { // Crossed the river
            return (rowDiff == -1 && colDiff == 0) || (rowDiff == 0 && colDiff == 1);
        }
    }
}

bool ChessBoard::checkGameOver() const
{
    bool redGeneralAlive = false;
    bool blackGeneralAlive = false;

    // 检查将帅是否还在棋盘上
    for (auto it = board.constBegin(); it != board.constEnd(); ++it) {
        const ChessPiece& piece = it.value();
        if (piece.type == PieceType::General) {
            if (piece.color == PieceColor::Red) {
                redGeneralAlive = true;
            } else {
                blackGeneralAlive = true;
            }
        }
        if (redGeneralAlive && blackGeneralAlive) {
            break;
        }
    }

    // If either player's General (King) is not on the board, the game ends.
    return !redGeneralAlive || !blackGeneralAlive;
}

PieceColor ChessBoard::getWinner() const
{
    bool redGeneralAlive = false;
    bool blackGeneralAlive = false;

    for (auto it = board.constBegin(); it != board.constEnd(); ++it) {
        const ChessPiece& piece = it.value();
        if (piece.type == PieceType::General) {
            if (piece.color == PieceColor::Red) {
                redGeneralAlive = true;
            } else {
                blackGeneralAlive = true;
            }
        }
        if (redGeneralAlive && blackGeneralAlive) {
            break;
        }
    }

    if (!redGeneralAlive) {
        return PieceColor::Black;
    } else if (!blackGeneralAlive) {
        return PieceColor::Red;
    } else {
        return PieceColor::None; // game is not over yet
    }
}

bool ChessBoard::isGeneralAlive(PieceColor color) const {
    for (auto it = board.begin(); it != board.end(); ++it) {
        if (it.value().type == PieceType::General && it.value().color == color) {
            return true;
        }
    }
    return false;
}

void ChessBoard::reset() {
    // clear the board
    board.clear();
    redScore = 0;
    blackScore = 0;
    // re-initialize the board
    initializeBoard();
}

int getPieceScore(PieceType type) {
    switch (type) {
        case PieceType::General: return static_cast<int>(PieceScore::General);
        case PieceType::Advisor: return static_cast<int>(PieceScore::Advisor);
        case PieceType::Elephant: return static_cast<int>(PieceScore::Elephant);
        case PieceType::Horse: return static_cast<int>(PieceScore::Horse);
        case PieceType::Chariot: return static_cast<int>(PieceScore::Chariot);
        case PieceType::Cannon: return static_cast<int>(PieceScore::Cannon);
        case PieceType::Soldier: return static_cast<int>(PieceScore::Soldier);
        default: return 0;
    }
}

int ChessBoard::getRedScore() const {
    return redScore;
}

int ChessBoard::getBlackScore() const {
    return blackScore;
}

QVector<QPair<int, int>> ChessBoard::getValidMoves(int row, int col) const
{
    QVector<QPair<int, int>> validMoves;

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (isValidMove(row, col, i, j)) {
                validMoves.append(qMakePair(i, j));
            }
        }
    }

    return validMoves;
}
