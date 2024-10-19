#include "chessboard.h"
#include <QDebug>

ChessBoard::ChessBoard() : redScore(0), blackScore(0) {
    initializeBoard();
}

void ChessBoard::initializeBoard() {
    board.resize(ROWS * COLS);
    std::fill(board.begin(), board.end(), ChessPiece());

    // Initialize red pieces
    board[0 * COLS + 0] = board[0 * COLS + 8] = ChessPiece(PieceType::Chariot, PieceColor::Red);
    board[0 * COLS + 1] = board[0 * COLS + 7] = ChessPiece(PieceType::Horse, PieceColor::Red);
    board[0 * COLS + 2] = board[0 * COLS + 6] = ChessPiece(PieceType::Elephant, PieceColor::Red);
    board[0 * COLS + 3] = board[0 * COLS + 5] = ChessPiece(PieceType::Advisor, PieceColor::Red);
    board[0 * COLS + 4] = ChessPiece(PieceType::General, PieceColor::Red);
    board[2 * COLS + 1] = board[2 * COLS + 7] = ChessPiece(PieceType::Cannon, PieceColor::Red);
    board[3 * COLS + 0] = board[3 * COLS + 2] = board[3 * COLS + 4] = board[3 * COLS + 6] = board[3 * COLS + 8] = ChessPiece(PieceType::Soldier, PieceColor::Red);

    // Initialize black pieces
    board[9 * COLS + 0] = board[9 * COLS + 8] = ChessPiece(PieceType::Chariot, PieceColor::Black);
    board[9 * COLS + 1] = board[9 * COLS + 7] = ChessPiece(PieceType::Horse, PieceColor::Black);
    board[9 * COLS + 2] = board[9 * COLS + 6] = ChessPiece(PieceType::Elephant, PieceColor::Black);
    board[9 * COLS + 3] = board[9 * COLS + 5] = ChessPiece(PieceType::Advisor, PieceColor::Black);
    board[9 * COLS + 4] = ChessPiece(PieceType::General, PieceColor::Black);
    board[7 * COLS + 1] = board[7 * COLS + 7] = ChessPiece(PieceType::Cannon, PieceColor::Black);
    board[6 * COLS + 0] = board[6 * COLS + 2] = board[6 * COLS + 4] = board[6 * COLS + 6] = board[6 * COLS + 8] = ChessPiece(PieceType::Soldier, PieceColor::Black);
}

ChessPiece ChessBoard::getPieceAt(int row, int col) const {
    if (!isInsideBoard(row, col)) {
        return ChessPiece();
    }
    return board[row * COLS + col];
}

ChessPiece ChessBoard::movePiece(int fromRow, int fromCol, int toRow, int toCol) {
    if (!isValidMove(fromRow, fromCol, toRow, toCol)) {
        return ChessPiece(); // Invalid move
    }

    ChessPiece& movingPiece = board[fromRow * COLS + fromCol];
    ChessPiece capturedPiece = board[toRow * COLS + toCol];

    // Execute the move
    board[toRow * COLS + toCol] = movingPiece;
    board[fromRow * COLS + fromCol] = ChessPiece(); // Empty the original square

    // Update scores
    if (capturedPiece.type != PieceType::Empty) {
        int score = getPieceScore(capturedPiece.type);
        if (capturedPiece.color == PieceColor::Red) {
            blackScore += score;
        } else {
            redScore += score;
        }
    }

    moveCount++;
    currentPlayer = (currentPlayer == PieceColor::Red) ? PieceColor::Black : PieceColor::Red;

    return capturedPiece;
}

bool ChessBoard::isValidMove(int fromRow, int fromCol, int toRow, int toCol) const {
    if (!isInsideBoard(fromRow, fromCol) || !isInsideBoard(toRow, toCol)) {
        return false;
    }

    const ChessPiece& fromPiece = board[fromRow * COLS + fromCol];
    const ChessPiece& toPiece = board[toRow * COLS + toCol];

    if (fromPiece.type == PieceType::Empty) {
        return false;
    }

    if (fromPiece.color == toPiece.color && toPiece.type != PieceType::Empty) {
        return false;
    }

    // Check if the move is valid for the specific piece type
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

void ChessBoard::reset() {
    board.fill(ChessPiece());
    redScore = 0;
    blackScore = 0;
    moveCount = 0;
    currentPlayer = PieceColor::Red;
    initializeBoard();
}

int ChessBoard::getRedScore() const {
    return redScore;
}

int ChessBoard::getBlackScore() const {
    return blackScore;
}

QVector<QPair<int, int>> ChessBoard::getValidMoves(int row, int col) const {
    QVector<QPair<int, int>> validMoves;
    ChessPiece piece = getPieceAt(row, col);

    if (piece.type == PieceType::Empty || piece.color == PieceColor::None) {
        return validMoves; // No moves available
    }

    switch (piece.type) {
        case PieceType::General:
            generateGeneralMoves(row, col, validMoves);
            break;
        case PieceType::Advisor:
            generateAdvisorMoves(row, col, validMoves);
            break;
        case PieceType::Elephant:
            generateElephantMoves(row, col, validMoves);
            break;
        case PieceType::Horse:
            generateHorseMoves(row, col, validMoves);
            break;
        case PieceType::Chariot:
            generateChariotMoves(row, col, validMoves);
            break;
        case PieceType::Cannon:
            generateCannonMoves(row, col, validMoves);
            break;
        case PieceType::Soldier:
            generateSoldierMoves(row, col, validMoves);
            break;
        default:
            break;
    }

    return validMoves;
}

void ChessBoard::generateGeneralMoves(int row, int col, QVector<QPair<int, int>>& moves) const {
    QVector<QPair<int, int>> directions = { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };

    for (const auto& dir : directions) {
        int newRow = row + dir.first;
        int newCol = col + dir.second;

        if (isInsideBoard(newRow, newCol) && isValidMove(row, col, newRow, newCol)) {
            moves.append(qMakePair(newRow, newCol));
        }
    }
}

void ChessBoard::generateAdvisorMoves(int row, int col, QVector<QPair<int, int>>& moves) const {
    QVector<QPair<int, int>> directions = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    PieceColor color = getPieceAt(row, col).color;

    for (const auto& dir : directions) {
        int newRow = row + dir.first;
        int newCol = col + dir.second;

        if (isInsideBoard(newRow, newCol) && 
            ((color == PieceColor::Red && isInRedPalace(newRow, newCol)) ||
             (color == PieceColor::Black && isInBlackPalace(newRow, newCol))) &&
            isValidMove(row, col, newRow, newCol)) {
            moves.append(qMakePair(newRow, newCol));
        }
    }
}

void ChessBoard::generateElephantMoves(int row, int col, QVector<QPair<int, int>>& moves) const {
    QVector<QPair<int, int>> directions = {{2, 2}, {2, -2}, {-2, 2}, {-2, -2}};
    PieceColor color = getPieceAt(row, col).color;

    for (const auto& dir : directions) {
        int newRow = row + dir.first;
        int newCol = col + dir.second;
        int midRow = row + dir.first / 2;
        int midCol = col + dir.second / 2;

        if (isInsideBoard(newRow, newCol) && 
            isInOwnSide(color, newRow) &&
            getPieceAt(midRow, midCol).type == PieceType::Empty &&
            isValidMove(row, col, newRow, newCol)) {
            moves.append(qMakePair(newRow, newCol));
        }
    }
}

void ChessBoard::generateChariotMoves(int row, int col, QVector<QPair<int, int>>& moves) const {
    QVector<QPair<int, int>> directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    for (const auto& dir : directions) {
        int newRow = row + dir.first;
        int newCol = col + dir.second;

        while (isInsideBoard(newRow, newCol)) {
            if (isValidMove(row, col, newRow, newCol)) {
                moves.append(qMakePair(newRow, newCol));
                if (getPieceAt(newRow, newCol).type != PieceType::Empty) {
                    break;  // Stop at the first piece encountered
                }
            } else {
                break;
            }
            newRow += dir.first;
            newCol += dir.second;
        }
    }
}

void ChessBoard::generateCannonMoves(int row, int col, QVector<QPair<int, int>>& moves) const {
    QVector<QPair<int, int>> directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    for (const auto& dir : directions) {
        int newRow = row + dir.first;
        int newCol = col + dir.second;
        bool foundScreen = false;

        while (isInsideBoard(newRow, newCol)) {
            if (!foundScreen) {
                if (getPieceAt(newRow, newCol).type == PieceType::Empty) {
                    moves.append(qMakePair(newRow, newCol));
                } else {
                    foundScreen = true;
                }
            } else {
                if (getPieceAt(newRow, newCol).type != PieceType::Empty &&
                    isValidMove(row, col, newRow, newCol)) {
                    moves.append(qMakePair(newRow, newCol));
                    break;
                }
            }
            newRow += dir.first;
            newCol += dir.second;
        }
    }
}

void ChessBoard::generateHorseMoves(int row, int col, QVector<QPair<int, int>>& moves) const {
    QVector<QPair<int, int>> directions = {{1, 2}, {1, -2}, {-1, 2}, {-1, -2}, {2, 1}, {2, -1}, {-2, 1}, {-2, -1}};

    for (const auto& dir : directions) {
        int newRow = row + dir.first;
        int newCol = col + dir.second;
        int legRow = row + (dir.first / 2);
        int legCol = col + (dir.second / 2);

        if (isInsideBoard(newRow, newCol) &&
            getPieceAt(legRow, legCol).type == PieceType::Empty &&
            isValidMove(row, col, newRow, newCol)) {
            moves.append(qMakePair(newRow, newCol));
        }
    }
}

void ChessBoard::generateSoldierMoves(int row, int col, QVector<QPair<int, int>>& moves) const {
    PieceColor color = getPieceAt(row, col).color;
    int forward = (color == PieceColor::Red) ? 1 : -1;

    // Forward move
    int newRow = row + forward;
    if (isInsideBoard(newRow, col) && isValidMove(row, col, newRow, col)) {
        moves.append(qMakePair(newRow, col));
    }

    // Sideways moves (only if the soldier has crossed the river)
    if ((color == PieceColor::Red && row > 4) || (color == PieceColor::Black && row < 5)) {
        for (int newCol : {col - 1, col + 1}) {
            if (isInsideBoard(row, newCol) && isValidMove(row, col, row, newCol)) {
                moves.append(qMakePair(row, newCol));
            }
        }
    }
}

// Check if the game is over
bool ChessBoard::checkGameOver() const {
    bool redGeneralAlive = false;
    bool blackGeneralAlive = false;

    for (int i = 0; i < ROWS * COLS; ++i) {
        const ChessPiece& piece = board[i];
        if (piece.type == PieceType::General) {
            if (piece.color == PieceColor::Red) {
                redGeneralAlive = true;
            } else if (piece.color == PieceColor::Black) {
                blackGeneralAlive = true;
            }
        }
        if (redGeneralAlive && blackGeneralAlive) {
            return false; // Both generals are alive, game is not over
        }
    }

    return true; // One of the generals is not found, game is over
}

// Get the winner of the game
PieceColor ChessBoard::getWinner() const {
    for (int i = 0; i < ROWS * COLS; ++i) {
        const ChessPiece& piece = board[i];
        if (piece.type == PieceType::General) {
            return piece.color;
        }
    }
    return PieceColor::None; // This should never happen in a valid game state
}

// Check if a position is inside the board
bool ChessBoard::isInsideBoard(int row, int col) const {
    return row >= 0 && row < ROWS && col >= 0 && col < COLS;
}

// Check if a move is valid for the General
bool ChessBoard::isValidGeneralMove(int fromRow, int fromCol, int toRow, int toCol) const {
    // Check if the move is within the palace
    bool isInPalace = (fromRow >= 0 && fromRow <= 2 && fromCol >= 3 && fromCol <= 5) ||
                      (fromRow >= 7 && fromRow <= 9 && fromCol >= 3 && fromCol <= 5);
    if (!isInPalace) {
        return false;
    }

    // Check if the move is only one step in any direction
    int rowDiff = std::abs(toRow - fromRow);
    int colDiff = std::abs(toCol - fromCol);
    return (rowDiff + colDiff == 1);
}

// Implement other piece-specific move validation functions similarly
bool ChessBoard::isValidAdvisorMove(int fromRow, int fromCol, int toRow, int toCol) const {
    // Check if the move is within the palace and diagonal
    bool isInPalace = (toRow >= 0 && toRow <= 2 && toCol >= 3 && toCol <= 5) ||
                      (toRow >= 7 && toRow <= 9 && toCol >= 3 && toCol <= 5);
    int rowDiff = std::abs(toRow - fromRow);
    int colDiff = std::abs(toCol - fromCol);
    return isInPalace && rowDiff == 1 && colDiff == 1;
}

bool ChessBoard::isValidElephantMove(int fromRow, int fromCol, int toRow, int toCol) const {
    // Check if the move is exactly two steps diagonally and doesn't cross the river
    int rowDiff = std::abs(toRow - fromRow);
    int colDiff = std::abs(toCol - fromCol);
    bool doesNotCrossRiver = (fromRow < 5 && toRow < 5) || (fromRow >= 5 && toRow >= 5);
    
    // Check if there's no piece blocking the elephant's path
    int midRow = (fromRow + toRow) / 2;
    int midCol = (fromCol + toCol) / 2;
    bool pathClear = getPieceAt(midRow, midCol).type == PieceType::Empty;

    return rowDiff == 2 && colDiff == 2 && doesNotCrossRiver && pathClear;
}

bool ChessBoard::isValidHorseMove(int fromRow, int fromCol, int toRow, int toCol) const {
    int rowDiff = std::abs(toRow - fromRow);
    int colDiff = std::abs(toCol - fromCol);
    
    if ((rowDiff == 2 && colDiff == 1) || (rowDiff == 1 && colDiff == 2)) {
        // Check if there's no piece blocking the horse's path
        int blockRow = fromRow + (toRow - fromRow) / 2;
        int blockCol = fromCol + (toCol - fromCol) / 2;
        return getPieceAt(blockRow, blockCol).type == PieceType::Empty;
    }
    return false;
}

bool ChessBoard::isValidChariotMove(int fromRow, int fromCol, int toRow, int toCol) const {
    if (fromRow != toRow && fromCol != toCol) {
        return false; // Move must be along a straight line
    }

    int step = (fromRow == toRow) ? (toCol > fromCol ? 1 : -1) : (toRow > fromRow ? 1 : -1);
    int start = (fromRow == toRow) ? fromCol : fromRow;
    int end = (fromRow == toRow) ? toCol : toRow;

    for (int i = start + step; i != end; i += step) {
        if (getPieceAt(fromRow == toRow ? fromRow : i, fromRow == toRow ? i : fromCol).type != PieceType::Empty) {
            return false; // Path is not clear
        }
    }
    return true;
}

bool ChessBoard::isValidCannonMove(int fromRow, int fromCol, int toRow, int toCol) const {
    if (fromRow != toRow && fromCol != toCol) {
        return false; // Move must be along a straight line
    }

    int step = (fromRow == toRow) ? (toCol > fromCol ? 1 : -1) : (toRow > fromRow ? 1 : -1);
    int start = (fromRow == toRow) ? fromCol : fromRow;
    int end = (fromRow == toRow) ? toCol : toRow;

    int pieceCount = 0;
    for (int i = start + step; i != end; i += step) {
        if (getPieceAt(fromRow == toRow ? fromRow : i, fromRow == toRow ? i : fromCol).type != PieceType::Empty) {
            pieceCount++;
        }
    }

    ChessPiece targetPiece = getPieceAt(toRow, toCol);
    if (targetPiece.type == PieceType::Empty) {
        return pieceCount == 0; // For moving, path must be clear
    } else {
        return pieceCount == 1; // For capturing, must jump over exactly one piece
    }
}

bool ChessBoard::isValidSoldierMove(int fromRow, int fromCol, int toRow, int toCol) const {
    int rowDiff = toRow - fromRow;
    int colDiff = std::abs(toCol - fromCol);

    if (getPieceAt(fromRow, fromCol).color == PieceColor::Red) {
        if (fromRow < 5) { // Not crossed river
            return rowDiff == 1 && colDiff == 0;
        } else { // Crossed river
            return (rowDiff == 1 && colDiff == 0) || (rowDiff == 0 && colDiff == 1);
        }
    } else { // Black soldier
        if (fromRow >= 5) { // Not crossed river
            return rowDiff == -1 && colDiff == 0;
        } else { // Crossed river
            return (rowDiff == -1 && colDiff == 0) || (rowDiff == 0 && colDiff == 1);
        }
    }
}

// Helper function to get piece score
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
