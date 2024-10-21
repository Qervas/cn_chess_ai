#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include <QVector>
#include <QString>
#include <QPair>

enum class PieceType {
    Empty, General, Advisor, Elephant, Horse, Chariot, Cannon, Soldier
};

enum class PieceColor {
    Red, Black, None
};

struct ChessPiece {
    PieceType type;
    PieceColor color;
    ChessPiece() : type(PieceType::Empty), color(PieceColor::None) {}
    ChessPiece(PieceType t, PieceColor c) : type(t), color(c) {}
};

enum class PieceScore {
    General = 1000,
    Advisor = 20,
    Elephant = 20,
    Horse = 40,
    Chariot = 90,
    Cannon = 45,
    Soldier = 10
};

class ChessBoard {
public:
    ChessBoard();
    void initializeBoard();
    ChessPiece getPieceAt(int row, int col) const;
    ChessPiece movePiece(int fromRow, int fromCol, int toRow, int toCol);
    bool isValidMove(int fromRow, int fromCol, int toRow, int toCol) const;
    void reset();
    int getRedScore() const;
    int getBlackScore() const;
    bool checkGameOver() const;
    PieceColor getWinner() const;
    PieceColor getCurrentPlayer() const { return currentPlayer; }
    int getMoveCount() const { return moveCount; }
    QVector<QPair<int, int>> getValidMoves(int row, int col) const;

    bool isInsideBoard(int row, int col) const;
    bool isValidGeneralMove(int fromRow, int fromCol, int toRow, int toCol) const;
    bool isValidAdvisorMove(int fromRow, int fromCol, int toRow, int toCol) const;
    bool isValidElephantMove(int fromRow, int fromCol, int toRow, int toCol) const;
    bool isValidHorseMove(int fromRow, int fromCol, int toRow, int toCol) const;
    bool isValidChariotMove(int fromRow, int fromCol, int toRow, int toCol) const;
    bool isValidCannonMove(int fromRow, int fromCol, int toRow, int toCol) const;
    bool isValidSoldierMove(int fromRow, int fromCol, int toRow, int toCol) const;

private:
    static constexpr int ROWS = 10;
    static constexpr int COLS = 9;
    QVector<ChessPiece> board; // Flat vector representing the board
    int moveCount{0};
    PieceColor currentPlayer{PieceColor::Red};
    bool isGeneralAlive(PieceColor color) const;

    bool isInRedPalace(int row, int col) const {
        return row >= 0 && row <= 2 && col >= 3 && col <= 5;
    }

    bool isInBlackPalace(int row, int col) const {
        return row >= 7 && row <= 9 && col >= 3 && col <= 5;
    }

    bool isInOwnSide(PieceColor color, int row) const {
        return (color == PieceColor::Red) ? (row >= 0 && row <= 4) : (row >= 5 && row <= 9);
    }

    int redScore;
    int blackScore;

	void generateGeneralMoves(int, int, QVector<QPair<int, int>>&) const;
	void generateAdvisorMoves(int, int, QVector<QPair<int, int>>&) const;
	void generateElephantMoves(int, int, QVector<QPair<int, int>>&) const;
	void generateChariotMoves(int, int, QVector<QPair<int, int>>&) const;
	void generateCannonMoves(int, int, QVector<QPair<int, int>>&) const;
	void generateHorseMoves(int, int, QVector<QPair<int, int>>&) const;
	void generateSoldierMoves(int, int, QVector<QPair<int, int>>&) const;

};

int getPieceScore(PieceType type);

#endif // CHESSBOARD_H
