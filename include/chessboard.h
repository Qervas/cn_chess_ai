#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include <QVector>
#include <QString>
#include <QMap>
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


struct Position {
    int row;
    int col;
    Position(int r = 0, int c = 0) : row(r), col(c) {}
    bool operator<(const Position& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
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
    PieceColor getCurrentPlayer() const{return currentPlayer;}
    int getMoveCount() const { return moveCount; }
    QVector<QPair<int, int>> getValidMoves(int row, int col) const;



private:
    QMap<Position, ChessPiece> board;
    int moveCount{0};
    PieceColor currentPlayer{PieceColor::Red};
    bool isInsideBoard(int row, int col) const;
    bool isValidGeneralMove(int fromRow, int fromCol, int toRow, int toCol) const;
    bool isValidAdvisorMove(int fromRow, int fromCol, int toRow, int toCol) const;
    bool isValidElephantMove(int fromRow, int fromCol, int toRow, int toCol) const;
    bool isValidHorseMove(int fromRow, int fromCol, int toRow, int toCol) const;
    bool isValidChariotMove(int fromRow, int fromCol, int toRow, int toCol) const;
    bool isValidCannonMove(int fromRow, int fromCol, int toRow, int toCol) const;
    bool isValidSoldierMove(int fromRow, int fromCol, int toRow, int toCol) const;
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
};

int getPieceScore(PieceType type);


#endif // CHESSBOARD_H
