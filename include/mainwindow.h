#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QGridLayout>
#include <QStatusBar>
#include <QLabel>
#include <QMediaPlayer>
#include <QEvent>
#include <QRegularExpression>
#include <QComboBox>
#include <QFile>
#include <QTextStream>
#include <QSoundEffect>
#include <QActionGroup>
#include <QMenu>
#include <QDebug>
#include <QMessageBox>
#include <QUrl>
#include <QRandomGenerator>
#include <QTimer>
#include <QMenuBar>
#include <QFileDialog>
#include <QInputDialog>
#include <QApplication>
#include "chessboard.h"
#include "chessai.h"

enum class ActionState {
    SelectPiece,
    MovePiece
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onButtonClicked();
    void onGameModeChanged(QAction *action);
    void onGameCompleted(int gameNumber, int redScore, int blackScore);
    void trainAI();
    void loadTrainedAI();
    void saveTrainedAI();
    void runAIGame();

private:

    int fromRow;
    int fromCol;
    bool gameStarted;
    ChessBoard chessBoard;
    QVector<QVector<QPushButton*>> buttons;
    PieceColor currentPlayer;
    int redScore;
    int blackScore;
    QLabel *redScoreLabel;
    QLabel *blackScoreLabel;
    QMediaPlayer *selectSound;
    QMediaPlayer *moveSound;
    QMediaPlayer *captureSound;
    ActionState currentActionState; // SelectPiece or MovePiece
    QSet<QPushButton*> highlightedButtons;
    QPushButton* currentSelectedButton;
    QComboBox *gameModeComboBox;
    ChessAI *chessAI;
    QAction* trainAIAction;
    QAction* loadAIAction;
    QAction* saveAIAction;
    QAction* runAIGameAction;
    int numGames{1000};
    enum class GameMode { HumanVsHuman, AIVsAI, HumanVsAI };
    GameMode currentGameMode;
    QFile logFile;
    PieceColor aiColor;
    QPair<int, int> lastAIMoveFrom;
    QPair<int, int> lastAIMoveTo;

    void createChessBoard();
    void updateBoardDisplay();
    void switchPlayer();
    void checkGameOver();
    void resetGame();
    void updateScoreDisplay();
    void updateScore(PieceType capturedPiece);
    bool eventFilter(QObject *obj, QEvent *event) override;
    void highlightLegalMoves(int row, int col);
    void clearHighlights();
    void highlightButton(QPushButton *button, bool highlight, const QString &color = "#FFA500");
    QString getInvalidMoveReason(int fromRow, int fromCol, int toRow, int toCol);
    void makeAIMove();
    void highlightAIMove(int fromRow, int fromCol, int toRow, int toCol);
    void updateButtonStyle(QPushButton* button);
    void setupMenuBar();
    void handleGameOver();

};

#endif // MAINWINDOW_H
