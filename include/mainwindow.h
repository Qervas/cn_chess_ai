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
#include <QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QProgressBar>
#include <QVBoxLayout>
#include <QStackedWidget>
#include "chessboard.h"
#include "chessai.h"
// using namespace QtCharts;

enum class ActionState {
    SelectPiece,
    MovePiece
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onButtonClicked();
    void onGameModeChanged(QAction *action);
    void trainAI();
    void loadTrainedAI();
    void saveTrainedAI();
    void runAIGame();
	void updateTrainingChart(int gameNumber, int redScore, int blackScore);


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
    PieceColor aiColor;
    QPair<int, int> lastAIMoveFrom;
    QPair<int, int> lastAIMoveTo;
    QChartView *trainingChartView;
    QLineSeries *redScoreSeries;
    QLineSeries *blackScoreSeries;
    QValueAxis *axisX;
    QValueAxis *axisY;
	QProgressBar *trainingProgressBar;
	QStackedWidget *stackedWidget;
	QWidget *chessboardPage;
	QWidget *trainingPage;
	QWidget *chessBoardWidget;

    void createChessBoard();
    void setupMenuBar();
	void setupTrainingChart();
	void setupPages();
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
    void handleGameOver();
	void onModelSaved(const QString&);
};

class Worker : public QObject
{
    Q_OBJECT
public:
    Worker(int numGames, const QString& modelFilename, QObject* parent = nullptr)
        : QObject(parent), numGames(numGames), modelFilename(modelFilename)
    {
        board = new ChessBoard();
        chessAI = new ChessAI(board);
    }
    ~Worker()
    {
        delete chessAI;
        delete board;
    }

public slots:
    void process()
    {
        // Connect chessAI signals to worker signals
        connect(chessAI, &ChessAI::gameCompleted, this, &Worker::gameCompleted);
        connect(chessAI, &ChessAI::trainingFinished, this, &Worker::onTrainingFinished);

        // Start training
        chessAI->train(numGames);

        emit finished();
    }

signals:
    void gameCompleted(int gameNumber, int redScore, int blackScore);
    void trainingFinished();
    void modelSaved(const QString& filename);
    void finished();

private slots:
    void onTrainingFinished()
    {
        // Save the model
        chessAI->saveModel(modelFilename);

        // Emit signal to inform the main thread
        emit modelSaved(modelFilename);
        emit trainingFinished();
    }

private:
    int numGames;
    QString modelFilename;
    ChessBoard* board;
    ChessAI* chessAI;
};

#endif // MAINWINDOW_H
