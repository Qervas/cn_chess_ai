#include "mainwindow.h"
#include "chessai.h"
#include <QMediaPlayer>
#include <QProcess>

QMediaPlayer *createMediaPlayer(QObject *parent) {
    QMediaPlayer *player = new QMediaPlayer(parent);
    if (!player->isAvailable()) {
        delete player;
        qDebug() << "Default backend not available, trying GStreamer";
        QProcess::execute("export", QStringList() << "QT_MULTIMEDIA_PREFERRED_PLUGINS=gstreamer");
        player = new QMediaPlayer(parent);
    }
    return player;
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), currentPlayer(PieceColor::Red), redScore(0), blackScore(0), gameStarted(false),
     currentActionState(ActionState::SelectPiece), currentSelectedButton(nullptr),
     lastAIMoveFrom(-1, -1), lastAIMoveTo(-1, -1)
{
    createChessBoard();
    updateBoardDisplay();

    setMinimumSize(700, 926);


    // create status bar
    redScoreLabel = new QLabel("Red Score: 0", this);
    blackScoreLabel = new QLabel("Black Score: 0", this);
    statusBar()->addPermanentWidget(redScoreLabel);
    statusBar()->addPermanentWidget(blackScoreLabel);

    // Initialize sound effects
    selectSound = createMediaPlayer(this);
    moveSound = createMediaPlayer(this);
    captureSound = createMediaPlayer(this);
    selectSound->setSource(QUrl("qrc:/resources/sounds/select.mp3"));
    moveSound->setSource(QUrl("qrc:/resources/sounds/move.mp3"));
    captureSound->setSource(QUrl("qrc:/resources/sounds/capture.mp3"));



    setStyleSheet(R"(
        QMainWindow {
            background-color: #f0d9b5;
        }
        QLabel {
            font-weight: bold;
            font-size: 14pt;
            color: #ff0000; /* Red color for visibility */
        }
        QStatusBar {
            font-weight: bold;
            font-size: 14pt;
            color: #ff0000; /* Red color for visibility */
        }
        QPushButton {
            background-color: #f0d9b5;
            border: 1px solid #b58863;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #f0d9b5;
            border: 2px solid #8f5902;
        }
    )");

    setWindowTitle("Chinese Chess");

    logFile.setFileName("ai_game_log.txt");
    if (!logFile.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Append)) {
        qDebug() << "Failed to open log file";
    }

    // create AI object
    chessAI = new ChessAI(&chessBoard, this);
    connect(chessAI, &ChessAI::gameCompleted, this, &MainWindow::onGameCompleted);

    setupMenuBar();
}

MainWindow::~MainWindow() {
    for (auto& row : buttons) {
        for (auto& button : row) {
            delete button;
        }
    }
    if (logFile.isOpen()) {
        logFile.close();
    }
    delete selectSound;
    delete moveSound;
    delete captureSound;
    delete chessAI;
}


void MainWindow::onGameModeChanged(QAction *action)
{
    if (action->text() == tr("Human vs Human")) {
        currentGameMode = GameMode::HumanVsHuman;
    } else if (action->text() == tr("Human vs AI")) {
        currentGameMode = GameMode::HumanVsAI;
    } else if (action->text() == tr("AI vs AI")) {
        currentGameMode = GameMode::AIVsAI;
    }

    resetGame();

    switch (currentGameMode) {
        case GameMode::HumanVsHuman:
            statusBar()->showMessage(tr("Mode: Human vs Human"));
            break;
        case GameMode::HumanVsAI:
            statusBar()->showMessage(tr("Mode: Human vs AI"));
            break;
        case GameMode::AIVsAI:
            statusBar()->showMessage(tr("Mode: AI vs AI"));
            runAIGame();
            break;
    }
}

void MainWindow::resetGame()
{
    chessBoard.reset();
    updateBoardDisplay();
    clearHighlights();
    currentPlayer = PieceColor::Red;

    // reset scores
    redScore = 0;
    blackScore = 0;
    updateScoreDisplay();

    lastAIMoveFrom = qMakePair(-1, -1);
    lastAIMoveTo = qMakePair(-1, -1);

    switch (currentGameMode) {
        case GameMode::AIVsAI:
            for (auto &row : buttons) {
                for (auto &button : row) {
                    button->setEnabled(false);
                }
            }
            break;
        case GameMode::HumanVsAI:
            aiColor = (QRandomGenerator::global()->bounded(2) == 0) ? PieceColor::Red : PieceColor::Black;
            for (auto &row : buttons) {
                for (auto &button : row) {
                    button->setEnabled(true);
                }
            }
            if (aiColor == PieceColor::Red) {
                QTimer::singleShot(500, this, &MainWindow::makeAIMove);
            }
            break;
        case GameMode::HumanVsHuman:
            for (auto &row : buttons) {
                for (auto &button : row) {
                    button->setEnabled(true);
                }
            }
            break;
    }

    // update status bar
    QString modeStr;
    switch (currentGameMode) {
        case GameMode::HumanVsHuman: modeStr = "Human vs Human"; break;
        case GameMode::AIVsAI: modeStr = "AI vs AI"; break;
        case GameMode::HumanVsAI: modeStr = "Human vs AI"; break;
    }
    statusBar()->showMessage(tr("New game started - Mode: %1 - Current player: Red").arg(modeStr));
}

void MainWindow::createChessBoard() {
    QWidget *centralWidget = new QWidget(this);
    QGridLayout *gridLayout = new QGridLayout(centralWidget);

    buttons.resize(10);
    for (int i = 0; i < 10; ++i) {
        buttons[i].resize(9);
        for (int j = 0; j < 9; ++j) {
            QPushButton *button = new QPushButton(this);
            button->setFixedSize(60, 60);
            button->setProperty("row", i);
            button->setProperty("col", j);
            connect(button, &QPushButton::clicked, this, &MainWindow::onButtonClicked);
            button->installEventFilter(this);  // add event filter to handle mouse hover events
            gridLayout->addWidget(button, i, j);
            buttons[i][j] = button;
        }
    }

    setCentralWidget(centralWidget);
}

void MainWindow::updateBoardDisplay() {
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 9; ++j) {
            ChessPiece piece = chessBoard.getPieceAt(i, j);
            QString text;
            QString englishText;
            QString styleSheet = "QPushButton { font-family: 'Noto Sans CJK SC'; font-size: 24px; ";
            switch (piece.type) {
                case PieceType::General:
                    text = piece.color == PieceColor::Red ? "帥" : "將";
                    englishText = piece.color == PieceColor::Red ? "King" : "General";
                    break;
                case PieceType::Advisor:
                    text = piece.color == PieceColor::Red ? "仕" : "士";
                    englishText = "Advisor";
                    break;
                case PieceType::Elephant:
                    text = piece.color == PieceColor::Red ? "相" : "象";
                    englishText = piece.color == PieceColor::Red ? "Bishop" : "Elephant";
                    break;
                case PieceType::Horse:
                    text = piece.color == PieceColor::Red ? "馬" : "馬";
                    englishText = "Horse";
                    break;
                case PieceType::Chariot:
                    text = piece.color == PieceColor::Red ? "車" : "車";
                    englishText = "Chariot";
                    break;
                case PieceType::Cannon:
                    text = piece.color == PieceColor::Red ? "炮" : "砲";
                    englishText = "Cannon";
                    break;
                case PieceType::Soldier:
                    text = piece.color == PieceColor::Red ? "兵" : "卒";
                    englishText = piece.color == PieceColor::Red ? "Soldier" : "Pawn";
                    break;
                default:
                    text = "";
                    englishText = "";
                    break;
            }
            if (piece.color == PieceColor::Red) {
                styleSheet += "color: #FF4136; }";
            } else if (piece.color == PieceColor::Black) {
                styleSheet += "color: #001f3f; }";
            } else {
                styleSheet += "}";
            }

            // Combine Chinese and English text
            QString combinedText = text + "\n" + englishText;
            buttons[i][j]->setText(combinedText);

            // Adjust button size to accommodate both texts
            buttons[i][j]->setFixedSize(80, 80);

            // Only update the style if this button is not the starting or ending point of the last AI move.
            if (qMakePair(i, j) != lastAIMoveFrom && qMakePair(i, j) != lastAIMoveTo) {
                buttons[i][j]->setStyleSheet(styleSheet);
            }
        }
    }
}

void MainWindow::onButtonClicked() {
    static bool isProcessing = false;
    if (isProcessing) return;
    isProcessing = true;

    QPushButton *button = qobject_cast<QPushButton*>(sender());
    if (!button) {
        isProcessing = false;
        return;
    }

    int row = button->property("row").toInt();
    int col = button->property("col").toInt();

    qDebug() << "Clicked on:" << row << col;

    ChessPiece piece = chessBoard.getPieceAt(row, col);

    if (currentActionState == ActionState::SelectPiece) {
        if (piece.type != PieceType::Empty && piece.color == currentPlayer) {
            // Select the piece
            clearHighlights();
            highlightButton(button, true, "#FFFF00");
            highlightLegalMoves(row, col);
            currentSelectedButton = button;
            fromRow = row;
            fromCol = col;
            selectSound->play();
            currentActionState = ActionState::MovePiece;
            qDebug() << "Piece selected, switched to MovePiece state";
        } else {
            statusBar()->showMessage("Please select your piece", 3000);
        }
    } else if (currentActionState == ActionState::MovePiece) {
        if (row == fromRow && col == fromCol) {
            // Deselect the piece
            clearHighlights();
            currentSelectedButton = nullptr;
            currentActionState = ActionState::SelectPiece;
            qDebug() << "Piece deselected, switched to SelectPiece state";
        } else if (chessBoard.isValidMove(fromRow, fromCol, row, col)) {
            // Move the piece
            ChessPiece capturedPiece = chessBoard.movePiece(fromRow, fromCol, row, col);
            if (capturedPiece.type != PieceType::Empty) {
                updateScore(capturedPiece.type);
                captureSound->play();
            } else {
                moveSound->play();
            }

            clearHighlights();
            updateBoardDisplay();
            checkGameOver();
            switchPlayer();

            currentActionState = ActionState::SelectPiece;
            currentSelectedButton = nullptr;

            qDebug() << "Moved piece from" << fromRow << fromCol << "to" << row << col;
        } else {
            QString reason = getInvalidMoveReason(fromRow, fromCol, row, col);
            statusBar()->showMessage(reason, 3000);
        }
    }

    // Check if it's AI's turn to move
    if (currentGameMode == GameMode::HumanVsAI && currentPlayer == aiColor) {
        QTimer::singleShot(500, this, &MainWindow::makeAIMove);
    }

    isProcessing = false;
}

void MainWindow::checkGameOver() {
    if (chessBoard.checkGameOver()) {
        PieceColor winner = chessBoard.getWinner();
        QString winnerStr = (winner == PieceColor::Red) ? "Red" : "Black";
        QMessageBox::information(this, "Game Over", winnerStr + " wins!");

        // Ask the player if they want to start a new game
        QMessageBox::StandardButton reply;
        reply = QMessageBox::question(this, "New Game", "Do you want to start a new game?",
                                      QMessageBox::Yes | QMessageBox::No);
        if (reply == QMessageBox::Yes) {
            resetGame();
        } else {
            // If the player chooses not to start a new game, you can close the window or perform other actions
            // Here we choose to close the window
            close();
        }
    }
}

void MainWindow::switchPlayer() {
    currentPlayer = (currentPlayer == PieceColor::Red) ? PieceColor::Black : PieceColor::Red;
    QString playerColor = (currentPlayer == PieceColor::Red) ? "Red" : "Black";
    statusBar()->showMessage("Current player: " + playerColor);
}

void MainWindow::updateScoreDisplay() {
    redScoreLabel->setText(QString("Red Score: %1").arg(redScore));
    blackScoreLabel->setText(QString("Black Score: %1").arg(blackScore));
}

void MainWindow::updateScore(PieceType capturedPiece) {
    int score = getPieceScore(capturedPiece);
    if (currentPlayer == PieceColor::Red) {
        redScore += score;
    } else {
        blackScore += score;
    }
    updateScoreDisplay();
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event) {
    if (!gameStarted) return QMainWindow::eventFilter(obj, event);

    QPushButton *button = qobject_cast<QPushButton*>(obj);
    if (button) {
        bool isChessButton = false;
        for (const auto& row : buttons) {
            if (row.contains(button)) {
                isChessButton = true;
                break;
            }
        }
    }
    return QMainWindow::eventFilter(obj, event);
}

void MainWindow::highlightLegalMoves(int row, int col) {
    ChessPiece piece = chessBoard.getPieceAt(row, col);
    qDebug() << "Highlighting legal moves for piece type:" << static_cast<int>(piece.type) << "at" << row << col;

    QVector<QPair<int, int>> validMoves = chessBoard.getValidMoves(row, col);

    for (const auto& move : validMoves) {
        int i = move.first;
        int j = move.second;
        highlightButton(buttons[i][j], true, "#90EE90");  // Use light green to highlight legal moves
        highlightedButtons.insert(buttons[i][j]);
    }

    qDebug() << "Total valid moves:" << validMoves.size();
}

void MainWindow::clearHighlights() {
    for (auto button : highlightedButtons) {
        button->setStyleSheet("");  // Reset to default style
    }
    highlightedButtons.clear();

    if (currentSelectedButton) {
        currentSelectedButton->setStyleSheet("");  // Reset to default style
        currentSelectedButton = nullptr;
    }

    fromRow = -1;
    fromCol = -1;

    // Reapply default style to all buttons
    for (auto &row : buttons) {
        for (auto &button : row) {
            ChessPiece piece = chessBoard.getPieceAt(button->property("row").toInt(), button->property("col").toInt());
            QString textColor = (piece.color == PieceColor::Red) ? "#FF0000" : "#000000";
            button->setStyleSheet(QString("QPushButton {"
                                          "background-color: #f0d9b5;"
                                          "border: 1px solid #b58863;"
                                          "border-radius: 5px;"
                                          "font-size: 24px;"
                                          "color: %1;"
                                          "}")
                                  .arg(textColor));
        }
    }
}

void MainWindow::highlightButton(QPushButton *button, bool highlight, const QString &color) {
    QString baseStyle = "QPushButton {"
                        "background-color: %1;"
                        "border: 3px solid #FF0000;"
                        "font-weight: bold;"
                        "font-size: 24px;"
                        "}";

    if (highlight) {
        QString textColor = button->property("color").toString();
        if (textColor.isEmpty()) {
            textColor = button->palette().color(QPalette::ButtonText).name();
        }

        button->setStyleSheet(baseStyle.arg(color) +
                              QString("QPushButton { color: %1; }"
                                      "QPushButton:hover { background-color: #FFA500; }").arg(textColor));
        highlightedButtons.insert(button);
    } else {
        if (!highlightedButtons.contains(button)) {
            QString textColor = button->property("color").toString();
            if (textColor.isEmpty()) {
                textColor = button->palette().color(QPalette::ButtonText).name();
            }

            button->setStyleSheet(QString("QPushButton {"
                                          "background-color: #f0d9b5;"
                                          "border: 1px solid #b58863;"
                                          "border-radius: 5px;"
                                          "font-size: 24px;"
                                          "color: %1;"
                                          "}"
                                          "QPushButton:hover { background-color: #e6c9a3; }").arg(textColor));
        }
    }
}

QString MainWindow::getInvalidMoveReason(int fromRow, int fromCol, int toRow, int toCol) {
    ChessPiece fromPiece = chessBoard.getPieceAt(fromRow, fromCol);
    ChessPiece toPiece = chessBoard.getPieceAt(toRow, toCol);

    if (fromPiece.type == PieceType::Empty) {
        return "Invalid move: No piece at the starting position";
    }

    if (fromPiece.color != currentPlayer) {
        return "Invalid move: Cannot move the opponent's piece";
    }

    if (toPiece.type != PieceType::Empty && toPiece.color == currentPlayer) {
        return "Invalid move: Target position already occupied by your piece";
    }

    // Return specific invalid move reasons based on the piece type
    switch (fromPiece.type) {
        case PieceType::General:
            return "Invalid move: The General can only move one step within the palace";
        case PieceType::Advisor:
            return "Invalid move: The Advisor can only move diagonally within the palace";
        case PieceType::Elephant:
            return "Invalid move: The Elephant can only move in a 'field' pattern and cannot cross the river";
        case PieceType::Horse:
            return "Invalid move: The Horse moves in an 'L' shape and cannot be blocked";
        case PieceType::Chariot:
            return "Invalid move: The Chariot can only move in straight lines and cannot jump over pieces";
        case PieceType::Cannon:
            return "Invalid move: The Cannon cannot jump over pieces when moving, but must jump over exactly one piece when capturing";
        case PieceType::Soldier:
            return "Invalid move: The Soldier can only move forward, and can move sideways after crossing the river";
        default:
            return "Invalid move: Unknown reason";
    }
}

void MainWindow::setupMenuBar()
{
    QMenu *gameMenu = menuBar()->addMenu(tr("Game"));
    QMenu *aiMenu = menuBar()->addMenu(tr("AI"));

    // Game mode selection
    QActionGroup *gameModeGroup = new QActionGroup(this);
    QAction *humanVsHumanAction = gameMenu->addAction(tr(" Human vs Human"));
    QAction *humanVsAIAction = gameMenu->addAction(tr("Human vs AI"));
    QAction *aiVsAIAction = gameMenu->addAction(tr("AI vs AI"));

    humanVsHumanAction->setCheckable(true);
    humanVsAIAction->setCheckable(true);
    aiVsAIAction->setCheckable(true);

    gameModeGroup->addAction(humanVsHumanAction);
    gameModeGroup->addAction(humanVsAIAction);
    gameModeGroup->addAction(aiVsAIAction);

    connect(gameModeGroup, &QActionGroup::triggered, this, &MainWindow::onGameModeChanged);

    gameMenu->addSeparator();

    // Start a new game
    QAction *newGameAction = gameMenu->addAction(tr("New Game"));
    connect(newGameAction, &QAction::triggered, this, &MainWindow::resetGame);

    // AI Menu
    trainAIAction = aiMenu->addAction(tr("train AI"));
    loadAIAction = aiMenu->addAction(tr("load trained AI"));
    saveAIAction = aiMenu->addAction(tr("save trained AI"));
    runAIGameAction = aiMenu->addAction(tr("Run AI Battle"));

    connect(trainAIAction, &QAction::triggered, this, &MainWindow::trainAI);
    connect(loadAIAction, &QAction::triggered, this, &MainWindow::loadTrainedAI);
    connect(saveAIAction, &QAction::triggered, this, &MainWindow::saveTrainedAI);
    connect(runAIGameAction, &QAction::triggered, this, &MainWindow::runAIGame);

    // default human vs human mode
    humanVsHumanAction->setChecked(true);
    currentGameMode = GameMode::HumanVsHuman;
}

void MainWindow::trainAI()
{
    bool ok;
    int episodes = QInputDialog::getInt(this, tr("Train AI"),
                                        tr("Number of episodes:"), 1000, 1, 1000000, 1, &ok);
    if (ok) {
        QMessageBox::information(this, tr("Training Started"),
                                 tr("AI training started. This may take a while."));
        chessAI->train(episodes);
        QMessageBox::information(this, tr("Training Completed"),
                                 tr("AI training completed."));
    }
}

void MainWindow::loadTrainedAI()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Load AI Model"),
                                                    "", tr("AI Model Files (*.bin)"));
    if (!filename.isEmpty()) {
        chessAI->loadModel(filename);
        QMessageBox::information(this, tr("Model Loaded"),
                                 tr("AI model loaded successfully."));
    }
}

void MainWindow::saveTrainedAI()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Save AI Model"),
                                                    "", tr("AI Model Files (*.bin)"));
    if (!filename.isEmpty()) {
        chessAI->saveModel(filename);
        QMessageBox::information(this, tr("Model Saved"),
                                 tr("AI model saved successfully."));
    }
}

void MainWindow::runAIGame()
{
    // Here we only run one AI vs AI game
    resetGame();
    currentGameMode = GameMode::AIVsAI;

    while (!chessBoard.checkGameOver()) {
        makeAIMove();
        updateBoardDisplay();
        QApplication::processEvents();
        QTimer::singleShot(500, this, [this]() {
            // Code that needs to be executed after a delay
            makeAIMove();
            updateBoardDisplay();
            checkGameOver();
        });
    }

    checkGameOver();
}

void MainWindow::makeAIMove()
{
    if (chessBoard.checkGameOver()) {
        handleGameOver();
        return;
    }

    QPair<QPair<int, int>, QPair<int, int>> move = chessAI->getAIMove(currentPlayer);
    int fromRow = move.first.first;
    int fromCol = move.first.second;
    int toRow = move.second.first;
    int toCol = move.second.second;

    // execute AI's move
    ChessPiece capturedPiece = chessBoard.movePiece(fromRow, fromCol, toRow, toCol);
    if (capturedPiece.type != PieceType::Empty) {
        updateScore(capturedPiece.type);
        captureSound->play();
    } else {
        moveSound->play();
    }

    updateBoardDisplay();

    // highlight AI's moves only in human vs AI mode
    if (currentGameMode == GameMode::HumanVsAI) {
        highlightAIMove(fromRow, fromCol, toRow, toCol);
    }

    if (chessBoard.checkGameOver()) {
        handleGameOver();
    } else {
        switchPlayer();
    }
}

void MainWindow::handleGameOver()
{
    PieceColor winner = chessBoard.getWinner();
    QString winnerStr = (winner == PieceColor::Red) ? "Red" : "Black";
    QMessageBox::information(this, "Game Over", winnerStr + " wins!");

    // ask if the player wants to start a new game
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "New Game", "Do you want to start a new game?",
                                  QMessageBox::Yes|QMessageBox::No);
    if (reply == QMessageBox::Yes) {
        resetGame();
    } else {
        // If the player chooses not to start a new game, you can close the window or perform other actions
        close();
    }
}

void MainWindow::onGameCompleted(int gameNumber, int redScore, int blackScore)
{
    updateScoreDisplay();
    QString result = (redScore > blackScore) ? "Red wins!" : (blackScore > redScore) ? "Black wins!" : "It's a draw!";

    // Write the result to the log file
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

void MainWindow::updateButtonStyle(QPushButton* button)
{
    ChessPiece piece = chessBoard.getPieceAt(button->property("row").toInt(), button->property("col").toInt());
    QString textColor = (piece.color == PieceColor::Red) ? "#FF0000" : "#000000";
    button->setStyleSheet(QString("QPushButton {"
                                  "background-color: #f0d9b5;"
                                  "border: 1px solid #b58863;"
                                  "border-radius: 5px;"
                                  "font-size: 24px;"
                                  "color: %1;"
                                  "}")
                          .arg(textColor));
}

void MainWindow::highlightAIMove(int fromRow, int fromCol, int toRow, int toCol)
{
    // Clear the previous highlight
    if (lastAIMoveFrom.first != -1) {
        updateButtonStyle(buttons[lastAIMoveFrom.first][lastAIMoveFrom.second]);
    }
    if (lastAIMoveTo.first != -1) {
        updateButtonStyle(buttons[lastAIMoveTo.first][lastAIMoveTo.second]);
    }

    // Set the new highlight
    buttons[fromRow][fromCol]->setStyleSheet(buttons[fromRow][fromCol]->styleSheet() +
        "QPushButton { background-color: #FFA500 !important; border: 2px solid #FF0000 !important; }");
    buttons[toRow][toCol]->setStyleSheet(buttons[toRow][toCol]->styleSheet() +
        "QPushButton { background-color: #32CD32 !important; border: 2px solid #FF0000 !important; }");

    // Update the last AI move positions
    lastAIMoveFrom = qMakePair(fromRow, fromCol);
    lastAIMoveTo = qMakePair(toRow, toCol);

    // Force the UI to update
    buttons[fromRow][fromCol]->update();
    buttons[toRow][toCol]->update();
}
