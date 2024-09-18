# Chinese Chess (Xiangqi) AI

This project is an implementation of Chinese Chess (Xiangqi) with an AI opponent using Qt and C++.

## Features

- Graphical user interface for Chinese Chess
- Human vs Human gameplay
- Human vs AI gameplay
- AI vs AI gameplay
- Deep Q-Learning Network (DQN) based AI
- AI training functionality
- Save and load trained AI models

## Requirements

- Qt 6 or later
- C++17 compatible compiler
- CMake 3.16 or later

## Building the Project

1. Make sure you have Qt and CMake installed on your system.
2. Clone this repository.
3. Navigate to the project directory.
4. Create a build directory:
   ```
   mkdir build
   cd build
   ```
5. Run CMake:
   ```
   cmake ..

   # for Windows user, if you want to call your vs compiler in the command line window
   cmake .. -G "Visual Studio 17 2022" # change the 17 and 2022 to suit your situation
   ```
6. Build the project:
   ```
   cmake --build .
   ```

## Project Structure

- `src/`: Contains the source files
- `include/`: Contains the header files
- `resources/`: Contains resource files (sounds, images)

How to Play

1. Run the executable generated after building the project.
2. Use the menu to select the game mode (Human vs Human, Human vs AI, or AI vs AI).
3. Click on a piece to select it, then click on a valid destination to move the piece.
4. The game ends when one player's General (King) is captured.

## AI Training

To train the AI:

1. Go to the AI menu in the application.
2. Select "Train AI" and input the number of training episodes.
3. Wait for the training to complete.
4. You can save the trained model using the "Save trained AI" option.
