# Connect 4 GPU Game

A CUDA-based Connect 4 game implementation where two GPUs compete against each other using different strategies.

## Project Overview

This project implements a Connect 4 game where:
- **GPU 1**: Uses a minimax algorithm with alpha-beta pruning
- **GPU 2**: Uses a Monte Carlo Tree Search (MCTS) algorithm
- Both GPUs run in parallel, competing for the best move
- The game state is visualized and can be replayed

## Features

- **Multi-GPU Competition**: Two GPUs with different strategies
- **Real-time Visualization**: ASCII-based board display
- **Game Replay**: Save and replay complete games
- **Performance Metrics**: Track GPU computation times
- **Configurable Board Size**: Support for different board dimensions

## Requirements

- CUDA Toolkit 11.0 or higher
- NVIDIA GPU with compute capability 6.0+
- C++ compiler with C++11 support
- Make or CMake for building

## Building the Project

```bash
# Compile the project
nvcc -o connect4_gpu connect4_gpu.cu -O3

# Run the game
./connect4_gpu
```

## Game Rules

- Players take turns dropping pieces into columns
- First player to connect 4 pieces horizontally, vertically, or diagonally wins
- If the board fills up without a winner, the game is a draw

## GPU Strategies

### GPU 1: Minimax with Alpha-Beta Pruning
- Evaluates all possible moves up to a certain depth
- Uses alpha-beta pruning to reduce search space
- Optimized for speed and accuracy

### GPU 2: Monte Carlo Tree Search (MCTS)
- Uses random sampling to evaluate board positions
- Balances exploration and exploitation
- Adapts strategy based on game state

## Output Format

The game outputs:
- Current board state after each move
- Which GPU made the move
- Computation time for each GPU
- Final game result
- Complete move history for replay

## File Structure

- `connect4_gpu.cu`: Main CUDA implementation
- `game_visualizer.py`: Python script for game replay visualization
- `README.md`: This file
- `Makefile`: Build configuration

## Usage Examples

```bash
# Run a single game
./connect4_gpu

# Run multiple games and save results
./connect4_gpu --games 10 --output results.txt

# Visualize a saved game
python game_visualizer.py results.txt
``` 