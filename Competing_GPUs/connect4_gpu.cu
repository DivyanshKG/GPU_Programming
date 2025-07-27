#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>

// Game constants
#define BOARD_WIDTH 7
#define BOARD_HEIGHT 6
#define BOARD_SIZE (BOARD_WIDTH * BOARD_HEIGHT)
#define WIN_LENGTH 4
#define MAX_MOVES (BOARD_WIDTH * BOARD_HEIGHT)
#define MINIMAX_DEPTH 4
#define MCTS_ITERATIONS 1000

// Game state representation
struct GameState {
    int board[BOARD_HEIGHT][BOARD_WIDTH];
    int currentPlayer;
    int moveCount;
    int lastMove;
};

// Move structure for replay
struct Move {
    int column;
    int player;
    float gpu1Time;
    float gpu2Time;
    int winner;
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Global variables for game replay
std::vector<Move> gameMoves;
std::vector<GameState> gameStates;

// Function declarations
__device__ bool checkWin(int board[BOARD_HEIGHT][BOARD_WIDTH], int row, int col, int player);
__device__ bool isValidMove(int board[BOARD_HEIGHT][BOARD_WIDTH], int col);
__device__ void makeMove(int board[BOARD_HEIGHT][BOARD_WIDTH], int col, int player);
__device__ int evaluateBoard(int board[BOARD_HEIGHT][BOARD_WIDTH], int player);
__device__ int getAvailableMoves(int board[BOARD_HEIGHT][BOARD_WIDTH], int moves[BOARD_WIDTH]);

// GPU 1: Simple evaluation-based strategy
__global__ void evaluationKernel(int* d_board, int* d_scores, int player) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= BOARD_WIDTH) return;
    
    // Copy board to local memory
    int localBoard[BOARD_HEIGHT][BOARD_WIDTH];
    for (int i = 0; i < BOARD_HEIGHT; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            localBoard[i][j] = d_board[i * BOARD_WIDTH + j];
        }
    }
    
    // Check if move is valid
    if (!isValidMove(localBoard, col)) {
        d_scores[col] = -10000;
        return;
    }
    
    // Make the move
    makeMove(localBoard, col, player);
    
    // Check for immediate win
    int row = 0;
    for (int r = BOARD_HEIGHT - 1; r >= 0; r--) {
        if (localBoard[r][col] == player) {
            row = r;
            break;
        }
    }
    
    if (checkWin(localBoard, row, col, player)) {
        d_scores[col] = 10000;  // Winning move
        return;
    }
    
    // Check if opponent can win next move
    int opponent = player == 1 ? 2 : 1;
    for (int c = 0; c < BOARD_WIDTH; c++) {
        if (isValidMove(localBoard, c)) {
            int tempBoard[BOARD_HEIGHT][BOARD_WIDTH];
            memcpy(tempBoard, localBoard, sizeof(tempBoard));
            makeMove(tempBoard, c, opponent);
            
            int tempRow = 0;
            for (int r = BOARD_HEIGHT - 1; r >= 0; r--) {
                if (tempBoard[r][c] == opponent) {
                    tempRow = r;
                    break;
                }
            }
            
            if (checkWin(tempBoard, tempRow, c, opponent)) {
                d_scores[col] = -5000;  // Block opponent's win
                return;
            }
        }
    }
    
    // Evaluate board position
    int score = evaluateBoard(localBoard, player) - evaluateBoard(localBoard, opponent);
    
    // Prefer center columns
    if (col == 3) score += 100;
    else if (col == 2 || col == 4) score += 50;
    else if (col == 1 || col == 5) score += 25;
    
    d_scores[col] = score;
}

// GPU 2: Monte Carlo Tree Search
__global__ void mctsKernel(int* d_board, int* d_visitCount, int* d_winCount, 
                           int iterations, int player) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= iterations) return;
    
    // Copy board to local memory
    int localBoard[BOARD_HEIGHT][BOARD_WIDTH];
    for (int i = 0; i < BOARD_HEIGHT; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            localBoard[i][j] = d_board[i * BOARD_WIDTH + j];
        }
    }
    
    // Select a random valid move
    int validMoves[BOARD_WIDTH];
    int numMoves = getAvailableMoves(localBoard, validMoves);
    if (numMoves == 0) return;
    
    int randomMove = validMoves[idx % numMoves];
    
    // Simulate random playout
    int simBoard[BOARD_HEIGHT][BOARD_WIDTH];
    memcpy(simBoard, localBoard, sizeof(simBoard));
    
    makeMove(simBoard, randomMove, player);
    
    // Simulate random game to completion
    int currentPlayer = player == 1 ? 2 : 1;
    int moves = 1;
    
    while (moves < MAX_MOVES) {
        int moves2[BOARD_WIDTH];
        int numMoves2 = getAvailableMoves(simBoard, moves2);
        if (numMoves2 == 0) break;
        
        int randomMove2 = moves2[idx % numMoves2];
        makeMove(simBoard, randomMove2, currentPlayer);
        
        // Check for win
        int row = 0;
        for (int r = BOARD_HEIGHT - 1; r >= 0; r--) {
            if (simBoard[r][randomMove2] == currentPlayer) {
                row = r;
                break;
            }
        }
        
        if (checkWin(simBoard, row, randomMove2, currentPlayer)) {
            if (currentPlayer == player) {
                atomicAdd(&d_winCount[randomMove], 1);
            }
            break;
        }
        
        currentPlayer = currentPlayer == 1 ? 2 : 1;
        moves++;
    }
    
    atomicAdd(&d_visitCount[randomMove], 1);
}

// Device functions
__device__ bool checkWin(int board[BOARD_HEIGHT][BOARD_WIDTH], int row, int col, int player) {
    // Check horizontal
    int count = 0;
    for (int c = 0; c < BOARD_WIDTH; c++) {
        if (board[row][c] == player) {
            count++;
            if (count >= WIN_LENGTH) return true;
        } else {
            count = 0;
        }
    }
    
    // Check vertical
    count = 0;
    for (int r = 0; r < BOARD_HEIGHT; r++) {
        if (board[r][col] == player) {
            count++;
            if (count >= WIN_LENGTH) return true;
        } else {
            count = 0;
        }
    }
    
    // Check diagonal (top-left to bottom-right)
    count = 0;
    int startRow = row - min(row, col);
    int startCol = col - min(row, col);
    while (startRow < BOARD_HEIGHT && startCol < BOARD_WIDTH) {
        if (board[startRow][startCol] == player) {
            count++;
            if (count >= WIN_LENGTH) return true;
        } else {
            count = 0;
        }
        startRow++;
        startCol++;
    }
    
    // Check diagonal (top-right to bottom-left)
    count = 0;
    startRow = row - min(row, BOARD_WIDTH - 1 - col);
    startCol = col + min(row, BOARD_WIDTH - 1 - col);
    while (startRow < BOARD_HEIGHT && startCol >= 0) {
        if (board[startRow][startCol] == player) {
            count++;
            if (count >= WIN_LENGTH) return true;
        } else {
            count = 0;
        }
        startRow++;
        startCol--;
    }
    
    return false;
}

__device__ bool isValidMove(int board[BOARD_HEIGHT][BOARD_WIDTH], int col) {
    return col >= 0 && col < BOARD_WIDTH && board[0][col] == 0;
}

__device__ void makeMove(int board[BOARD_HEIGHT][BOARD_WIDTH], int col, int player) {
    for (int row = BOARD_HEIGHT - 1; row >= 0; row--) {
        if (board[row][col] == 0) {
            board[row][col] = player;
            break;
        }
    }
}

__device__ int evaluateBoard(int board[BOARD_HEIGHT][BOARD_WIDTH], int player) {
    int score = 0;
    
    // Evaluate horizontal lines
    for (int row = 0; row < BOARD_HEIGHT; row++) {
        for (int col = 0; col <= BOARD_WIDTH - WIN_LENGTH; col++) {
            int count = 0;
            int empty = 0;
            for (int i = 0; i < WIN_LENGTH; i++) {
                if (board[row][col + i] == player) count++;
                else if (board[row][col + i] == 0) empty++;
            }
            if (count + empty == WIN_LENGTH) {
                score += count * count;
            }
        }
    }
    
    // Evaluate vertical lines
    for (int row = 0; row <= BOARD_HEIGHT - WIN_LENGTH; row++) {
        for (int col = 0; col < BOARD_WIDTH; col++) {
            int count = 0;
            int empty = 0;
            for (int i = 0; i < WIN_LENGTH; i++) {
                if (board[row + i][col] == player) count++;
                else if (board[row + i][col] == 0) empty++;
            }
            if (count + empty == WIN_LENGTH) {
                score += count * count;
            }
        }
    }
    
    return score;
}

__device__ int getAvailableMoves(int board[BOARD_HEIGHT][BOARD_WIDTH], int moves[BOARD_WIDTH]) {
    int count = 0;
    for (int col = 0; col < BOARD_WIDTH; col++) {
        if (isValidMove(board, col)) {
            moves[count++] = col;
        }
    }
    return count;
}

// Host functions
void printBoard(const GameState& state) {
    printf("\n");
    for (int row = 0; row < BOARD_HEIGHT; row++) {
        printf("|");
        for (int col = 0; col < BOARD_WIDTH; col++) {
            if (state.board[row][col] == 0) {
                printf(" ");
            } else if (state.board[row][col] == 1) {
                printf("X");
            } else {
                printf("O");
            }
            printf("|");
        }
        printf("\n");
    }
    printf("+");
    for (int col = 0; col < BOARD_WIDTH; col++) {
        printf("--");
    }
    printf("+\n");
    for (int col = 0; col < BOARD_WIDTH; col++) {
        printf(" %d", col);
    }
    printf("\n\n");
}

bool checkWinHost(const GameState& state, int row, int col, int player) {
    // Check horizontal
    int count = 0;
    for (int c = 0; c < BOARD_WIDTH; c++) {
        if (state.board[row][c] == player) {
            count++;
            if (count >= WIN_LENGTH) return true;
        } else {
            count = 0;
        }
    }
    
    // Check vertical
    count = 0;
    for (int r = 0; r < BOARD_HEIGHT; r++) {
        if (state.board[r][col] == player) {
            count++;
            if (count >= WIN_LENGTH) return true;
        } else {
            count = 0;
        }
    }
    
    // Check diagonals
    // Top-left to bottom-right
    count = 0;
    int startRow = row - std::min(row, col);
    int startCol = col - std::min(row, col);
    while (startRow < BOARD_HEIGHT && startCol < BOARD_WIDTH) {
        if (state.board[startRow][startCol] == player) {
            count++;
            if (count >= WIN_LENGTH) return true;
        } else {
            count = 0;
        }
        startRow++;
        startCol++;
    }
    
    // Top-right to bottom-left
    count = 0;
    startRow = row - std::min(row, BOARD_WIDTH - 1 - col);
    startCol = col + std::min(row, BOARD_WIDTH - 1 - col);
    while (startRow < BOARD_HEIGHT && startCol >= 0) {
        if (state.board[startRow][startCol] == player) {
            count++;
            if (count >= WIN_LENGTH) return true;
        } else {
            count = 0;
        }
        startRow++;
        startCol--;
    }
    
    return false;
}

void makeMoveHost(GameState& state, int col, int player) {
    for (int row = BOARD_HEIGHT - 1; row >= 0; row--) {
        if (state.board[row][col] == 0) {
            state.board[row][col] = player;
            state.lastMove = col;
            break;
        }
    }
}

int getBestMoveGPU1(const GameState& state) {
    int* d_board, *d_scores;
    
    CUDA_CHECK(cudaMalloc(&d_board, BOARD_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scores, BOARD_WIDTH * sizeof(int)));
    
    // Copy board to device
    int flatBoard[BOARD_SIZE];
    for (int i = 0; i < BOARD_HEIGHT; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            flatBoard[i * BOARD_WIDTH + j] = state.board[i][j];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_board, flatBoard, BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel
    evaluationKernel<<<(BOARD_WIDTH + 255) / 256, 256>>>(d_board, d_scores, state.currentPlayer);
    
    // Get results
    int scores[BOARD_WIDTH];
    CUDA_CHECK(cudaMemcpy(scores, d_scores, BOARD_WIDTH * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Find best move
    int bestMove = 0;
    int bestScore = scores[0];
    for (int i = 1; i < BOARD_WIDTH; i++) {
        if (scores[i] > bestScore) {
            bestScore = scores[i];
            bestMove = i;
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_board));
    CUDA_CHECK(cudaFree(d_scores));
    
    return bestMove;
}

int getBestMoveGPU2(const GameState& state) {
    int* d_board, *d_visitCount, *d_winCount;
    
    CUDA_CHECK(cudaMalloc(&d_board, BOARD_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visitCount, BOARD_WIDTH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_winCount, BOARD_WIDTH * sizeof(int)));
    
    // Copy board to device
    int flatBoard[BOARD_SIZE];
    for (int i = 0; i < BOARD_HEIGHT; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            flatBoard[i * BOARD_WIDTH + j] = state.board[i][j];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_board, flatBoard, BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    
    // Initialize arrays
    int visitCount[BOARD_WIDTH] = {0};
    int winCount[BOARD_WIDTH] = {0};
    CUDA_CHECK(cudaMemcpy(d_visitCount, visitCount, BOARD_WIDTH * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_winCount, winCount, BOARD_WIDTH * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel
    mctsKernel<<<(MCTS_ITERATIONS + 255) / 256, 256>>>(d_board, d_visitCount, 
                                                       d_winCount, MCTS_ITERATIONS, state.currentPlayer);
    
    // Get results
    CUDA_CHECK(cudaMemcpy(visitCount, d_visitCount, BOARD_WIDTH * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(winCount, d_winCount, BOARD_WIDTH * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Find best move based on win rate
    int bestMove = 0;
    float bestRate = 0.0f;
    for (int i = 0; i < BOARD_WIDTH; i++) {
        if (visitCount[i] > 0) {
            float rate = (float)winCount[i] / visitCount[i];
            if (rate > bestRate) {
                bestRate = rate;
                bestMove = i;
            }
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_board));
    CUDA_CHECK(cudaFree(d_visitCount));
    CUDA_CHECK(cudaFree(d_winCount));
    
    return bestMove;
}

void saveGameReplay(const char* filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        printf("Error: Could not open file for writing\n");
        return;
    }
    
    file << "Connect 4 Game Replay\n";
    file << "=====================\n\n";
    
    for (size_t i = 0; i < gameMoves.size(); i++) {
        file << "Move " << (i + 1) << ":\n";
        file << "Player: " << (gameMoves[i].player == 1 ? "X (GPU1-Evaluation)" : "O (GPU2-MCTS)") << "\n";
        file << "Column: " << gameMoves[i].column << "\n";
        file << "GPU1 Time: " << gameMoves[i].gpu1Time << " ms\n";
        file << "GPU2 Time: " << gameMoves[i].gpu2Time << " ms\n";
        file << "Winner: " << gameMoves[i].winner << "\n\n";
        
        // Print board state
        const GameState& state = gameStates[i];
        for (int row = 0; row < BOARD_HEIGHT; row++) {
            file << "|";
            for (int col = 0; col < BOARD_WIDTH; col++) {
                if (state.board[row][col] == 0) file << " ";
                else if (state.board[row][col] == 1) file << "X";
                else file << "O";
                file << "|";
            }
            file << "\n";
        }
        file << "+";
        for (int col = 0; col < BOARD_WIDTH; col++) file << "--";
        file << "+\n";
        for (int col = 0; col < BOARD_WIDTH; col++) file << " " << col;
        file << "\n\n";
    }
    
    file.close();
    printf("Game replay saved to %s\n", filename);
}

int main() {
    printf("Connect 4 GPU Game\n");
    printf("==================\n");
    printf("GPU 1: Evaluation-based Strategy\n");
    printf("GPU 2: Monte Carlo Tree Search\n\n");
    
    // Initialize game state
    GameState state;
    memset(&state, 0, sizeof(state));
    state.currentPlayer = 1;
    
    // Clear move history
    gameMoves.clear();
    gameStates.clear();
    
    printf("Starting game...\n");
    printBoard(state);
    
    // Game loop
    while (state.moveCount < MAX_MOVES) {
        printf("Player %d's turn (%s)\n", state.currentPlayer, 
               state.currentPlayer == 1 ? "GPU1-Evaluation" : "GPU2-MCTS");
        
        // Save current state
        gameStates.push_back(state);
        
        // Get move from appropriate GPU
        int move;
        float gpu1Time = 0.0f, gpu2Time = 0.0f;
        
        if (state.currentPlayer == 1) {
            // GPU 1's turn
            clock_t start = clock();
            move = getBestMoveGPU1(state);
            clock_t end = clock();
            gpu1Time = ((float)(end - start)) / CLOCKS_PER_SEC * 1000.0f;
            gpu2Time = 0.0f;
        } else {
            // GPU 2's turn
            clock_t start = clock();
            move = getBestMoveGPU2(state);
            clock_t end = clock();
            gpu2Time = ((float)(end - start)) / CLOCKS_PER_SEC * 1000.0f;
            gpu1Time = 0.0f;
        }
        
        printf("GPU%d chose column %d (Time: %.2f ms)\n", state.currentPlayer, move, 
               state.currentPlayer == 1 ? gpu1Time : gpu2Time);
        
        // Make the move
        makeMoveHost(state, move, state.currentPlayer);
        state.moveCount++;
        
        // Record the move
        Move moveRecord;
        moveRecord.column = move;
        moveRecord.player = state.currentPlayer;
        moveRecord.gpu1Time = gpu1Time;
        moveRecord.gpu2Time = gpu2Time;
        moveRecord.winner = 0;
        
        // Check for win
        int row = 0;
        for (int r = BOARD_HEIGHT - 1; r >= 0; r--) {
            if (state.board[r][move] == state.currentPlayer) {
                row = r;
                break;
            }
        }
        
        if (checkWinHost(state, row, move, state.currentPlayer)) {
            printf("\nPlayer %d (%s) wins!\n", state.currentPlayer,
                   state.currentPlayer == 1 ? "GPU1-Evaluation" : "GPU2-MCTS");
            moveRecord.winner = state.currentPlayer;
            gameMoves.push_back(moveRecord);
            printBoard(state);
            break;
        }
        
        // Check for draw
        if (state.moveCount == MAX_MOVES) {
            printf("\nGame is a draw!\n");
            moveRecord.winner = 0;
            gameMoves.push_back(moveRecord);
            printBoard(state);
            break;
        }
        
        gameMoves.push_back(moveRecord);
        printBoard(state);
        
        // Switch players
        state.currentPlayer = state.currentPlayer == 1 ? 2 : 1;
    }
    
    // Save game replay
    saveGameReplay("game_replay.txt");
    
    printf("\nGame completed! Replay saved to game_replay.txt\n");
    printf("Total moves: %d\n", (int)gameMoves.size());
    
    return 0;
} 