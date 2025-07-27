#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include "utils.h"
#include <fstream>

#define TILE_SIZE 32

// CUDA kernel for matrix multiplication with shared memory tiling
__global__ void matMulKernel(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N)
        C[row * N + col] = sum;
}

// Host function to initialize matrices
void initializeMatrix(float* mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    const int N = 1024; // Matrix size N x N
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    float* h_C_ref = (float*)malloc(bytes);

    // Initialize matrices
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // CPU reference for correctness and timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += h_A[row * N + k] * h_B[k * N + col];
            }
            h_C_ref[row * N + col] = sum;
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_diff = cpu_end - cpu_start;

    // Check correctness
    bool correct = compareMatrices(h_C, h_C_ref, N);
    std::cout << "\nCUDA matrix multiplication completed in " << ms / 1000.0 << " seconds." << std::endl;
    std::cout << "CPU matrix multiplication completed in " << cpu_diff.count() << " seconds." << std::endl;
    std::cout << "Result is " << (correct ? "CORRECT" : "INCORRECT") << std::endl;

    // Log results to CSV
    std::ofstream log("results/results.csv", std::ios::app);
    log << N << "," << ms / 1000.0 << "," << cpu_diff.count() << "," << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    log.close();

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
} 