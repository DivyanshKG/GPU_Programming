#pragma once
#include <iostream>

// Compare two matrices for correctness
bool compareMatrices(const float* A, const float* B, int N, float epsilon = 1e-3f) {
    for (int i = 0; i < N * N; ++i) {
        if (fabs(A[i] - B[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": " << A[i] << " vs " << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Print a matrix (for debugging, small N)
void printMatrix(const float* mat, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << mat[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
} 