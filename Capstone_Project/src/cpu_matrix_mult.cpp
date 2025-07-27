#include <iostream>
#include <vector>
#include <chrono>

void cpuMatrixMultiply(const float* A, const float* B, float* C, int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * N * sizeof(float);
    std::vector<float> A(N * N), B(N * N), C(N * N);

    // Initialize matrices
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cpuMatrixMultiply(A.data(), B.data(), C.data(), N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "CPU matrix multiplication completed in " << diff.count() << " seconds." << std::endl;
    return 0;
} 