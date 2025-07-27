#include <cuda_runtime.h>
#include <stdio.h>

// Simple CUDA kernel to test compilation
__global__ void testKernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2;
    }
}

int main() {
    printf("CUDA Connect 4 Test Program\n");
    printf("==========================\n\n");
    
    // Test CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("Error: No CUDA devices found!\n");
        return -1;
    }
    
    printf("Found %d CUDA device(s):\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  Device %d: %s\n", i, prop.name);
        printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("    Global Memory: %lu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("    Multiprocessors: %d\n", prop.multiProcessorCount);
    }
    
    // Test simple kernel execution
    printf("\nTesting kernel execution...\n");
    
    int n = 100;
    int *h_data = new int[n];
    int *d_data;
    
    cudaMalloc(&d_data, n * sizeof(int));
    
    testKernel<<<(n + 255) / 256, 256>>>(d_data, n);
    
    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify results
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (h_data[i] != i * 2) {
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("✓ Kernel execution successful!\n");
    } else {
        printf("✗ Kernel execution failed!\n");
    }
    
    // Cleanup
    cudaFree(d_data);
    delete[] h_data;
    
    printf("\nCUDA environment is ready for Connect 4 game!\n");
    
    return 0;
} 