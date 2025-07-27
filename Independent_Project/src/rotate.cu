#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Simple PGM reader (grayscale, ASCII P2 or binary P5)
unsigned char* read_pgm(const char* filename, int* width, int* height) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    char header[3];
    fscanf(f, "%2s", header);
    if (header[0] != 'P' || (header[1] != '5' && header[1] != '2')) { fclose(f); return NULL; }
    int w, h, maxval;
    fscanf(f, "%d %d %d", &w, &h, &maxval);
    fgetc(f); // skip single whitespace
    *width = w; *height = h;
    unsigned char* data = (unsigned char*)malloc(w * h);
    if (header[1] == '5') {
        fread(data, 1, w * h, f);
    } else {
        for (int i = 0; i < w * h; ++i) {
            int v; fscanf(f, "%d", &v); data[i] = (unsigned char)v;
        }
    }
    fclose(f);
    return data;
}

// Simple PGM writer (binary P5)
void write_pgm(const char* filename, unsigned char* data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height, f);
    fclose(f);
}

// CUDA kernel for 90 degree clockwise rotation
__global__ void rotate90(unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int in_idx = y * w + x;
        int out_idx = x * h + (h - 1 - y);
        out[out_idx] = in[in_idx];
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s input.pgm output.pgm\n", argv[0]);
        return 1;
    }
    int w, h;
    unsigned char* img = read_pgm(argv[1], &w, &h);
    if (!img) { printf("Failed to read image\n"); return 1; }
    unsigned char* out = (unsigned char*)malloc(w * h);
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, w * h);
    cudaMalloc(&d_out, w * h);
    cudaMemcpy(d_in, img, w * h, cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    rotate90<<<grid, block>>>(d_in, d_out, w, h);
    cudaMemcpy(out, d_out, w * h, cudaMemcpyDeviceToHost);
    write_pgm(argv[2], out, h, w); // Note: width and height swapped
    cudaFree(d_in); cudaFree(d_out);
    free(img); free(out);
    printf("Rotated image written to %s\n", argv[2]);
    return 0;
} 