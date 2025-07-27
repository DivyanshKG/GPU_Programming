# Makefile for CUDA Matrix Multiplication Project (Windows)

# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -O2
NVCCFLAGS = -O2

# Targets
all: matrix_mul cpu_matrix_mul

matrix_mul: src/main.cu
	$(NVCC) $(NVCCFLAGS) -o matrix_mul src/main.cu

cpu_matrix_mul: src/cpu_matrix_mult.cpp
	$(CXX) $(CXXFLAGS) -o cpu_matrix_mul src/cpu_matrix_mult.cpp

clean:
	del /Q matrix_mul.exe cpu_matrix_mul.exe 