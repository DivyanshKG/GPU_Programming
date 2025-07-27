# CUDA Matrix Multiplication Capstone Project

## Overview
This project implements large matrix multiplication using a custom CUDA kernel, demonstrating the power of GPU acceleration. It includes both a GPU (CUDA) and CPU implementation for benchmarking and comparison.

## Features
- CUDA kernel for matrix multiplication (with shared memory optimization)
- CPU implementation for baseline comparison
- Performance benchmarking and correctness checks
- Easy-to-follow build and run instructions

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with CUDA 11+)
- C++ compiler (e.g., g++)
- (Optional) Make

## Getting Started
1. Clone this repository.
2. Build the project:
   ```sh
   make
   ```
3. Run the executable:
   ```sh
   ./matrix_mul
   ```

## Project Structure
- `src/` - Source code (CUDA and CPU)
- `data/` - (Optional) Input/output matrices
- `results/` - Output logs, performance data
- `README.md` - This file
- `Makefile` - Build instructions

## How GPU is Used
The core matrix multiplication is performed by a custom CUDA kernel, leveraging the parallelism of the GPU. Shared memory tiling is used for performance.

## Example Results
Results and performance benchmarks will be added in the `results/` folder.

## Windows Build Instructions (No Make)

If you do not have `make` installed, you can build the executables manually:

1. Open the **NVIDIA CUDA Command Prompt** (or ensure `nvcc` is in your PATH).
2. Build the CUDA version:
   ```sh
   nvcc -O2 -o matrix_mul.exe src/main.cu
   ```
3. Build the CPU version (requires g++):
   ```sh
   g++ -O2 -o cpu_matrix_mul.exe src/cpu_matrix_mult.cpp
   ```

## Running the Programs

- To run the CUDA version:
  ```sh
  matrix_mul.exe
  ```
- To run the CPU version:
  ```sh
  cpu_matrix_mul.exe
  ```

## Results
- The program will print timing and correctness information to the console.
- You can redirect output to a file for record-keeping:
  ```sh
  matrix_mul.exe > results\run1.txt
  ```

## Visualizing Results

To visualize performance (runtimes and speedup), first ensure you have Python and the required libraries:

```sh
pip install matplotlib pandas
```

Then run:

```sh
python visualize_results.py
```

This will generate plots in the `results/` folder and display them on screen.

--- 