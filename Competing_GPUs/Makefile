# Makefile for Connect 4 GPU Game

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_60
CXX = g++
CXX_FLAGS = -std=c++11 -O3

# Target executable
TARGET = connect4_gpu

# Source files
CUDA_SOURCES = connect4_gpu.cu
PYTHON_SCRIPT = game_visualizer.py

# Default target
all: $(TARGET)

# Compile CUDA program
$(TARGET): $(CUDA_SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Run the game
run: $(TARGET)
	./$(TARGET)

# Run the visualizer
visualize: $(PYTHON_SCRIPT)
	python3 $(PYTHON_SCRIPT)

# Clean build files
clean:
	rm -f $(TARGET) *.o game_replay.txt

# Install Python dependencies (if needed)
install-deps:
	pip3 install tkinter

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "CUDA not found. Please install CUDA Toolkit."
	@nvidia-smi || echo "NVIDIA GPU not found or drivers not installed."

# Run multiple games
test-games: $(TARGET)
	@echo "Running 5 test games..."
	@for i in 1 2 3 4 5; do \
		echo "Game $$i:"; \
		./$(TARGET); \
		echo "-------------------"; \
	done

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the CUDA program"
	@echo "  run          - Run a single game"
	@echo "  visualize    - Run the game visualizer"
	@echo "  test-games   - Run 5 test games"
	@echo "  check-cuda   - Check CUDA installation"
	@echo "  install-deps - Install Python dependencies"
	@echo "  clean        - Remove build files"
	@echo "  help         - Show this help message"

.PHONY: all run visualize clean install-deps check-cuda test-games help 