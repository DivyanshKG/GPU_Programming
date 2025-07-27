# Simple Makefile for CUDA image rotation

SRC=src/rotate.cu
BIN=bin/rotate

all: $(BIN)

$(BIN): $(SRC)
	mkdir -p bin
	nvcc -o $(BIN) $(SRC)

clean:
	rm -f $(BIN) 