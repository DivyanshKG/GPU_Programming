#!/bin/bash
set -e
mkdir -p bin
for img in data/*.pgm; do
  name=$(basename "$img" .pgm)
  ./bin/rotate "$img" "bin/rotated_${name}.pgm"
done 