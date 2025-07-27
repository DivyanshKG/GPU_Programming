# CUDA Image Rotation (Minimal)

This project demonstrates simple image rotation (90° clockwise) using CUDA, with no external libraries. It processes grayscale PGM images.

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc)
- Bash shell (for run.sh)

## Build
```sh
make
```

## Usage
Place your `.pgm` images in the `data/` directory. Then run:
```sh
./run.sh
```
This will rotate all images in `data/` and save the results as `bin/rotated_<originalname>.pgm`.

You can also run the program manually:
```sh
./bin/rotate input.pgm output.pgm
```

## Notes
- Only binary (P5) or ASCII (P2) PGM images are supported.
- Output images are always in binary (P5) format.
