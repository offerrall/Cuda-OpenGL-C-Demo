# Cuda-OpenGL-C-Demo

This repository serves as a reference for setting up interoperability between CUDA, OpenGL, and SDL in C++.

## Description

This project includes a simple demonstration using a Pixel Buffer Object (PBO) to transfer data between CUDA and OpenGL and display it in an SDL window. It implements Conway's Game of Life, using CUDA to compute the state of each cell.

## Key Files

- **main.cpp**: Sets up SDL and OpenGL, initializes the CUDA context, and runs the main loop.
- **effect.cu**: Defines the CUDA kernel to update the state of each cell in the Game of Life.
- **build.py**: Script for compiling the project with `nvcc`, specifying the paths for CUDA, SDL, and GLEW libraries.

## Requirements

- NVIDIA CUDA Toolkit (v12.6 or compatible)
- SDL2
- GLEW

## Compilation

Run the following command to compile the project (ensure library paths are correctly set in `build.py`):
```bash
python build.py
