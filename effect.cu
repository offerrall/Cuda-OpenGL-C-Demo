#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" void launch_cuda_effect(uchar4* d_output, int width, int height, int* d_current, int* d_next);

__global__ void game_of_life_kernel(uchar4* d_output, int width, int height, int* d_current, int* d_next) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;


    int live_neighbors = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                live_neighbors += d_current[ny * width + nx];
            }
        }
    }

    int cell = d_current[idx];
    d_next[idx] = (cell == 1 && (live_neighbors == 2 || live_neighbors == 3)) || (cell == 0 && live_neighbors == 3);

    d_output[idx] = d_next[idx] ? make_uchar4(255, 255, 255, 255) : make_uchar4(0, 0, 0, 255);
}

extern "C" void launch_cuda_effect(uchar4* d_output, int width, int height, int* d_current, int* d_next) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    game_of_life_kernel<<<gridSize, blockSize>>>(d_output, width, height, d_current, d_next);
}
