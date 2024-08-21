#include <hip/hip_runtime.h>
#include "matmul.h"

__global__ void matmul_kernel(const float* A, const float* B, float* C, int n) {
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, int n) {
    __shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];
    int row = hipBlockIdx_y * BLOCK_SIZE + hipThreadIdx_y;
    int col = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;

    // Pointer advancement
    A += row * n;
    B += col;
    C += row * n + col;

    float sum = 0.0f;
    for (int t = 0; t < n; t += BLOCK_SIZE) {
        if (row < n && t + hipThreadIdx_x < n) {
            tile_A[hipThreadIdx_y][hipThreadIdx_x] = A[t + hipThreadIdx_x];
        } else {
            tile_A[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;
        }
        if (t + hipThreadIdx_y < n && col < n) {
            tile_B[hipThreadIdx_y][hipThreadIdx_x] = B[(t + hipThreadIdx_y) * n];
        } else {
            tile_B[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += tile_A[hipThreadIdx_y][k] * tile_B[k][hipThreadIdx_x];
        }
        __syncthreads();
    }
    if (row < n && col < n) {
        *C = sum;
    }
}

void init_matrix(float *matrix, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
        matrix[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * 2.0f - 1.0f;
    }
}
