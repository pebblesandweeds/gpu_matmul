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

__global__ void matmul_scalar_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {
    int threadCol = hipThreadIdx_x;
    int threadRow = hipThreadIdx_y;
    __shared__ float shared_a[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE * BLOCK_SIZE];
    float result[TILE_SIZE][TILE_SIZE] = {{0.0f}};
    int row = hipBlockIdx_y * BLOCK_SIZE + threadRow * TILE_SIZE;
    int col = hipBlockIdx_x * BLOCK_SIZE + threadCol * TILE_SIZE;

    A += row * n;
    B += col;
    C += row * n + col;

    for (int block_index = 0; block_index < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++block_index) {
        for (int i = 0; i < TILE_SIZE; ++i) {
            for (int j = 0; j < TILE_SIZE; ++j) {
                int sharedIdx = threadRow * TILE_SIZE * BLOCK_SIZE + threadCol * TILE_SIZE + i * BLOCK_SIZE + j;
                int globalColA = threadCol * TILE_SIZE + j;
                if (row + i < n && block_index * BLOCK_SIZE + globalColA < n) {
                    shared_a[sharedIdx] = A[i * n + globalColA];
                } else {
                    shared_a[sharedIdx] = 0.0f;
                }
                int globalRowB = threadRow * TILE_SIZE + i;
                if (block_index * BLOCK_SIZE + globalRowB < n && col + j < n) {
                    shared_b[sharedIdx] = B[globalRowB * n + j];
                } else {
                    shared_b[sharedIdx] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            for (int i = 0; i < TILE_SIZE; ++i) {
                for (int j = 0; j < TILE_SIZE; ++j) {
                    result[i][j] += shared_a[(threadRow * TILE_SIZE + i) * BLOCK_SIZE + k] *
                                    shared_b[k * BLOCK_SIZE + threadCol * TILE_SIZE + j];
                }
            }
        }
        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * n;
    }

    for (int i = 0; i < TILE_SIZE; ++i) {
        for (int j = 0; j < TILE_SIZE; ++j) {
            if (row + i < n && col + j < n) {
                C[i * n + j] = result[i][j];
            }
        }
    }
}

void init_matrix(float *matrix, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
        matrix[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * 2.0f - 1.0f;
    }
}
