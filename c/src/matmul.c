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
    __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE * TILE_SIZE];

    int row = hipBlockIdx_y * BLOCK_SIZE + hipThreadIdx_y;
    int col = hipBlockIdx_x * BLOCK_SIZE * TILE_SIZE + hipThreadIdx_x;

    float sum[TILE_SIZE] = {0.0f};

    for (int t = 0; t < n; t += BLOCK_SIZE) {
        // Load tile_A
        if (row < n && t + hipThreadIdx_x < n) {
            tile_A[hipThreadIdx_y][hipThreadIdx_x] = A[row * n + t + hipThreadIdx_x];
        } else {
            tile_A[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;
        }

        // Load tile_B
        for (int i = 0; i < TILE_SIZE; i++) {
            if (t + hipThreadIdx_y < n && col + i * BLOCK_SIZE < n) {
                tile_B[hipThreadIdx_y][hipThreadIdx_x + i * BLOCK_SIZE] = B[(t + hipThreadIdx_y) * n + col + i * BLOCK_SIZE];
            } else {
                tile_B[hipThreadIdx_y][hipThreadIdx_x + i * BLOCK_SIZE] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial sums
        for (int k = 0; k < BLOCK_SIZE; k++) {
            for (int i = 0; i < TILE_SIZE; i++) {
                sum[i] += tile_A[hipThreadIdx_y][k] * tile_B[k][hipThreadIdx_x + i * BLOCK_SIZE];
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    for (int i = 0; i < TILE_SIZE; i++) {
        if (row < n && col + i * BLOCK_SIZE < n) {
            C[row * n + col + i * BLOCK_SIZE] = sum[i];
        }
    }
}

void init_matrix(float *matrix, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
        matrix[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * 2.0f - 1.0f;
    }
}
