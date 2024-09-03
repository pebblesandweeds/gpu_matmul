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

__global__ void matmul_scalar_kernel(const float* A, const float* B, float* C, int n) {
    const int TILE_SIZE = 4;
    __shared__ float shared_a[BLOCK_SIZE * TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE * BLOCK_SIZE];
    int blockRow = hipBlockIdx_y;
    int blockColumn = hipBlockIdx_x;
    int threadRow = hipThreadIdx_x / BLOCK_SIZE;
    int threadColumn = hipThreadIdx_x % BLOCK_SIZE;

    A += blockRow * BLOCK_SIZE * n;
    B += blockColumn * BLOCK_SIZE;
    C += blockRow * BLOCK_SIZE * n + blockColumn * BLOCK_SIZE;

    float threadResults[thread_multiplier] = {0.0};

    for (int bkIdx = 0; bkIdx < n; bkIdx += TILE_SIZE) {
        shared_a[hipThreadIdx_x] = A[hipThreadIdx_x % TILE_SIZE + (hipThreadIdx_x / TILE_SIZE) * n];
        shared_b[hipThreadIdx_x] = B[(hipThreadIdx_x / BLOCK_SIZE) * n + hipThreadIdx_x % BLOCK_SIZE];
        __syncthreads();

        A += TILE_SIZE;
        B += TILE_SIZE * n;

        for (int k = 0; k < TILE_SIZE; ++k) {
            float tmpB = shared_b[k * BLOCK_SIZE + threadColumn];
            for (int resIdx = 0; resIdx < thread_multiplier; ++resIdx) {
                threadResults[resIdx] +=
                    shared_a[(threadRow * thread_multiplier + resIdx) * TILE_SIZE + k] * tmpB;
            }
        }
        __syncthreads();
    }

    for (int resIdx = 0; resIdx < thread_multiplier; ++resIdx) {
        C[(threadRow * thread_multiplier + resIdx) * n + threadColumn] = threadResults[resIdx];
    }
}

void init_matrix(float *matrix, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
        matrix[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * 2.0f - 1.0f;
    }
}
