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
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE * TILE_SIZE];

    const uint cRow = hipBlockIdx_y;
    const uint cCol = hipBlockIdx_x;
    const int threadCol = hipThreadIdx_x % BLOCK_SIZE;
    const int threadRow = hipThreadIdx_x / BLOCK_SIZE;

    A += cRow * BLOCK_SIZE * n;
    B += cCol * BLOCK_SIZE * TILE_SIZE;
    C += cRow * BLOCK_SIZE * n + cCol * BLOCK_SIZE * TILE_SIZE;

    float threadResults[TILE_SIZE] = {0.0f};

    for (int t = 0; t < n; t += BLOCK_SIZE) {
        // Load A into shared memory
        if (cRow * BLOCK_SIZE + threadRow < n && t + threadCol < n)
            As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * n + threadCol];
        else
            As[threadRow * BLOCK_SIZE + threadCol] = 0.0f;

        // Load B into shared memory
        for (int i = 0; i < TILE_SIZE; ++i) {
            if (t + threadRow < n && cCol * BLOCK_SIZE * TILE_SIZE + threadCol + i * BLOCK_SIZE < n)
                Bs[threadRow * BLOCK_SIZE * TILE_SIZE + threadCol + i * BLOCK_SIZE] =
                    B[threadRow * n + threadCol + i * BLOCK_SIZE];
            else
                Bs[threadRow * BLOCK_SIZE * TILE_SIZE + threadCol + i * BLOCK_SIZE] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float tmpA = As[threadRow * BLOCK_SIZE + k];
            for (int i = 0; i < TILE_SIZE; ++i) {
                threadResults[i] += tmpA * Bs[k * BLOCK_SIZE * TILE_SIZE + threadCol + i * BLOCK_SIZE];
            }
        }

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * n;
    }

    // Write results to global memory
    for (int i = 0; i < TILE_SIZE; ++i) {
        int row = cRow * BLOCK_SIZE + threadRow;
        int col = cCol * BLOCK_SIZE * TILE_SIZE + threadCol + i * BLOCK_SIZE;
        if (row < n && col < n) {
            C[threadRow * n + threadCol + i * BLOCK_SIZE] = threadResults[i];
        }
    }
}

void init_matrix(float *matrix, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
        matrix[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * 2.0f - 1.0f;
    }
}
