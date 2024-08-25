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

__global__ void matmul_shared_kernel(const float *A, const float *B, float *C, int n) {
    __shared__ float shared_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE][BLOCK_SIZE];
    int bx = hipBlockIdx_x;
    int by = hipBlockIdx_y;
    int tx = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int row = by * BLOCK_SIZE + ty * TM;
    int col = bx * BLOCK_SIZE + tx;

    float threadResults[TM] = {0.0f};

    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        for (int i = 0; i < TM; ++i) {
            if (row + i < n && tile * BLOCK_SIZE + tx < n)
                shared_a[ty * TM + i][tx] = A[(row + i) * n + tile * BLOCK_SIZE + tx];
            else
                shared_a[ty * TM + i][tx] = 0.0f;
        }

        for (int i = 0; i < TM; ++i) {
            if (tile * BLOCK_SIZE + ty * TM + i < n && col < n)
                shared_b[ty * TM + i][tx] = B[(tile * BLOCK_SIZE + ty * TM + i) * n + col];
            else
                shared_b[ty * TM + i][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                threadResults[i] += shared_a[ty * TM + i][k] * shared_b[k][tx];
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        if (row + i < n && col < n)
            C[(row + i) * n + col] = threadResults[i];
    }
}

__global__ void matmul_complex_kernel(const float* A, const float* B, float* C, int n) {
    __shared__ float shared_a[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE * BLOCK_SIZE * TILE_SIZE];

    const int block_row = hipBlockIdx_y;
    const int block_col = hipBlockIdx_x;
    const int thread_col = hipThreadIdx_x % BLOCK_SIZE;
    const int thread_row = hipThreadIdx_x / BLOCK_SIZE;

    A += block_row * BLOCK_SIZE * n;
    B += block_col * BLOCK_SIZE * TILE_SIZE;
    C += block_row * BLOCK_SIZE * n + block_col * BLOCK_SIZE * TILE_SIZE;

    float thread_results[TILE_SIZE] = {0.0f};

    for (int t = 0; t < n; t += BLOCK_SIZE) {
        // Load A into shared memory
        if (block_row * BLOCK_SIZE + thread_row < n && t + thread_col < n)
            shared_a[thread_row * BLOCK_SIZE + thread_col] = A[thread_row * n + thread_col];
        else
            shared_a[thread_row * BLOCK_SIZE + thread_col] = 0.0f;

        // Load B into shared memory
        for (int i = 0; i < TILE_SIZE; ++i) {
            if (t + thread_row < n && block_col * BLOCK_SIZE * TILE_SIZE + thread_col + i * BLOCK_SIZE < n)
                shared_b[thread_row * BLOCK_SIZE * TILE_SIZE + thread_col + i * BLOCK_SIZE] =
                    B[thread_row * n + thread_col + i * BLOCK_SIZE];
            else
                shared_b[thread_row * BLOCK_SIZE * TILE_SIZE + thread_col + i * BLOCK_SIZE] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float tmpA = shared_a[thread_row * BLOCK_SIZE + k];
            for (int i = 0; i < TILE_SIZE; ++i) {
                thread_results[i] += tmpA * shared_b[k * BLOCK_SIZE * TILE_SIZE + thread_col + i * BLOCK_SIZE];
            }
        }

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * n;
    }

    // Write results to global memory
    for (int i = 0; i < TILE_SIZE; ++i) {
        int row = block_row * BLOCK_SIZE + thread_row;
        int col = block_col * BLOCK_SIZE * TILE_SIZE + thread_col + i * BLOCK_SIZE;
        if (row < n && col < n) {
            C[thread_row * n + thread_col + i * BLOCK_SIZE] = thread_results[i];
        }
    }
}

void init_matrix(float *matrix, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
        matrix[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * 2.0f - 1.0f;
    }
}
