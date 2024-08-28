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
    int block_row = hipBlockIdx_y;
    int block_col = hipBlockIdx_x;

    __shared__ float a_shared[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE * BLOCK_SIZE];

    int threadCol = hipThreadIdx_x % (BLOCK_SIZE / TM);
    int threadRow = hipThreadIdx_x / (BLOCK_SIZE / TM);

    int warpId = hipThreadIdx_x / 32;
    int laneId = hipThreadIdx_x % 32;

    A += block_row * BLOCK_SIZE * n;
    B += block_col * BLOCK_SIZE;
    C += block_row * BLOCK_SIZE * n + block_col * BLOCK_SIZE;

    float results[TM] = {0.0f};

    for (int shared_block_idx = 0; shared_block_idx < n; shared_block_idx += BLOCK_SIZE) {
        int row_a_base = (warpId * TM) % BLOCK_SIZE;
        int col_b_base = (warpId * TM) % BLOCK_SIZE;

        for (int i = 0; i < TM; i++) {
            int row_a = (row_a_base + i) % BLOCK_SIZE;
            int col_b = (col_b_base + i) % BLOCK_SIZE;
            a_shared[row_a * BLOCK_SIZE + laneId] = A[row_a * n + laneId];
            b_shared[col_b * BLOCK_SIZE + laneId] = B[col_b * n + laneId];
        }

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * n;

        for (int dot_product_idx = 0; dot_product_idx < BLOCK_SIZE; ++dot_product_idx) {
            float a_val = a_shared[threadRow * BLOCK_SIZE + dot_product_idx];
            float b_cache = b_shared[dot_product_idx * BLOCK_SIZE + threadCol * TM];

            for (int t = 0; t < TM; ++t) {
                results[t] += a_val * b_cache;
                b_cache = b_shared[dot_product_idx * BLOCK_SIZE + threadCol * TM + t + 1];
            }
        }

        __syncthreads();

    }

    for (int t = 0; t < TM; ++t) {
        C[threadRow * n + threadCol * TM + t] = results[t];
    }
}

void init_matrix(float *matrix, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
        matrix[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * 2.0f - 1.0f;
    }
}
