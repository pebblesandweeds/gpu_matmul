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

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty * TM;
    int col = bx * BLOCK_SIZE + tx;

    float threadResults[TM] = {0.0f};

    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Combined loading of A and B
        for (int i = 0; i < TM; ++i) {
            int shared_idx = ty * TM + i;
            if (row + i < n && tile * BLOCK_SIZE + tx < n) {
                shared_a[shared_idx][tx] = A[(row + i) * n + tile * BLOCK_SIZE + tx];
                shared_b[shared_idx][tx] = B[(tile * BLOCK_SIZE + shared_idx) * n + col];
            } else {
                shared_a[shared_idx][tx] = 0.0f;
                shared_b[shared_idx][tx] = 0.0f;
            }
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

void init_matrix(float *matrix, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
        matrix[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * 2.0f - 1.0f;
    }
}
