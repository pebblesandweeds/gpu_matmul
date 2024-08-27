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
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    float threadResults[TM] = {0.0f};

    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Loading of A and B
        for (int i = 0; i < BLOCK_SIZE; i += blockDim.y) {
            int shared_row = ty + i;
            if (row + i < n && tile * BLOCK_SIZE + tx < n) {
                shared_a[shared_row][tx] = A[(row + i) * n + tile * BLOCK_SIZE + tx];
                shared_b[shared_row][tx] = B[(tile * BLOCK_SIZE + shared_row) * n + col];
            } else {
                shared_a[shared_row][tx] = 0.0f;
                shared_b[shared_row][tx] = 0.0f;
            }
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float tmpB = shared_b[k][tx];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                threadResults[i] += shared_a[ty + i * blockDim.y][k] * tmpB;
            }
        }
        __syncthreads();
    }

    // Writing results
    for (int i = 0; i < TM; ++i) {
        int write_row = row + i * blockDim.y;
        if (write_row < n && col < n)
            C[write_row * n + col] = threadResults[i];
    }
}

void init_matrix(float *matrix, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
        matrix[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * 2.0f - 1.0f;
    }
}
