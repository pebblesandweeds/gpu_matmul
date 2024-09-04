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

    int threadCol = hipThreadIdx_x % (BLOCK_SIZE / thread_multiplier);
    int threadRow = hipThreadIdx_x / (BLOCK_SIZE / thread_multiplier);

    __shared__ float shared_a[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE * BLOCK_SIZE];

    float threadResults[thread_multiplier * thread_multiplier] = {0.0f};
    float regM[thread_multiplier] = {0.0f};
    float regN[thread_multiplier] = {0.0f};

    int threads_per_block = BLOCK_SIZE * BLOCK_SIZE / (thread_multiplier * thread_multiplier);
    int stride = threads_per_block / BLOCK_SIZE;

    A += hipBlockIdx_y * BLOCK_SIZE * n;
    B += hipBlockIdx_x * BLOCK_SIZE;
    C += hipBlockIdx_y * BLOCK_SIZE * n + hipBlockIdx_x * BLOCK_SIZE;

    for (int block_index = 0; block_index < n; block_index += BLOCK_SIZE) {
        for (int i = 0; i < BLOCK_SIZE; i += stride) {
            int row = hipThreadIdx_x / BLOCK_SIZE;
            int col = hipThreadIdx_x % BLOCK_SIZE;
            shared_a[(row + i) * BLOCK_SIZE + col] = A[(row + i) * n + col];
            shared_b[(row + i) * BLOCK_SIZE + col] = B[(row + i) * n + col];
        }

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * n;

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            for (int i = 0; i < thread_multiplier; ++i) {
                regM[i] = shared_a[(threadRow * thread_multiplier + i) * BLOCK_SIZE + k];
                regN[i] = shared_b[k * BLOCK_SIZE + threadCol * thread_multiplier + i];
            }
            for (int rm = 0; rm < thread_multiplier; ++rm) {
                for (int rn = 0; rn < thread_multiplier; ++rn) {
                    threadResults[rm * thread_multiplier + rn] += regM[rm] * regN[rn];
                }
            }
        }
        __syncthreads();

    }

    for (int rm = 0; rm < thread_multiplier; ++rm) {
        for (int rn = 0; rn < thread_multiplier; ++rn) {
            C[(threadRow * thread_multiplier + rm) * n + threadCol * thread_multiplier + rn] = threadResults[rm * thread_multiplier + rn];
        }
    }
}

void init_matrix(float *matrix, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
        matrix[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * 2.0f - 1.0f;
    }
}
