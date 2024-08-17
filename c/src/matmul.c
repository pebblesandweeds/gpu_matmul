#include <stdlib.h>
#include <hip/hip_runtime.h>
#include "matmul.h"

void init_matrix(float *matrix, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
        matrix[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * 2.0f - 1.0f;
    }
}

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

void naive_matmul(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 blockDim = {16, 16, 1};
    dim3 gridDim = {(N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, 1};
    hipLaunchKernelGGL(matmul_kernel, gridDim, blockDim, 0, NULL, d_A, d_B, d_C, N);
    CHECK(hipGetLastError());
}
