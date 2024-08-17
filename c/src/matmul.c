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

void init_matrix(float *matrix, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
        matrix[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * 2.0f - 1.0f;
    }
}
