#ifndef MATMUL_H
#define MATMUL_H

#include <hip/hip_runtime.h>

#define CHECK(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            fprintf(stderr, "Error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void matmul_kernel(const float* A, const float* B, float* C, int n);
__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, int n);
void init_matrix(float *matrix, int n);

#endif // MATMUL_H
