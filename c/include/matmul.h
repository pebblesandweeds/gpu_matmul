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

void init_matrix(float *matrix, int n);
void naive_matmul(const float* d_A, const float* d_B, float* d_C, int N);

#endif // MATMUL_H
