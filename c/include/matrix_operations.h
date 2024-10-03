#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <rocblas/rocblas.h>

void initialize_matrices(float *A, float *B, int n);
void transpose_matrix(const float *src, float *dst, int n);
void perform_matrix_multiplication(rocblas_handle handle, float *d_A, float *d_B, float *d_C, int N, int NUM_RUNS);

#endif // MATRIX_OPERATIONS_H
