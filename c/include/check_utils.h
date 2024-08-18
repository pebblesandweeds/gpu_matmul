#ifndef CHECK_UTILS_H
#define CHECK_UTILS_H

#include <stdbool.h>

bool partial_verify(const float* A, const float* B, const float* C_gpu, int N, float tolerance);
double compute_trace_of_product(const float* A, const float* B, int N);
double frobenius_norm(const float* M, int N);

#endif
