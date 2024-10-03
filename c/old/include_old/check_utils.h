#ifndef CHECK_UTILS_H
#define CHECK_UTILS_H
#include <stdbool.h>

bool partial_verify(const float* A, const float* B, const float* C_gpu, int size, float tolerance);

#endif
