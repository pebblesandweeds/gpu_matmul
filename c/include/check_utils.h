#ifndef CHECK_UTILS_H
#define CHECK_UTILS_H

#include "matmul.h"
#include <stdbool.h>

bool check_matrices(const float* C1, const float* C2, int N, float tolerance);

#endif
