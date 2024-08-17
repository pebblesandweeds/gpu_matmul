#ifndef CHECK_UTILS_H
#define CHECK_UTILS_H

#include "matmul_lib.h"
#include <stdbool.h>

bool check_matrices(float C1[N][N], float C2[N][N], float tolerance);

#endif
