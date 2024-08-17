#include "../include/check_utils.h"
#include "../include/matmul.h"
#include <math.h>

bool check_matrices(const float* C1, const float* C2, int N, float tolerance) {
    for (int i = 0; i < N * N; i++) {
        if (fabs(C1[i] - C2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}
