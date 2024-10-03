#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../include/spot_check.h"

void spot_check(const float *A, const float *B, const float *C_gpu, int n) {
    printf("Performing random spot checks between CPU and GPU results...\n");
    srand(time(NULL));
    for (int i = 0; i < NUM_SPOT_CHECKS; ++i) {
        int row = rand() % n;
        int col = rand() % n;
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        float gpu_value = C_gpu[row * n + col];
        float abs_error = fabsf(sum - gpu_value);
        float rel_error = abs_error / (fabsf(sum) + 1e-8f);
        if (rel_error > 1e-4f) {
            printf("Mismatch at C[%d, %d]: Expected = %.6e, GPU = %.6e, Relative Error = %.6e\n", row, col, sum, gpu_value, rel_error);
        } else {
            printf("Value match at C[%d, %d]: %.6e (Relative Error: %.6e)\n", row, col, gpu_value, rel_error);
        }
    }
}
