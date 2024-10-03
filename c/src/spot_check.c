#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../include/spot_check.h"

void spot_check(const float *A, const float *B, const float *C_gpu, int n) {
    printf("Performing random spot checks between CPU and GPU results...\n");
    srand(time(NULL));
    int mismatch_count = 0;

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
            printf("Mismatch at C[%d, %d]: Expected = %.6e, GPU = %.6e, Relative Error = %.6e\n", 
                   row, col, sum, gpu_value, rel_error);
            mismatch_count++;
        }
    }

    if (mismatch_count == 0) {
        printf("Success: All %d spot checks passed within the relative error threshold.\n", NUM_SPOT_CHECKS);
    } else {
        printf("Found %d mismatches out of %d spot checks.\n", mismatch_count, NUM_SPOT_CHECKS);
    }
}
