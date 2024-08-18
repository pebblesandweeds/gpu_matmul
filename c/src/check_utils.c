#include "../include/check_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SAMPLE_SIZE 1000

bool partial_verify(const float* A, const float* B, const float* C_gpu, int N, float tolerance) {
    int mismatches = 0;
    
    srand(12345);  // Seed for reproducibility

    for (int s = 0; s < SAMPLE_SIZE; s++) {
        int i = rand() % N;
        int j = rand() % N;
        
        float expected = 0.0f;
        for (int k = 0; k < N; k++) {
            expected += A[i * N + k] * B[k * N + j];
        }
        
        float actual = C_gpu[i * N + j];
        
        if (fabsf(actual - expected) > tolerance * fabsf(expected)) {
            mismatches++;
            printf("Mismatch at (%d, %d): Expected %f, Got %f\n", i, j, expected, actual);
            if (mismatches > 10) {
                printf("Too many mismatches, stopping comparison.\n");
                break;
            }
        }
    }

    float mismatch_rate = (float)mismatches / SAMPLE_SIZE;
    printf("Mismatch rate: %.2f%%\n", mismatch_rate * 100);

    return mismatch_rate < 0.01;  // Consider it correct if less than 1% mismatch
}

double compute_trace_of_product(const float* A, const float* B, int N) {
    double trace = 0.0;
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += (double)A[i * N + k] * (double)B[k * N + i];
        }
        trace += sum;
    }
    return trace;
}

double frobenius_norm(const float* M, int N) {
    double sum = 0.0;
    for (int i = 0; i < N * N; i++) {
        sum += (double)M[i] * (double)M[i];
    }
    return sqrt(sum);
}
