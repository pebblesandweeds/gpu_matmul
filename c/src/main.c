#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "../include/timer.h"
#include "../include/matrix_operations.h"
#include "../include/spot_check.h"
#include "../include/utils.h"

#define N 16384 // Matrix size
#define NUM_RUNS 25

int main() {
    size_t size = N * N * sizeof(float);

    print_gpu_info();
    print_precision();

    float *h_A, *h_B, *h_C, *h_A_trans, *h_B_trans, *h_C_trans;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    h_A_trans = (float*)malloc(size);
    h_B_trans = (float*)malloc(size);
    h_C_trans = (float*)malloc(size);

    initialize_matrices(h_A, h_B, N);
    transpose_matrix(h_A, h_A_trans, N);
    transpose_matrix(h_B, h_B_trans, N);

    CHECK_HIP(hipMalloc(&d_A, size));
    CHECK_HIP(hipMalloc(&d_B, size));
    CHECK_HIP(hipMalloc(&d_C, size));

    float transfer_time = time_memory_transfer(h_A_trans, h_B_trans, d_A, d_B, size);
    printf("Memory transfer to device time: %f ms\n", transfer_time);

    rocblas_handle handle;
    CHECK_ROCBLAS(rocblas_create_handle(&handle));

    perform_matrix_multiplication(handle, d_A, d_B, d_C, N, NUM_RUNS);

    float transfer_back_time = time_memory_transfer_back(h_C, d_C, size);
    printf("Memory transfer from device time: %f ms\n", transfer_back_time);

    transpose_matrix(h_C, h_C_trans, N);

    spot_check(h_A, h_B, h_C_trans, N);

    cleanup(handle, d_A, d_B, d_C, h_A, h_B, h_C, h_A_trans, h_B_trans, h_C_trans);

    return 0;
}
