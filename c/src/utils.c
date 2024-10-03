#include <stdio.h>
#include <stdlib.h>
#include "../include/utils.h"

void print_gpu_info() {
    hipDevice_t device;
    hipDeviceProp_t props;
    CHECK_HIP(hipGetDevice(&device));
    CHECK_HIP(hipGetDeviceProperties(&props, device));
    printf("GPU: %s\n", props.name);
    printf("Total GPU memory: %zu MB\n", props.totalGlobalMem / (1024 * 1024));
    printf("GPU clock rate: %d MHz\n", props.clockRate / 1000);
}

void print_precision() {
    printf("Matrix Element Precision: %s\n", get_precision_string(sizeof(float)));
    printf("rocBLAS Function: rocblas_sgemm (Single Precision)\n");
}

const char* get_precision_string(size_t size) {
    switch(size) {
        case sizeof(float):
            return "Single Precision (32-bit)";
        case sizeof(double):
            return "Double Precision (64-bit)";
        default:
            return "Unknown Precision";
    }
}

void cleanup(rocblas_handle handle, float *d_A, float *d_B, float *d_C, 
             float *h_A, float *h_B, float *h_C, 
             float *h_A_trans, float *h_B_trans, float *h_C_trans) {
    CHECK_ROCBLAS(rocblas_destroy_handle(handle));
    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_A_trans);
    free(h_B_trans);
    free(h_C_trans);
}
