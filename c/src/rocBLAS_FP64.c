#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <time.h>
#include <math.h>

#define N 16384 // Matrix size
#define NUM_SPOT_CHECKS 50

#define CHECK_HIP(stmt) do {                                 \
    hipError_t err = stmt;                                   \
    if (err != hipSuccess) {                                 \
        printf("HIP error: %s\n", hipGetErrorString(err));   \
        exit(1);                                             \
    }                                                        \
} while(0)

#define CHECK_ROCBLAS(stmt) do {                             \
    rocblas_status status = stmt;                            \
    if (status != rocblas_status_success) {                  \
        printf("rocBLAS error: %d\n", status);               \
        exit(1);                                             \
    }                                                        \
} while(0)

void print_matrix(const double *matrix, int n, const char *name) {
    printf("%s:\n", name);
    for (int i = 0; i < n && i < 4; i++) {
        for (int j = 0; j < n && j < 4; j++) {
            printf("%8.4f ", matrix[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void spot_check(const double *A, const double *B, const double *C_gpu, int n) {
    printf("Performing random spot checks between CPU and GPU results...\n");
    srand(time(NULL));
    for (int i = 0; i < NUM_SPOT_CHECKS; ++i) {
        int row = rand() % n;
        int col = rand() % n;
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        double gpu_value = C_gpu[row * n + col];
        double abs_error = fabs(sum - gpu_value);
        double rel_error = abs_error / (fabs(sum) + 1e-8);
        if (rel_error > 1e-5) {
            printf("Mismatch at C[%d, %d]: Expected = %.6e, GPU = %.6e, Relative Error = %.6e\n", row, col, sum, gpu_value, rel_error);
        } else {
            printf("Value match at C[%d, %d]: %.6e (Relative Error: %.6e)\n", row, col, gpu_value, rel_error);
        }
    }
}

void transpose_matrix(const double *src, double *dst, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dst[j * n + i] = src[i * n + j];
        }
    }
}

int main() {
    const int NUM_RUNS = 25;
    size_t size = N * N * sizeof(double);

    hipDevice_t device;
    hipDeviceProp_t props;
    CHECK_HIP(hipGetDevice(&device));
    CHECK_HIP(hipGetDeviceProperties(&props, device));
    printf("GPU: %s\n", props.name);
    printf("Total GPU memory: %zu MB\n", props.totalGlobalMem / (1024 * 1024));
    printf("GPU clock rate: %d MHz\n", props.clockRate / 1000);

    double *h_A, *h_B, *h_C, *h_A_trans, *h_B_trans, *h_C_trans;
    double *d_A, *d_B, *d_C;

    h_A = (double*)malloc(size);
    h_B = (double*)malloc(size);
    h_C = (double*)malloc(size);
    h_A_trans = (double*)malloc(size);
    h_B_trans = (double*)malloc(size);
    h_C_trans = (double*)malloc(size);

    // Initialize matrices with random values between -1 and 1
    for (int i = 0; i < N*N; i++) {
        h_A[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        h_B[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
    }

    // Transpose matrices A and B
    transpose_matrix(h_A, h_A_trans, N);
    transpose_matrix(h_B, h_B_trans, N);

    // Allocate device memory
    CHECK_HIP(hipMalloc(&d_A, size));
    CHECK_HIP(hipMalloc(&d_B, size));
    CHECK_HIP(hipMalloc(&d_C, size));

    // Create events for timing
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    // Time memory transfer to device
    CHECK_HIP(hipEventRecord(start));
    CHECK_HIP(hipMemcpy(d_A, h_A_trans, size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B_trans, size, hipMemcpyHostToDevice));
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    float transfer_time;
    CHECK_HIP(hipEventElapsedTime(&transfer_time, start, stop));
    printf("Memory transfer to device time: %f ms\n", transfer_time);

    // Create rocBLAS handle
    rocblas_handle handle;
    CHECK_ROCBLAS(rocblas_create_handle(&handle));

    // Set up the matrix multiplication
    const double alpha = 1.0;
    const double beta = 0.0;

    // Calculate total FLOPs for matrix multiplication
    double total_flops = 2.0 * N * N * N;

    // Perform multiple runs
    for (int run = 0; run < NUM_RUNS; run++) {
        CHECK_HIP(hipEventRecord(start));
        CHECK_ROCBLAS(rocblas_dgemm(handle,
                                    rocblas_operation_none, rocblas_operation_none,
                                    N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));
        CHECK_HIP(hipEventRecord(stop));
        CHECK_HIP(hipEventSynchronize(stop));

        float compute_time;
        CHECK_HIP(hipEventElapsedTime(&compute_time, start, stop));
        double seconds = compute_time / 1000.0;
        double tflops = total_flops / (seconds * 1e12);

        printf("Run %d: Matrix multiplication time: %f ms, Performance: %.2f TFLOPS\n",
               run+1, compute_time, tflops);
    }

    // Time memory transfer back to host
    CHECK_HIP(hipEventRecord(start));
    CHECK_HIP(hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost));
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    float transfer_back_time;
    CHECK_HIP(hipEventElapsedTime(&transfer_back_time, start, stop));
    printf("Memory transfer from device time: %f ms\n", transfer_back_time);

    // Transpose the result matrix C
    transpose_matrix(h_C, h_C_trans, N);

    // Print small portions of the matrices to verify computation
    print_matrix(h_A, N, "Matrix A (original)");
    print_matrix(h_B, N, "Matrix B (original)");
    print_matrix(h_C_trans, N, "Result Matrix C (transposed)");

    // Perform spot check
    spot_check(h_A, h_B, h_C_trans, N);

    // Clean up
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
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    return 0;
}

