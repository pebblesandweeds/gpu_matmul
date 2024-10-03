#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include "matmul.h"
#include "timer.h"
#include "check_utils.h"
#include <rocblas/rocblas.h>

void print_gpu_info(int deviceId) {
    hipDeviceProp_t deviceProp;
    CHECK(hipGetDeviceProperties(&deviceProp, deviceId));
    printf("Device %d: %s\n", deviceId, deviceProp.name);
    printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total global memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Clock rate: %.2f GHz\n", deviceProp.clockRate * 1e-6f);
    printf("  Number of compute units: %d\n", deviceProp.multiProcessorCount);
    printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("  Warp size: %d\n", deviceProp.warpSize);
    printf("  Max registers per block: %d\n", deviceProp.regsPerBlock);
    printf("  Max registers per multiprocessor: %d\n", deviceProp.regsPerMultiprocessor);
    printf("  Max shared memory per block: %zu KB\n", deviceProp.sharedMemPerBlock / 1024);
    printf("  Shared memory per multiprocessor: %zu B\n", deviceProp.maxSharedMemoryPerMultiProcessor);
    printf("  Max warps per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize);
}

int main(int argc, char* argv[]) {
    int deviceId = 0; // Default to first GPU
    if (argc > 1) {
        deviceId = atoi(argv[1]);
    }
    int deviceCount;
    CHECK(hipGetDeviceCount(&deviceCount));
    printf("Number of available GPUs: %d\n", deviceCount);
    if (deviceId < 0 || deviceId >= deviceCount) {
        fprintf(stderr, "Invalid device ID. Please specify a device ID between 0 and %d.\n", deviceCount - 1);
        return 1;
    }
    CHECK(hipSetDevice(deviceId));
    print_gpu_info(deviceId);

    float *h_A, *h_B, *h_C_naive, *h_C_scalar, *h_C_mfma;
    float *d_A, *d_B, *d_C_naive, *d_C_scalar, *d_C_mfma;
    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C_naive = (float*)malloc(size);
    h_C_scalar = (float*)malloc(size);
    h_C_mfma = (float*)malloc(size);

    init_matrix(h_A, N);
    init_matrix(h_B, N);

    CHECK(hipMalloc((void**)&d_A, size));
    CHECK(hipMalloc((void**)&d_B, size));
    CHECK(hipMalloc((void**)&d_C_naive, size));
    CHECK(hipMalloc((void**)&d_C_scalar, size));
    CHECK(hipMalloc((void**)&d_C_mfma, size));

    CHECK(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));

    hipEvent_t start, stop;

    // Naive matrix multiplication
    dim3 blockDim = {BLOCK_SIZE, BLOCK_SIZE, 1};
    dim3 gridDim = {(N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, 1};
    start_timer(&start, &stop);
    hipLaunchKernelGGL(matmul_kernel, gridDim, blockDim, 0, NULL, d_A, d_B, d_C_naive, N);
    CHECK(hipGetLastError());
    float milliseconds_naive = stop_timer(start, stop);
    CHECK(hipMemcpy(h_C_naive, d_C_naive, size, hipMemcpyDeviceToHost));

    // Shared memory matrix multiplication
    dim3 scalargridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 scalarblockDim(BLOCK_SIZE * BLOCK_SIZE / (thread_multiplier * thread_multiplier), 1);
    start_timer(&start, &stop);
    hipLaunchKernelGGL(matmul_scalar_kernel, scalargridDim, scalarblockDim, 0, NULL, d_A, d_B, d_C_scalar, N);
    CHECK(hipGetLastError());
    float milliseconds_scalar = stop_timer(start, stop);
    CHECK(hipMemcpy(h_C_scalar, d_C_scalar, size, hipMemcpyDeviceToHost));

    // MFMA matrix multiplication
    dim3 mfmaBlockDim(64,1);
    dim3 mfmaGridDim(256, 16384);
    start_timer(&start, &stop);
    hipLaunchKernelGGL(gemm_mfma_naive, mfmaGridDim, mfmaBlockDim, 0, NULL, d_C_mfma, (__fp16*)d_A, (__fp16*)d_B, d_C_mfma, 1.0f, 0.0f);
    CHECK(hipGetLastError());
    float milliseconds_mfma = stop_timer(start, stop);
    CHECK(hipMemcpy(h_C_mfma, d_C_mfma, size, hipMemcpyDeviceToHost));

    // Verify implementations
    printf("\nVerifying naive implementation:\n");

    bool naive_correct = partial_verify(h_A, h_B, h_C_naive, N * N, 1e-4);

    // Verify scalar memory implementation
    printf("\nVerifying scalar memory implementation:\n");
    bool scalar_correct = partial_verify(h_A, h_B, h_C_scalar, N * N, 1e-4);

    // Verify MFMA implementation
    printf("\nVerifying MFMA implementation:\n");
    bool mfma_correct = partial_verify(h_A, h_B, h_C_mfma, N * N, 1e-4);

    if (naive_correct && scalar_correct && mfma_correct) {
        printf("All implementations appear to be correct.\n");
    } else {
        printf("Some implementations may have issues:\n");
        printf("Naive: %s\n", naive_correct ? "Correct" : "Incorrect");
        printf("Scalar: %s\n", scalar_correct ? "Correct" : "Incorrect");
        printf("MFMA: %s\n", mfma_correct ? "Correct" : "Incorrect");
    }

    double seconds_naive = milliseconds_naive / 1000.0;
    double seconds_scalar = milliseconds_scalar / 1000.0;
    double seconds_mfma = milliseconds_mfma / 1000.0;
    long long flop = 2LL * N * N * N;
    double gflops_naive = (flop / seconds_naive) / 1e9;
    double gflops_scalar = (flop / seconds_scalar) / 1e9;
    double gflops_mfma = (flop / seconds_mfma) / 1e9;
    printf("Naive GPU Matmul time taken: %.6f seconds\n", seconds_naive);
    printf("Naive GPU Matmul: %.2f GFLOPS\n", gflops_naive);
    printf("Shared memory GPU Matmul time taken: %.6f seconds\n", seconds_scalar);
    printf("Shared memory GPU Matmul: %.2f GFLOPS\n", gflops_scalar);
    printf("MFMA GPU Matmul time taken: %.6f seconds\n", seconds_mfma);
    printf("MFMA GPU Matmul: %.2f GFLOPS\n", gflops_mfma);

    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_scalar);
    free(h_C_mfma);
    CHECK(hipFree(d_A));
    CHECK(hipFree(d_B));
    CHECK(hipFree(d_C_naive));
    CHECK(hipFree(d_C_scalar));
    CHECK(hipFree(d_C_mfma));

    return 0;
}
