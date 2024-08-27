#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include "matmul.h"
#include "timer.h"
#include "check_utils.h"

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

    float *h_A, *h_B, *h_C_naive, *h_C_shared;
    float *d_A, *d_B, *d_C_naive, *d_C_shared;
    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C_naive = (float*)malloc(size);
    h_C_shared = (float*)malloc(size);

    init_matrix(h_A, N);
    init_matrix(h_B, N);

    CHECK(hipMalloc((void**)&d_A, size));
    CHECK(hipMalloc((void**)&d_B, size));
    CHECK(hipMalloc((void**)&d_C_naive, size));
    CHECK(hipMalloc((void**)&d_C_shared, size));

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
    dim3 sharedgridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 sharedblockDim(BLOCK_SIZE, BLOCK_SIZE / TM);
    start_timer(&start, &stop);
    hipLaunchKernelGGL(matmul_shared_kernel, sharedgridDim, sharedblockDim, 0, NULL, d_A, d_B, d_C_shared, N);
    CHECK(hipGetLastError());
    float milliseconds_shared = stop_timer(start, stop);
    CHECK(hipMemcpy(h_C_shared, d_C_shared, size, hipMemcpyDeviceToHost));

    // Verify naive implementation
    printf("\nVerifying naive implementation:\n");
    bool naive_correct = partial_verify(h_A, h_B, h_C_naive, N * N, 1e-4);

    // Verify shared memory implementation
    printf("\nVerifying shared memory implementation:\n");
    bool shared_correct = partial_verify(h_A, h_B, h_C_shared, N * N, 1e-4);

    if (naive_correct && shared_correct) {
        printf("Both implementations appear to be correct.\n");
    } else if (naive_correct) {
        printf("Naive implementation is correct, but shared memory implementation may have issues.\n");
    } else if (shared_correct) {
        printf("Shared memory implementation is correct, but naive implementation may have issues.\n");
    } else {
        printf("Both implementations may have issues.\n");
    }

    double seconds_naive = milliseconds_naive / 1000.0;
    double seconds_shared = milliseconds_shared / 1000.0;
    long long flop = 2LL * N * N * N;
    double gflops_naive = (flop / seconds_naive) / 1e9;
    double gflops_shared = (flop / seconds_shared) / 1e9;
    printf("Naive GPU Matmul time taken: %.6f seconds\n", seconds_naive);
    printf("Naive GPU Matmul: %.2f GFLOPS\n", gflops_naive);
    printf("Shared memory GPU Matmul time taken: %.6f seconds\n", seconds_shared);
    printf("Shared memory GPU Matmul: %.2f GFLOPS\n", gflops_shared);

    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_shared);
    CHECK(hipFree(d_A));
    CHECK(hipFree(d_B));
    CHECK(hipFree(d_C_naive));
    CHECK(hipFree(d_C_shared));

    return 0;
}
