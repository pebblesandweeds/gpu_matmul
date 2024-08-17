#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include "matmul.h"
#include "timer.h"

#define N 16384  

void print_gpu_info(int deviceId) {
    hipDeviceProp_t deviceProp;
    CHECK(hipGetDeviceProperties(&deviceProp, deviceId));
    printf("Device %d: %s\n", deviceId, deviceProp.name);
    printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total global memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Clock rate: %.2f GHz\n", deviceProp.clockRate * 1e-6f);
    printf("  Number of compute units: %d\n", deviceProp.multiProcessorCount);
    printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
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

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    init_matrix(h_A, N);
    init_matrix(h_B, N);

    CHECK(hipMalloc((void**)&d_A, size));
    CHECK(hipMalloc((void**)&d_B, size));
    CHECK(hipMalloc((void**)&d_C, size));

    CHECK(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));

    dim3 blockDim = {16, 16, 1};
    dim3 gridDim = {(N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, 1};

    hipEvent_t start, stop;
    start_timer(&start, &stop);

    hipLaunchKernelGGL(matmul_kernel, gridDim, blockDim, 0, NULL, d_A, d_B, d_C, N);
    CHECK(hipGetLastError());

    float milliseconds = stop_timer(start, stop);

    CHECK(hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost));

    double seconds = milliseconds / 1000.0;
    long long flop = 2LL * N * N * N;
    double gflops = (flop / seconds) / 1e9;
    printf("GPU Matmul time taken: %.6f seconds\n", seconds);
    printf("GPU Matmul: %.2f GFLOPS\n", gflops);

    free(h_A);
    free(h_B);
    free(h_C);
    CHECK(hipFree(d_A));
    CHECK(hipFree(d_B));
    CHECK(hipFree(d_C));

    return 0;
}
