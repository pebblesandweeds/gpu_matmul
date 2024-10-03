#include "../include/timer.h"
#include "../include/utils.h"

float time_memory_transfer(const float *h_A, const float *h_B, float *d_A, float *d_B, size_t size) {
    hipEvent_t start, stop;
    float transfer_time;

    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    CHECK_HIP(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    CHECK_HIP(hipEventElapsedTime(&transfer_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    return transfer_time;
}

float time_memory_transfer_back(float *h_C, float *d_C, size_t size) {
    hipEvent_t start, stop;
    float transfer_back_time;

    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    CHECK_HIP(hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost));
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    CHECK_HIP(hipEventElapsedTime(&transfer_back_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    return transfer_back_time;
}
