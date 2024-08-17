#include <hip/hip_runtime.h>
#include "timer.h"
#include "matmul.h"

float time_matmul(const float* d_A, const float* d_B, float* d_C, int N) {
    hipEvent_t start, stop;
    float milliseconds = 0;

    CHECK(hipEventCreate(&start));
    CHECK(hipEventCreate(&stop));
    
    CHECK(hipEventRecord(start, NULL));
    naive_matmul(d_A, d_B, d_C, N);
    CHECK(hipEventRecord(stop, NULL));
    CHECK(hipEventSynchronize(stop));
    
    CHECK(hipEventElapsedTime(&milliseconds, start, stop));

    CHECK(hipEventDestroy(start));
    CHECK(hipEventDestroy(stop));

    return milliseconds;
}
