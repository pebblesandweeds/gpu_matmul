#include <hip/hip_runtime.h>
#include "timer.h"

void start_timer(hipEvent_t *start, hipEvent_t *stop) {
    CHECK(hipEventCreate(start));
    CHECK(hipEventCreate(stop));
    CHECK(hipEventRecord(*start, NULL));
}

float stop_timer(hipEvent_t start, hipEvent_t stop) {
    float milliseconds = 0;
    CHECK(hipEventRecord(stop, NULL));
    CHECK(hipEventSynchronize(stop));
    CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    CHECK(hipEventDestroy(start));
    CHECK(hipEventDestroy(stop));
    return milliseconds;
}
