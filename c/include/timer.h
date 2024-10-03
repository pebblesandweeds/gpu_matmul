#ifndef TIMER_H
#define TIMER_H

#include <hip/hip_runtime.h>

float time_memory_transfer(const float *h_A, const float *h_B, float *d_A, float *d_B, size_t size);
float time_memory_transfer_back(float *h_C, float *d_C, size_t size);

#endif // TIMER_H
