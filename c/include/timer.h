#ifndef TIMER_H
#define TIMER_H

#include <hip/hip_runtime.h>

void start_timer(hipEvent_t *start, hipEvent_t *stop);
float stop_timer(hipEvent_t start, hipEvent_t stop);

#endif // TIMER_H
