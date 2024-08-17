#include "../include/check_utils.h"
#include "../include/matmul.h"
#include <math.h>

bool check_matrices(float C1[N][N], float C2[N][N], float tolerance) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(C1[i][j] - C2[i][j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}
