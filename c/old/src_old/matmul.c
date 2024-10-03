#include <hip/hip_runtime.h>
#include "matmul.h"
#include <rocblas/rocblas.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int n) {
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matmul_scalar_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {
    int threadCol = hipThreadIdx_x % (BLOCK_SIZE / thread_multiplier);
    int threadRow = hipThreadIdx_x / (BLOCK_SIZE / thread_multiplier);

    __shared__ float shared_a[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE * BLOCK_SIZE];

    A += hipBlockIdx_y * BLOCK_SIZE * n;
    B += hipBlockIdx_x * BLOCK_SIZE;
    C += hipBlockIdx_y * BLOCK_SIZE * n + hipBlockIdx_x * BLOCK_SIZE;

    float threadResults[thread_multiplier * thread_multiplier] = {0.0f};

    float regM[thread_multiplier] = {0.0f};
    float regN[thread_multiplier] = {0.0f};

    int threads_per_block = BLOCK_SIZE * BLOCK_SIZE / (thread_multiplier * thread_multiplier);
    int stride = threads_per_block / BLOCK_SIZE;

    for (int block_index = 0; block_index < n; block_index += BLOCK_SIZE) {
        for (int i = 0; i < BLOCK_SIZE; i += stride) {
            int row = hipThreadIdx_x / BLOCK_SIZE;
            int col = hipThreadIdx_x % BLOCK_SIZE;
            shared_a[(row + i) * BLOCK_SIZE + col] = A[(row + i) * n + col];
            shared_b[(row + i) * BLOCK_SIZE + col] = B[(row + i) * n + col];
        }

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * n;

        for (int dot_product_index = 0; dot_product_index < BLOCK_SIZE; ++dot_product_index) {
            for (int i = 0; i < thread_multiplier; ++i) {
                regM[i] = shared_a[(threadRow * thread_multiplier + i) * BLOCK_SIZE + dot_product_index];
                regN[i] = shared_b[dot_product_index * BLOCK_SIZE + threadCol * thread_multiplier + i];
            }

            for (int rm = 0; rm < thread_multiplier; rm += 2) {
                for (int rn = 0; rn < thread_multiplier; rn += 2) {
                    threadResults[rm * thread_multiplier + rn] = __fmaf_rn(regM[rm], regN[rn], threadResults[rm * thread_multiplier + rn]);
                    threadResults[rm * thread_multiplier + rn + 1] = __fmaf_rn(regM[rm], regN[rn + 1], threadResults[rm * thread_multiplier + rn + 1]);
                    threadResults[(rm + 1) * thread_multiplier + rn] = __fmaf_rn(regM[rm + 1], regN[rn], threadResults[(rm + 1) * thread_multiplier + rn]);
                    threadResults[(rm + 1) * thread_multiplier + rn + 1] = __fmaf_rn(regM[rm + 1], regN[rn + 1], threadResults[(rm + 1) * thread_multiplier + rn + 1]);
                }
            }
        }

        __syncthreads();

    }
    for (int rm = 0; rm < thread_multiplier; ++rm) {
        for (int rn = 0; rn < thread_multiplier; ++rn) {
            C[(threadRow * thread_multiplier + rm) * n + threadCol * thread_multiplier + rn] = threadResults[rm * thread_multiplier + rn];
        }
    }
}

// Define constants for wave size and thread block size
#define WAVE_SIZE 64
#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16

// Define types that match the MFMA intrinsic expectations
typedef __fp16 __attribute__((__vector_size__(4 * sizeof(__fp16)))) AFragT;  // Vector of 4 half-precision floats (__fp16)
typedef __fp16 __attribute__((__vector_size__(4 * sizeof(__fp16)))) BFragT;  // Vector of 4 half-precision floats (__fp16)
typedef float __attribute__((__vector_size__(4 * sizeof(float)))) CFragT;  // Vector of 4 single-precision floats
typedef float  __attribute__((__vector_size__(4 * sizeof(float)))) AccumFragT;  // Vector of 4 single-precision floats (float)

// Perform Multiply-Accumulate using MFMA
__device__ AccumFragT mfma_f32_16x16x16f16(AFragT aFrag, BFragT bFrag, AccumFragT accumFrag) {
    // Return the result of the MFMA instruction directly
    return __builtin_amdgcn_mfma_f32_16x16x16f16(aFrag, bFrag, accumFrag, 0, 0, 0);
}

// Now we add the load_A_16x16_col_major function below the fragment definitions
__device__ AFragT load_A_16x16_col_major(const __fp16* input, int ld) {
    AFragT a_frag;

    // Constants for the vector width and dimension (BLOCK_M)
    const uint32_t VW = BLOCK_M / WAVE_SIZE;  // Vector width, based on fragment size
    const uint32_t Dim = BLOCK_M;             // Dimension, as BLOCK_M

    // Calculate the 2D coordinates where each thread starts loading
    uint32_t row = threadIdx.x % Dim;         // Row coordinate
    uint32_t col = (threadIdx.x / Dim) * VW;  // Column coordinate, increments by vector width

    // Step in 2D space for moving between loads
    uint32_t step_row = 0;                    // Row does not move for vector width
    uint32_t step_col = 1;                    // Column step for the next element in the vector

    // Helper macro to compute column-major offsets in 1D
    #define col_major_offset(row, col, ld) ((row) + (col) * (ld))

    // Compute starting offset and step offset for loading 4 elements
    uint32_t start_offset = col_major_offset(row, col, ld);              // Starting point
    uint32_t k_offset = col_major_offset(step_row, step_col, ld);        // Offset step for next element

    // Load 4 non-contiguous elements into the fragment (spread across multiple rows and columns)
    a_frag[0] = input[start_offset];                // First element
    a_frag[1] = input[start_offset + k_offset];     // Second element
    a_frag[2] = input[start_offset + 2 * k_offset]; // Third element
    a_frag[3] = input[start_offset + 3 * k_offset]; // Fourth element

    return a_frag;  // Return the fragment
}

__device__ BFragT load_B_16x16_row_major(const __fp16* input, int ld) {
    BFragT b_frag;

    // Constants for the vector width and dimension (BLOCK_N)
    const uint32_t VW = BLOCK_N / WAVE_SIZE;  // Vector width, based on fragment size
    const uint32_t Dim = BLOCK_N;             // Dimension, as BLOCK_N

    // Calculate the 2D coordinates where each thread starts loading
    uint32_t row = (threadIdx.x / Dim) * VW;  // Row coordinate (vertical block position)
    uint32_t col = threadIdx.x % Dim;         // Column coordinate (horizontal position within row)

    // Step in 2D space for moving between loads
    uint32_t step_row = 1;                    // Row step for the next element in the vector
    uint32_t step_col = 0;                    // Column does not move for vector width

    // Helper macro to compute row-major offsets in 1D
    #define row_major_offset(row, col, ld) ((row) * (ld) + (col))

    // Compute starting offset and step offset for loading 4 elements
    uint32_t start_offset = row_major_offset(row, col, ld);              // Starting point
    uint32_t k_offset = row_major_offset(step_row, step_col, ld);        // Offset step for next element

    // Load 4 non-contiguous elements into the fragment (spread across multiple rows)
    b_frag[0] = input[start_offset];                // First element
    b_frag[1] = input[start_offset + k_offset];     // Second element
    b_frag[2] = input[start_offset + 2 * k_offset]; // Third element
    b_frag[3] = input[start_offset + 3 * k_offset]; // Fourth element

    return b_frag;  // Return the fragment
}

__device__ CFragT load_C_16x16_col_major(const float* input, int ld) {
    CFragT c_frag;

    // Constants for the vector width and dimension (BLOCK_N)
    const uint32_t VW = BLOCK_N / WAVE_SIZE;  // Vector width, based on fragment size
    const uint32_t Dim = BLOCK_N;             // Dimension, as BLOCK_N

    // Calculate the 2D coordinates where each thread starts loading
    uint32_t row = threadIdx.x % Dim;         // Row coordinate
    uint32_t col = (threadIdx.x / Dim) * VW;  // Column coordinate, increments by vector width

    // Step in 2D space for moving between loads
    uint32_t step_row = 1;                    // Move to next row
    uint32_t step_col = 0;                    // Stay in the same column

    // Helper macro to compute col-major offsets in 1D
    #define col_major_offset(row, col, ld) ((row) + (col) * (ld))

    // Compute starting offset and step offset for loading 4 elements
    uint32_t start_offset = col_major_offset(row, col, ld);              // Starting point
    uint32_t k_offset = col_major_offset(step_row, step_col, ld);        // Offset step for next element

    // Load 4 non-contiguous elements into the fragment (spread across multiple rows and columns)
    c_frag[0] = input[start_offset];                // First element
    c_frag[1] = input[start_offset + k_offset];     // Second element
    c_frag[2] = input[start_offset + 2 * k_offset]; // Third element
    c_frag[3] = input[start_offset + 3 * k_offset]; // Fourth element

    return c_frag;  // Return the fragment
}

__device__ void store_C_16x16_col_major(float* output, const CFragT c_frag, int ld) {
    // Constants for the vector width and dimension (BLOCK_N)
    const uint32_t VW = BLOCK_N / WAVE_SIZE;  // Vector width, based on fragment size
    const uint32_t Dim = BLOCK_N;             // Dimension, as BLOCK_N

    // Calculate the 2D coordinates where each thread starts storing
    uint32_t row = threadIdx.x % Dim;         // Row coordinate
    uint32_t col = (threadIdx.x / Dim) * VW;  // Column coordinate, increments by vector width

    // Step in 2D space for moving between stores
    uint32_t step_row = 1;                    // Move to next row
    uint32_t step_col = 0;                    // Stay in the same column

    // Helper macro to compute col-major offsets in 1D
    #define col_major_offset(row, col, ld) ((row) + (col) * (ld))

    // Compute starting offset and step offset for storing 4 elements
    uint32_t start_offset = col_major_offset(row, col, ld);              // Starting point
    uint32_t k_offset = col_major_offset(step_row, step_col, ld);        // Offset step for next element

    // Store 4 non-contiguous elements from the fragment into memory
    output[start_offset] = c_frag[0];                // Store first element
    output[start_offset + k_offset] = c_frag[1];     // Store second element
    output[start_offset + 2 * k_offset] = c_frag[2]; // Store third element
    output[start_offset + 3 * k_offset] = c_frag[3]; // Store fourth element
}

__global__ void gemm_mfma_naive(float* __restrict__ D,
                                const __fp16* __restrict__ A,
                                const __fp16* __restrict__ B,
                                const float* __restrict__ C,
                                float alpha,
                                float beta) {
    
    int block_tile_x = blockIdx.x * BLOCK_N;
    int block_tile_y = blockIdx.y * BLOCK_M;
    AccumFragT accum = {0.0f};
    AFragT a_frag = load_A_16x16_col_major(A, BLOCK_M);
    BFragT b_frag = load_B_16x16_row_major(B, BLOCK_N);
    CFragT c_frag = load_C_16x16_col_major(C, BLOCK_N);
    accum = mfma_f32_16x16x16f16(a_frag, b_frag, accum);
    store_C_16x16_col_major(D, accum, BLOCK_M);

}

void init_matrix(float *matrix, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
    matrix[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * 2.0f - 1.0f;
    }
}
