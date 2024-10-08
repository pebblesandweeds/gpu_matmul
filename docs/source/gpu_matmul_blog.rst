Accelerating Matrix Multiplication on AMD GPUs with rocBLAS in C
================================================================

.. admonition:: Highlights 

 Matrix multiplication is the core operation behind deep learning, driving the computations in neural networks for model training and inference. This blog post demonstrates how AMD's rocBLAS library can be used in C to achieve matrix multiplication performance comparable to PyTorch's implementation, leveraging low-level control for efficient use of AMD GPUs.

 - **PyTorch Baseline**: Achieves **~37.5 TFLOPS** using `this simple code <https://github.com/pebblesandweeds/gpu_matmul/blob/main/pytorch/pytorch_matmul.py>`_. PyTorch's high-level API (``torch.matmul``) abstracts the underlying rocBLAS operations, providing ease of use without sacrificing performance.

 - **rocBLAS Implementation in C**: Matches PyTorch at **~37.5 TFLOPS** with `this C implementation <https://github.com/pebblesandweeds/gpu_matmul/blob/main/c/src/matrix_operations.c>`_. By directly calling ``rocblas_sgemm()``, we expose GPU programming concepts like memory allocation, data transfer, and operation parameters which provide insight into the underlying processes that high-level APIs abstract away.

 This comparison showcases how low-level C programming with rocBLAS can achieve performance parity with high-level frameworks like PyTorch. The C implementation offers a valuable learning opportunity, introducing developers to GPU programming concepts while maintaining high performance. It serves as a bridge between high-level APIs and custom GPU kernel development, providing a deeper understanding of GPU computing without sacrificing efficiency.

 Get all of the code `in this repo <https://github.com/pebblesandweeds/gpu_matmul>`_.


Introduction
------------

Matrix multiplication is a cornerstone operation in machine learning and deep learning, powering critical computations in neural networks such as forward and backward propagation. In `our previous blog post <https://blog.pebblesandweeds.com/cpu_matmul_blog.html#why-is-matrix-multiplication-important>`_, we explored this fundamental operation by implementing matrix multiplication from scratch in C on AMD EPYC CPUs. This exploration laid the groundwork for understanding the core principles behind matrix multiplication, setting the stage for our journey into GPU-accelerated computations. 

Building on that foundation, this blog post extends our exploration to GPU acceleration. We demonstrate how to harness the power of AMD GPUs for high-performance matrix multiplication using the rocBLAS library in C. Our goal is to showcase how low-level C implementations can achieve performance parity with high-level libraries like PyTorch, while offering developers greater control over the computational workflow and a deeper understanding of GPU programming concepts.

Matrix Multiplication on CPUs vs. GPUs
--------------------------------------

Implementing matrix multiplication differs significantly between CPUs and GPUs. Understanding these differences sheds light on why GPU acceleration is crucial for deep learning computations.

- **CPUs** are optimized for general-purpose, sequential tasks. They excel at handling smaller workloads, complex operations, and situations requiring low latency for individual operations or when working with sparse matrices. Efficient CPU matrix multiplication in C focuses on cache utilization and instruction-level parallelism. While CPUs can leverage multiple cores through threading, their parallelism is limited by core count, making them less ideal for very large matrix operations.

- **GPUs**, by contrast, are designed for massively parallel computation, making them well-suited for the dense arithmetic operations required by matrix multiplication. GPUs contain thousands of lightweight cores that can perform many matrix operations simultaneously, leading to substantial performance gains for large workloads. Writing GPU code for matrix multiplication involves leveraging specialized libraries, such as AMD's rocBLAS, which provides optimized linear algebra implementations that fully exploit the GPU's parallel architecture.

In this blog, we move from CPU-based matrix multiplication to implementing it on GPUs using AMD's rocBLAS library in C. GPUs handle data differently, relying on parallel execution and optimized memory transfers to achieve high throughput. Understanding these differences is crucial when writing C code that takes advantage of GPU acceleration, allowing us to fully harness the capabilities of GPUs for deep learning tasks.

AMD GPU Programming in C
------------------------

**Why use C instead of PyTorch?**

Using PyTorch offers a high-level, user-friendly interface to perform matrix multiplication on GPUs, abstracting away most of the complexity. However, by writing matrix multiplication in C, we gain direct, low-level control over the GPU and understand the internal workings behind the scenes. This approach is crucial for learning how to optimize and fully exploit the hardware for maximum performance. It also offers an educational perspective, helping us understand what PyTorch does automatically, and provides insights for those who want to go beyond existing frameworks.

**Why rocBLAS?**

rocBLAS is a high-level library provided by AMD that offers efficient GPU implementations of BLAS operations, including matrix multiplication. This is an ideal starting point for programming GPUs with C, as it abstracts many of the complexities of directly writing GPU kernels while still providing a hands-on experience with GPU programming. Writing custom GPU kernels is a complex task and out of scope for this blog. Starting with rocBLAS allows us to learn the fundamentals of GPU programming and gain performance improvements without diving into the intricacies of kernel development right away.

**Why AMD?**

Simply put, AMD is awesome. While there is an abundance of CUDA (NVIDIA) resources available online, there are fewer guides for programming on AMD GPUs, and we wanted to fill that gap. AMD’s ROCm platform provides a powerful environment for GPU programming, and this blog aims to showcase how to effectively use it. Plus, working with AMD GPUs provides a broader perspective for GPU programming, going beyond the NVIDIA-centric focus that is so common in the industry.

GPU Matrix Multiplication with rocBLAS
--------------------------------------

Writing efficient GPU kernels can be challenging, as it requires careful handling of memory access patterns, synchronization, and the coordination of thousands of parallel threads to exploit modern GPU architectures. For tasks like matrix multiplication, starting with optimized libraries such as `rocBLAS` is beneficial, as it provides high-level APIs that abstract away much of the complexity, enabling developers to focus on leveraging GPU acceleration without diving into the intricacies of kernel development.

`rocBLAS` offers a set of optimized linear algebra routines specifically designed for AMD GPUs, making it an ideal choice for efficient matrix multiplication. By using `rocBLAS`, developers can achieve high performance without manually managing low-level GPU features, which can be time-consuming and error-prone. This guide will walk through how to use rocBLAS for implementing matrix multiplication in C, highlighting how to achieve efficient results by utilizing this powerful library.

*Matrix Multiplication Formulas*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's start with the basic matrix multiplication formula. For matrices :math:`A`, :math:`B`, and :math:`C` of dimensions :math:`m \times k`, :math:`k \times n`, and :math:`m \times n` respectively, we can express the multiplication element-wise as:

.. math::

   c_{ij} = \sum_{p=1}^k a_{ip} b_{pj}

where:

- :math:`c_{ij}` is the element in the :math:`i`-th row and :math:`j`-th column of :math:`C`
- :math:`a_{ip}` is the element in the :math:`i`-th row and :math:`p`-th column of :math:`A`
- :math:`b_{pj}` is the element in the :math:`p`-th row and :math:`j`-th column of :math:`B`

This formula shows that each element :math:`c_{ij}` of :math:`C` is calculated by taking the dot product of the :math:`i`-th row of :math:`A` and the :math:`j`-th column of :math:`B`.

While this basic formula is fundamental, many advanced linear algebra libraries, including rocBLAS, use a more sophisticated formula for their General Matrix Multiplication (GEMM) routine. This enhanced formula provides greater flexibility and efficiency in matrix computations.

The rocBLAS GEMM formula can be expressed as:

.. math::

   C = \alpha \cdot \text{op}(A) \cdot \text{op}(B) + \beta \cdot C

Or in element-wise form:

.. math::

   c_{ij} = \alpha \cdot \sum_{p=1}^k \text{op}(a)_{ip} \cdot \text{op}(b)_{pj} + \beta \cdot c_{ij}

These formulas might look intimidating at first, but let's break them down step by step:

The :math:`C` on the right side: 
This :math:`C` represents the initial values in the result matrix. By including it on the right side, we can update existing values instead of always starting from scratch. This is useful in many algorithms that build up a result over multiple steps.

:math:`\alpha` and :math:`\beta`:
These are simple numbers used to adjust the importance of different parts of the calculation. Think of them as volume knobs:
- :math:`\alpha` controls how much of the new multiplication (A·B) we include
- :math:`\beta` controls how much of the original C we keep

:math:`\text{op}(A)` and :math:`\text{op}(B)` (or :math:`\text{op}(a)_{ip}` and :math:`\text{op}(b)_{pj}` in the element-wise form):
Sometimes in matrix operations, we need to flip a matrix on its diagonal (transpose it). Instead of creating a new, flipped matrix, which would take up more memory, we can just pretend we flipped it. That's what :math:`\text{op}()` does - it either leaves the matrix (or element) as-is or treats it as if it were flipped, depending on what we need.

Here's a step-by-step breakdown of what this formula does:

a) Multiply A and B (with possible flipping): :math:`\text{op}(A) \cdot \text{op}(B)` or :math:`\sum_{p=1}^k \text{op}(a)_{ip} \cdot \text{op}(b)_{pj}`
b) Adjust the importance of this multiplication: :math:`\alpha \cdot (\text{op}(A) \cdot \text{op}(B))` or :math:`\alpha \cdot \sum_{p=1}^k \text{op}(a)_{ip} \cdot \text{op}(b)_{pj}`
c) Adjust the importance of the original C: :math:`\beta \cdot C` or :math:`\beta \cdot c_{ij}`
d) Add these together to get the final C: :math:`\alpha \cdot (\text{op}(A) \cdot \text{op}(B)) + \beta \cdot C` or :math:`\alpha \cdot \sum_{p=1}^k \text{op}(a)_{ip} \cdot \text{op}(b)_{pj} + \beta \cdot c_{ij}`

This formula is more flexible than the basic one because:
- It can easily incorporate existing calculations (:math:`\beta \cdot C`)
- It can adjust the balance between new and existing calculations (:math:`\alpha` and :math:`\beta`)
- It can handle flipped matrices without actually flipping them in memory (:math:`\text{op}()`)

This flexibility makes it useful for a wide range of complex calculations in scientific computing, machine learning, and other fields that work with large sets of numbers.

*rocBLAS SGEMM API*
^^^^^^^^^^^^^^^^^^^

The rocBLAS library provides the `rocblas_sgemm` function for single-precision floating-point matrix multiplication. Here's a breakdown of its parameters:

* `handle`: A `rocblas_handle` that manages the library context, created using `rocblas_create_handle()`.
* `transA`, `transB`: Indicate whether matrices A and B are transposed (`rocblas_operation_transpose`) or not (`rocblas_operation_none`).
* `m`, `n`, `k`: Dimensions of the matrices where `m` and `n` define the size of C, and `k` is the shared dimension between A and B.
* `alpha`: Pointer to a scalar multiplier for matrices A and B.
* `A`, `B`: Pointers to matrices A and B in GPU memory.
* `lda`, `ldb`: Leading dimensions of matrices A and B, defining the stride between rows or columns.
* `beta`: Pointer to a scalar multiplier for matrix C.
* `C`: Pointer to matrix C in GPU memory, where the result is stored.
* `ldc`: Leading dimension of matrix C, similar to `lda` and `ldb`.

The general form of the `rocblas_sgemm` function call can be represented mathematically as:

.. math::

   \text{rocblas\_sgemm}(handle, transA, transB, m, n, k, \alpha, A, lda, B, ldb, \beta, C, ldc)

And here's a high-level code snippet demonstrating how to call the `rocblas_sgemm` function:

.. code-block:: c

   rocblas_status rocblas_sgemm(
       rocblas_handle handle,
       rocblas_operation transA, rocblas_operation transB,
       int m, int n, int k,
       const float *alpha,
       const float *A, int lda,
       const float *B, int ldb,
       const float *beta,
       float *C, int ldc
   );

Using this API, you can perform complex matrix multiplications with a single function call, taking advantage of rocBLAS's optimized implementation for AMD GPUs.




Benchmarking Setup and Code Organization
----------------------------------------

*Matrix Configuration and Benchmarking Strategy*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our implementation performs matrix multiplication using the formula C = A x B, where both matrices A and B are square matrices of size N × N. We set N to a static size of 8,192, simplifying the implementation and laying the groundwork for future extensions to non-square matrices. By defining N with a preprocessor C macro (``#define N 8192``), we can enable aggressive compiler optimizations and ensure consistent runtime behavior.

In this setup, we are not implementing separate kernels with varying block sizes because each matrix is fixed at N × N. A kernel typically refers to an optimized code block designed for flexible execution across varying data sizes or hardware conditions. Since N is static in our implementation, the complexity of multiple kernels is unnecessary, allowing us to focus on optimizing for a single, fixed configuration.

Memory Requirements
'''''''''''''''''''

With N = 8,192, each matrix contains 67,108,864 elements. Using 32-bit floating-point precision (often referred to as "single precision" or "FP32"), the size of each matrix (A, B, and C) is calculated as follows:

.. math::

   67,108,864 \times 4 \text{ bytes} = 268,435,456 \text{ bytes} \approx 268 \text{ MB}

This results in a total memory requirement of approximately 805 MB for all three matrices.

Computational Complexity
''''''''''''''''''''''''

Calculating the computational effort for matrix multiplication involves determining the total number of floating point operations (FLOPs) needed. When multiplying two :math:`N \times N` matrices, the resulting matrix is also :math:`N \times N` (:math:`N^2` elements). Each element is the result of a dot product between a row from the first matrix and a column from the second matrix. This involves:

- **Multiplications:** Each element requires multiplying :math:`N` pairs of numbers (one from the row and one from the column).

- **Additions:** The products from the multiplications are then summed together, requiring :math:`N - 1` additions (adding two numbers requires one addition, adding three numbers requires two additions, etc).

Thus, the total number of FLOPs is calculated as:

.. math::

   \text{Total FLOPs} = 2N^3 - N^2

For large matrices, the :math:`2N^3` term contributes primarily to the total FLOPs, so it is often used to estimate the computational effort. This simplifies to:

.. math::

   \text{Total FLOPs} = 2N^3

This simplification highlights how the computational effort grows with the size of the matrices. For our chosen matrix size of 8192 x 8192, this results in:

.. math::

   2 \times 8192^3 = 1,099,511,627,776 \approx 1.1 \text{ TFLOPs}

This large number of operations underscores the computational intensity of large-scale matrix multiplication and highlights the importance of our optimization efforts. It is also important to note the distinction between FLOPs, which measure the total operations required, and FLOPS (Floating Point Operations Per Second), which indicate the system's performance capability.

Cache Considerations
''''''''''''''''''''

We chose this large N value (8,192) to represent a realistic problem size for our matrix multiplication.  With our matrix size of approximately 268MB each, the entire problem (all three matrices) doesn't fit in L3 cache simultaneously, but significant portions of the working set can potentially reside in cache during computation. This creates a scenario where careful cache management becomes crucial for performance. Our setup allows us to:

* Explore the effects of cache blocking and tiling optimizations
* Observe how different algorithms balance cache utilization and main memory access
* Understand performance characteristics that bridge cached and non-cached operations
* Investigate how implementations handle a problem that doesn't neatly fit entirely in cache, but is also not so large as to make cache optimizations irrelevant

This approach provides insight into algorithm design for real-world, cache-sensitive computations.

Benchmarking Environment
''''''''''''''''''''''''
For our benchmarks, we used an AWS c7a.32xlarge instance with the following specifications:

- **Processor:** AMD EPYC 9R14
- **Cores:** 2 sockets, 64 cores per socket (128 cores total, without simultaneous multithreading)
- **L3 Cache:** 512MB

The total working set size is about 805MB (three 268MB matrices), which is larger than the L3 cache. This setup allows us to observe how the cache handles large matrix multiplications and its impact on performance, as the entire workload cannot fit in the cache at once.  This setup ensures the dataset exceeds the cache size, providing a realistic assessment of the algorithm’s performance. 

*Code Structure and Organization*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our matrix multiplication code is organized into separate modules for clarity and maintainability. The primary files are:

* `matmul_lib.c <https://github.com/pebblesandweeds/cpu_matmul/blob/dev/c/src/matmul_lib.c>`_: Contains the core matrix multiplication functions.
* `main.c <https://github.com/pebblesandweeds/cpu_matmul/blob/dev/c/src/main.c>`_: Serves as the entry point, calling functions from ``matmul_lib.c``.
* `Makefile <https://github.com/pebblesandweeds/cpu_matmul/blob/main/c/Makefile>`_: Specifies the build process using the ``gcc`` compiler with optimization flags ``CFLAGS = -mavx2 -fopenmp -O3 -march=native -I./include``

For a detailed overview of our project structure and how we implement various matrix multiplication methods and optimizations, refer to our `README.md <https://github.com/pebblesandweeds/cpu_matmul/blob/dev/README.md#project-structure>`_. The code snippets in this blog exclude `#pragma` directives for simplicity; the full code with parallel instructions is available in the repository.

Naive Matrix Multiplication 
---------------------------

We begin with a basic matrix multiplication method in C to illustrate the fundamental algorithm and its inefficiencies. The following sections will provide a visual representation, the mathematical formula, and the implementation of this approach.

*Visual and Formulaic Representation*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The process is illustrated with an animation showing an 8x8 matrix multiplication. Each frame captures the computation of matrix :math:`C` elements as the sum of products from matrices :math:`A` and :math:`B`.

The corresponding mathematical operation is described by the formula:

.. math::
    C_{ij} = \sum_{k=1}^{N} A_{ik} B_{kj}

*Naive Implementation in C*
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Following this formula, our C code implementation employs three nested loops to perform the matrix multiplication. This basic method is straightforward but not optimized for performance, particularly with large matrices where the computational overhead becomes significant.

.. code-block:: c

   void matmul(float A[N][N], float B[N][N], float C[N][N]) {
       for (int i = 0; i < N; i++) {
           for (int j = 0; j < N; j++) {
               for (int k = 0; k < N; k++) {
                   C[i][j] += A[i][k] * B[k][j];
               }
           }
       }
   }

*Naive Matrix Multiplication Performance* 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This naive approach effectively illustrates the link between algorithmic simplicity and computational inefficiency. With N set to 8,192, the computation involves approximately 1,099.51 billion floating-point operations. Despite the high-end CPU, our AWS c7a.32xlarge instance only achieves a performance of **~25 GFLOPS**.  This demonstrates the significant gap between the naive method's performance and the optimizations needed and sets the stage for exploring more advanced optimization techniques in the following sections.
 
Optimizing Matrix Multiplication
--------------------------------

While the naive matrix multiplication implementation helps understand the basic algorithm, it is inefficient for large matrices.  It processes matrices in row-major order, the default in C, where rows of matrix A are multiplied by columns of matrix B. This access pattern leads to frequent cache misses because it disrupts spatial locality, as matrix elements are stored contiguously in memory. The mismatch between access patterns and memory layout results in poor cache utilization and increased memory latency, significantly impacting performance. 

To address these inefficiencies, we use tiling, blocking, and loop unrolling. Tiling and blocking restructure computations to improve data locality by dividing matrices into smaller blocks, which enhances cache usage. Loop unrolling reduces the overhead of loop control by expanding loops, allowing more operations to be performed in parallel. These methods collectively improve data locality and make better use of CPU caches, significantly enhancing performance. For more detailed information on these techniques, see `Tiling and Blocking <https://en.wikipedia.org/wiki/Loop_nest_optimization#Tiling>`_ and `Loop Unrolling <https://en.wikipedia.org/wiki/Loop_unrolling>`_.

*Optimized Implementation in C*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our optimized matrix multiplication implementation leverages these techniques to minimize cache misses and maximize computational throughput. The following C code demonstrates the use of blocking, tiling, and unrolling to improve performance:

.. code-block:: c

   #define BLOCK_SIZE 64 // Optimizes memory across L1/L2/L3; fetch data in chunks 
   #define TILE_SIZE 32 // Improves CPUs data processing; balances CPU resources and data caching
   #define UNROLL_FACTOR 4 // Increases parallel operations w/out overwhelming memory

   void matmul_scalar(float A[N][N], float B[N][N], float C[N][N]) {
   // Outer loops for block-wise operations
    for (int i = 0; i < N; i += BLOCK_SIZE) {
    for (int j = 0; j < N; j += BLOCK_SIZE) {
    for (int k = 0; k < N; k += BLOCK_SIZE) {
        // Inner loops for tile-wise operations within blocks
        for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ii += TILE_SIZE) {
        for (int jj = j; jj < j + BLOCK_SIZE && jj < N; jj += TILE_SIZE) {
        // Loop unrolling for innermost loop
        for (int kk = k; kk < k + BLOCK_SIZE && kk < N; kk += UNROLL_FACTOR) {
            float c_temp = C[ii][jj]; // Temp variable for accumulation
            // Compute on tiles
            for (int iii = ii; iii < ii + TILE_SIZE && iii < i + BLOCK_SIZE && iii < N; iii++) {
            for (int jjj = jj; jjj < jj + TILE_SIZE && jjj < j + BLOCK_SIZE && jjj < N; jjj++) {
                // Matrix multiplication within a tile
                c_temp += A[iii][kk] * B[kk][jjj];
            }
            C[iii][jjj] = c_temp; // Store accumulated results
            }
        }
        }
        }
    }
    }
    }
   }

*Optimized Matrix Multiplication Performance*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By optimizing matrix multiplication, we achieve a significant performance boost. Our approach in the code above employs three key strategies: dividing matrices into cache-friendly blocks, further subdividing into efficiently processable tiles, and using loop unrolling for parallel operations. These techniques work together to ensure optimal data availability and CPU resource utilization.

On the AWS c7a.32xlarge instance, this optimized implementation achieves approximately **500 GFLOPS**, representing more than a *20x increase* over the naive approach. This improvement stems from better use of the CPU's cache hierarchy, reduced memory access times, and increased instruction-level parallelism. While further scalar optimizations are possible, we're approaching the limits of what can be achieved without leveraging more advanced hardware features. The next step in boosting performance is to utilize vectorized operations, which we'll explore in the following section.

Vectorized Matrix Multiplication
--------------------------------

*Scalar vs. Vectorized Operations*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scalar operations process data one element at a time, performing calculations sequentially. In contrast, vectorized operations use a Single Instruction, Multiple Data (SIMD) approach, processing multiple data elements simultaneously. This parallelism is implemented on CPUs through SIMD instructions, which leverage hardware capabilities to execute the same operation on multiple data points in a single instruction cycle.

To write vectorized code, several elements are necessary:

1. **SIMD Instructions**: SIMD instructions, such as AVX, enable parallel processing by applying the same operation across multiple data elements in a single instruction. This includes `Fused Multiply-Add (FMA) <https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation>`_, which performs multiplication and addition together. For more information on SIMD, see `Wikipedia <https://en.wikipedia.org/wiki/SIMD>`_. 

2. **Data Alignment**: Properly aligning data in memory is crucial for SIMD processing. Aligned data ensures that SIMD instructions can access data efficiently, avoiding costly misaligned memory accesses. Learn more about `Data Alignment <https://en.wikipedia.org/wiki/Data_structure_alignment>`_. 

3. **Loop Unrolling**: Loop unrolling enhances vectorized operations by expanding loop iterations, reducing overhead, and allowing more operations to be performed in parallel. This technique improves the efficiency of SIMD instructions. More details can be found at `Loop Unrolling <https://en.wikipedia.org/wiki/Loop_unrolling>`_.
 
4. **Prefetching**: Prefetching involves loading data into the CPU cache before it is needed, reducing cache misses and ensuring that data is readily available when required. This technique optimizes memory access patterns and improves performance. Learn about `Prefetching <https://en.wikipedia.org/wiki/Cache_prefetching>`_. 

5. **Transposition**: Matrix transposition rearranges data to improve access patterns, particularly for matrix operations. By aligning data in a more efficient layout, transposition reduces cache misses and speeds up computations. For more on this, see `Matrix Transposition <https://en.wikipedia.org/wiki/Transpose>`_. 

*Vectorized Implementation in C*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is the C implementation of matrix multiplication using vectorization techniques to enhance performance:

.. code-block:: c

   void matmul_vectorized(float A[N][N], float B[N][N], float C[N][N]) {
       // Data alignment (allocate memory for B_col)
       float (*B_col)[N] = aligned_alloc(32, N * N * sizeof(float));
       if (B_col == NULL) {
           fprintf(stderr, "Memory allocation failed\n");
           exit(1);
       }
       // Transposition (transpose B into B_col for better memory access patterns)
       for (int j = 0; j < N; j += 32) {
           for (int k = 0; k < N; k++) {
               for (int jj = 0; jj < 32 && j + jj < N; jj++) {
                   B_col[j+jj][k] = B[k][j+jj];
               }
           }
       }
       {
           for (int j = 0; j < N; j += 32) {
               for (int i = 0; i < N; i += 32) {
                   // SIMD instructions (__m256 for 256-bit for SIMD operations)
                   __m256 c[32][32];
                   for (int ii = 0; ii < 32; ii++) {
                       for (int jj = 0; jj < 32; jj++) {
                           c[ii][jj] = _mm256_setzero_ps();
                       }
                   }
                   for (int k = 0; k < N; k += 32) {
                       // Prefetching (fetch data into cache before we use it)
                       if (k + 128 < N) {
                           for (int ii = 0; ii < 32; ii++) {
                               _mm_prefetch((char*)&A[i+ii][k + 128], _MM_HINT_T1);
                               _mm_prefetch((char*)&B_col[j+ii][k + 128], _MM_HINT_T1);
                           }
                       }
                       __m256 a[32][4], b[32][4];
                       for (int ii = 0; ii < 32; ii++) {
                           for (int kk = 0; kk < 4; kk++) {
                               a[ii][kk] = _mm256_loadu_ps(&A[i+ii][k+kk*8]);
                               b[ii][kk] = _mm256_load_ps(&B_col[j+ii][k+kk*8]);
                           }
                       }
                       // Loop unrolling (unroll inner loop for vector operations) and FMA (fused multiply-add)
                       for (int ii = 0; ii < 32; ii++) {
                           for (int jj = 0; jj < 32; jj++) {
                               c[ii][jj] = _mm256_fmadd_ps(a[ii][0], b[jj][0], c[ii][jj]);
                               c[ii][jj] = _mm256_fmadd_ps(a[ii][1], b[jj][1], c[ii][jj]);
                               c[ii][jj] = _mm256_fmadd_ps(a[ii][2], b[jj][2], c[ii][jj]);
                               c[ii][jj] = _mm256_fmadd_ps(a[ii][3], b[jj][3], c[ii][jj]);
                           }
                       }
                   }
                   // SIMD Instructions (final matrix multiplication reduction using SIMD)
                   for (int ii = 0; ii < 32 && i + ii < N; ii++) {
                       for (int jj = 0; jj < 32 && j + jj < N; jj++) {
                           __m256 sum = c[ii][jj];
                           __m128 sum_high = _mm256_extractf128_ps(sum, 1);
                           __m128 sum_low = _mm256_castps256_ps128(sum);
                           __m128 sum_all = _mm_add_ps(sum_high, sum_low);
                           sum_all = _mm_hadd_ps(sum_all, sum_all);
                           sum_all = _mm_hadd_ps(sum_all, sum_all);
                           float result = _mm_cvtss_f32(sum_all);
                           C[i+ii][j+jj] += result;
                       }
                   }
               }
           }
       }
       free(B_col);
   }

*Performance Improvement*
^^^^^^^^^^^^^^^^^^^^^^^^^

The vectorized implementation greatly improves performance by applying the vectorized techniques described earlier. Data alignment optimizes memory access for SIMD operations, while transposition refines data layout to enhance access patterns for matrix operations. SIMD instructions and 256-bit AVX `YMM registers <https://en.wikipedia.org/wiki/Processor_register>`_ enable parallel processing of up to eight single-precision floating-point numbers per cycle, boosting data throughput. Prefetching reduces cache misses by pre-loading data, and loop unrolling enhances vector operation efficiency by cutting loop overhead and allowing more parallel instruction execution. These combined techniques leverage the CPU’s vectorization capabilities to deliver substantial performance gains.

On the AWS c7a.32xlarge instance, this vectorized approach achieves approximately **3,000 GFLOPS**, representing a *6x performance increase* over the previously optimized scalar implementation.  This contrast underscores the efficiency of vectorized operations, which use SIMD to process multiple data elements simultaneously along with our other alighment optimizations.  This significant performance gain highlights the effectiveness of these advanced techniques in enhancing computational efficiency for large-scale matrix operations. 

Conclusion
----------

Our exploration of matrix multiplication optimization reveals significant performance gains. Starting with a naive implementation at 25 GFLOPS, we improved to 500 GFLOPS with scalar optimization, marking a 20x increase. Vectorized operations then further boosted performance to 3,000 GFLOPS, achieving a 120x improvement from the initial implementation. This progress highlights the impact of optimizations such as cache-friendly blocking, efficient tiling, and SIMD vectorization.

Our vectorized C implementation nearly matches NumPy's 3,500 GFLOPS, showing the effectiveness of low-level optimizations. This experience with CPU optimizations enhances our understanding of memory management and parallelism, providing a strong foundation for future GPU optimizations, where similar principles will be applied in a different context.

Thanks for reading, more details can be our `cpu_matmul <https://github.com/pebblesandweeds/cpu_matmul>`_ Github repo. Stay tuned for our next blog, where we will explore matrix multiplication optimizations on GPUs.

Further Reading
---------------

* `GEMM Optimization Tutorial <https://github.com/flame/how-to-optimize-gemm>`_ and `BLISlab Tutorial <https://github.com/flame/blislab/blob/master/tutorial.pdf>`_
* `Beating NumPy in 150 lines of C Code <https://salykova.github.io/matmul-cpu>`_ plus the `repo <https://github.com/salykova/matmul.c>`_
* George Hotz's six hour video stream `Can You Mutliply a Matrix? <https://youtu.be/VgSQ1GOC86s?si=HP1VB1UDF384_xQt>`_ and `gemm.c code <https://github.com/tinygrad/tinygrad/blob/master/extra/gemm/gemm.c>`_
