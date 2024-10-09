Accelerating Matrix Multiplication on AMD GPUs with rocBLAS in C
================================================================

.. admonition:: Highlights 

 Matrix multiplication is the core operation behind deep learning, driving the computations in neural networks for model training, fine-tuning, and inference. This blog post demonstrates how AMD's rocBLAS library can be used in C to achieve matrix multiplication performance comparable to PyTorch's implementation, leveraging low-level control for efficient use of GPUs.

 - **Problem Scale**: We perform multiplication of two 16,384 x 16,384 matrices, requiring ~3.21 GB of memory and ~8.8 TFLOPs of computation.

 - **PyTorch Baseline**: Achieves **~37.5 TFLOPS** using `this simple code <https://github.com/pebblesandweeds/gpu_matmul/blob/main/pytorch/pytorch_matmul.py>`_. PyTorch's high-level API (``torch.matmul``) abstracts the underlying rocBLAS operations, providing ease of use without sacrificing performance.

 - **rocBLAS Implementation in C**: Matches PyTorch at **~37.5 TFLOPS** with `this C implementation <https://github.com/pebblesandweeds/gpu_matmul/blob/main/c/src/matrix_operations.c>`_. By directly calling ``rocblas_sgemm()``, we expose GPU programming concepts like memory allocation, data transfer, and operation parameters which provide insight into the underlying processes that high-level APIs abstract away.

 - **Performance Gain**: Our GPU implementation achieves a 12.5x speedup over our `optimized CPU version <https://github.com/pebblesandweeds/cpu_matmul/blob/main/c/src/matmul_lib.c>`_ (3 TFLOPS to 37.5 GFLOPS).

 - **Accuracy Verification**: The C implementation includes spot-checking to verify GPU computation accuracy against CPU results.

 This comparison showcases how low-level C programming with rocBLAS can achieve performance parity with high-level frameworks like PyTorch. The C implementation offers a valuable learning opportunity, introducing developers to GPU programming concepts and serves as a bridge between high-level APIs and custom GPU kernel development, providing a deeper understanding of GPU computing without sacrificing efficiency.

 All benchmarks were run on an AMD Instinct MI250X GPU, demonstrating the capabilities of AMD's high-performance hardware for deep learning.

 Get all of the code `in this repo <https://github.com/pebblesandweeds/gpu_matmul>`_.

Introduction
------------

Matrix multiplication is fundamental to deep learning, powering neural network computations in both forward propagation and backpropagation.  In `our previous blog post <https://blog.pebblesandweeds.com/cpu_matmul_blog.html#why-is-matrix-multiplication-important>`_, we explored implementing matrix multiplication from scratch in C on `AMD EPYC CPUs <https://aws.amazon.com/ec2/instance-types/c7a/>`_. This CPU implementation laid the groundwork for understanding the core principles behind matrix multiplication, preparing us to focus next on GPU-accelerated computations.

Building on the CPU-based matrix multiplication foundation, this blog post extends our exploration to GPU-based computations. We demonstrate how to harness the power of AMD GPUs for high-performance matrix multiplication using the `rocBLAS library <https://github.com/rocm/rocBLAS>`_ in C. While rocBLAS is not as low-level as building custom GPU kernels from scratch, it provides a middle ground, offering more granular control than high-level libraries like PyTorch. Our goal is to showcase how this C implementation with rocBLAS can achieve performance parity with high-level libraries, while offering developers greater insight into GPU resource management and provide a deeper understanding of GPU programming concepts without the complexity of writing kernels from scratch.

Matrix Multiplication: CPUs vs. GPUs
------------------------------------

Implementing matrix multiplication differs significantly between CPUs and GPUs, largely due to the way each architecture handles parallelism and memory access. These differences impact how we approach performance optimization for large-scale operations.

- **CPUs** are optimized for general-purpose, sequential tasks. They excel at handling smaller workloads, complex operations, and situations requiring low latency for individual operations or when working with sparse matrices. Efficient CPU matrix multiplication in C focuses on cache utilization and instruction-level parallelism. While CPUs can leverage multiple cores through threading, their parallelism is limited by core count, making them less ideal for very large matrix operations.

- **GPUs**, by contrast, are designed for massively parallel computation, making them well-suited for the dense arithmetic operations required by matrix multiplication. GPUs contain thousands of lightweight cores that can perform many matrix operations simultaneously, leading to substantial performance gains for large workloads.

Modern GPUs are equipped with various computation units, such as `SIMD processors <https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/#registers-and-occupancy>`_ and specialized matrix multiplication units, known as Tensor Cores in NVIDIA GPUs and Matrix Cores in AMD GPUs. While SIMD units offer flexibility, dedicated units like Matrix Cores deliver significantly higher performance for matrix operations. Writing efficient GPU code requires using specialized libraries that fully utilize these architectural features. Libraries such as AMD’s rocBLAS for AMD GPUs and NVIDIA’s cuBLAS are designed to harness these matrix units, providing performance far beyond what general-purpose GPU code can achieve.

In this blog, we shift from CPU-based matrix multiplication to implementing it on GPUs using AMD's rocBLAS library in C. GPUs process data differently, leveraging parallel execution and optimized memory transfers to achieve high throughput. Understanding these differences is essential when writing C code that fully utilizes GPU capabilities, providing the foundation for more complex deep learning tasks in frameworks like PyTorch.

AMD GPU Programming in C
------------------------

**Why use C instead of PyTorch?**

Using PyTorch provides a high-level, user-friendly interface for performing matrix multiplication on GPUs, abstracting away much of the complexity. However, writing matrix multiplication in C gives us direct, low-level control over the GPU, offering insight into how the hardware operates behind the scenes. This understanding is key for those looking to write custom GPU kernels in C/C++ in the future, as it helps in optimizing code and fully exploiting the hardware for maximum performance. It also offers a deeper understanding of what PyTorch handles automatically, equipping developers with the knowledge needed to go beyond existing frameworks.

**Why rocBLAS?**

rocBLAS is a high-level library provided by AMD that offers efficient GPU implementations of BLAS operations, including matrix multiplication. This is an ideal starting point for programming GPUs with C, as it abstracts many of the complexities of directly writing GPU kernels while still providing a hands-on experience with GPU programming.  Starting with rocBLAS allows us to learn the fundamentals of GPU programming and gain performance improvements without diving into the intricacies of kernel development right away.

**Why AMD?**

Because AMD is awesome! While there are an abundance of CUDA (NVIDIA) resources available online, there are fewer guides for programming on AMD GPUs, and we wanted to fill that gap. AMD’s ROCm platform provides a complete environment for GPU programming, and this blog aims to showcase how to effectively use just a small piece of the ROCm toolkit. Lastly, working with AMD GPUs also broadens a wider  understanding of GPU programming by moving beyond the prevalent NVIDIA-centric approach in the industry.

GPU Matrix Multiplication with rocBLAS
--------------------------------------

Writing efficient GPU kernels involves managing memory access patterns, synchronization, and the coordination of thousands of parallel threads to fully exploit modern GPU architectures. For matrix multiplication, using an optimized library like rocBLAS simplifies this process by providing a range of APIs that abstract away much of the complexity. This allows developers to take advantage of GPU computation without needing to manually manage the intricacies of kernel development.

rocBLAS contains numerous optimized linear algebra routines tailored for AMD GPUs. In this section, we will focus on a single function, `sgemm`, which handles single precision (fp32) matrix multiplication. This function represents a small part of the larger rocBLAS library, which is designed to optimize performance while minimizing the need for low-level management of GPU operations. By leveraging rocBLAS, developers can achieve high performance for matrix multiplication in C without the overhead of manual GPU feature management.


*Matrix Multiplication Formulas*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's start with the basic matrix multiplication formula. Consider three matrices A, B, and C with the following dimensions:

.. math::
   A &= m \times k \\
   B &= k \times n \\
   C &= m \times n

The matrix multiplication of A and B resulting in C can be expressed as:

.. math::

   C = A \cdot B

On an element-wise level, this operation can be written as:

.. math::

   c_{ij} = \sum_{p=1}^k a_{ip} b_{pj}

Here, :math:`c_{ij}` represents the element in the i-th row and j-th column of C, calculated by taking the dot product of the i-th row of A and the j-th column of B. The indices i, j, and p range from 1 to m, n, and k respectively.

This formula demonstrates how each element of the resulting matrix C is computed through a series of multiplications and additions, utilizing corresponding elements from matrices A and B.

While this basic formula is fundamental, many advanced linear algebra libraries, including rocBLAS, use a more sophisticated formula for their General Matrix Multiplication (GEMM) routine. This enhanced formula provides greater flexibility and efficiency in matrix computations.

The rocBLAS GEMM formula can be expressed as:

.. math::

   C = \alpha \cdot \text{op}(A) \cdot \text{op}(B) + \beta \cdot C

Or in element-wise form:

.. math::

   c_{ij} = \alpha \cdot \sum_{p=1}^k \text{op}(a)_{ip} \cdot \text{op}(b)_{pj} + \beta \cdot c_{ij}

These formulas might look intimidating at first, but let's break them down:

* **C on both sides:** The :math:`C` on the right side represents the initial values in the result matrix. This allows for updating existing values instead of starting from scratch, useful in algorithms that build up results over multiple steps. The final step adds this scaled original C (:math:`\beta \cdot C`) to the new multiplication result.

* **α and β:** These scalar values adjust the importance of different parts of the calculation. Think of them as volume knobs - :math:`\alpha` controls the contribution of the new multiplication (A·B), while :math:`\beta` determines how much of the original C to retain. This allows for fine-tuning the balance between new and existing calculations.

* **op(A) and op(B):** The :math:`\text{op}()` function allows for matrix transposition without creating a new matrix. It either leaves the matrix as-is or treats it as if it were transposed, depending on the operation needed.  Transposition within the rocBLAS GEMM has performance implications that we typically try to avoid by transposing matrices where required prior to calling the GEMM API.   

This formula offers greater flexibility than the basic matrix multiplication:

* **Memory efficiency**:
  By using :math:`\text{op}()`, it avoids creating new copies of transposed matrices, saving memory allocations and reducing data movement when required.
* **Computational versatility**:
  The :math:`\alpha` and :math:`\beta` parameters enable a wide range of operations beyond simple multiplication, such as blending results from multiple calculations or performing iterative updates in complex algorithms.

Although this formula is valuable in scientific computing and specialized machine learning, typical deep learning scenarios often use simplified versions. For standard neural network operations:

* :math:`\alpha` is usually set to 1 since we want to scale the result of the matrix multiplication directly without any changes.
* :math:`\beta` is typically 0 because we often ignore any pre-existing values in the output matrix, focusing only on the new result. In some cases, such as gradient accumulation during backpropagation, :math:`\beta` may be set to 1 (or other values) to retain and add to previous values.

The rocBLAS GEMM formula extends basic matrix multiplication with flexible operations and scaling factors, allowing efficient handling of transposed matrices and in-place updates. While it offers broad capabilities for scientific computing, deep learning commonly uses simplified versions with α set to 1 and β to 0 or 1, depending on the operation. 

*rocBLAS SGEMM API*
^^^^^^^^^^^^^^^^^^^

The `rocblas_sgemm` function in the rocBLAS library performs single-precision floating-point matrix multiplication (SGEMM). Here's a breakdown of its key components for those unfamiliar with GPU programming:

* **handle**: A `rocblas_handle` manages the internal state and resources of the rocBLAS library and is created with `rocblas_create_handle()` before performing any operations.
* **transA**, **transB**: These parameters specify whether matrices A and B should be transposed before multiplication. Use `rocblas_operation_none` for no transpose or `rocblas_operation_transpose` to transpose the matrix.
* **m**, **n**, **k**: These define the dimensions of the matrices. `m` and `n` are the rows and columns of matrix C (the result), while `k` is the shared dimension between A and B.
* **alpha**, **beta**: These scalar values control how the result of `A * B` is combined with matrix C. `alpha` scales `A * B`, and `beta` scales any existing values in matrix C.
* **A**, **B**, **C**: These are **pointers to the matrices in GPU memory**. The matrices (A, B, and C) exist on the host initially, but they must be copied to the GPU using device pointers (`d_A`, `d_B`, `d_C`). These device pointers are passed to `rocblas_sgemm`, not the host pointers.
* **lda**, **ldb**, **ldc**: These are the leading dimensions of matrices A, B, and C, which define the stride between rows or columns, ensuring proper memory layout.

Here’s a high-level code snippet showing how to call `rocblas_sgemm`:

.. code-block:: c

   rocblas_sgemm(handle,
                 transA, transB,
                 m, n, k,
                 &alpha,
                 d_A, lda,
                 d_B, ldb,
                 &beta,
                 d_C, ldc);

   // where:
   // handle:     rocblas_handle managing the rocBLAS context.
   // transA/B:   rocblas_operation_none (no transpose) or rocblas_operation_transpose (use the transposed matrix).
   // m, n, k:    Matrix dimensions; m = rows of C, n = cols of C, k = shared dimension of A and B.
   // alpha:      Scalar pointer, scales A * B.
   // d_A, d_B:       Pointers to matrices A and B in GPU memory.
   // lda/ldb:    Leading dimensions of A and B (stride between rows/cols).
   // beta:       Scalar pointer, scales existing values in C.
   // d_C:          Pointer to output matrix C in GPU memory.
   // ldc:        Leading dimension of matrix C.

Using this API, you can perform complex matrix multiplications with a single function call, taking advantage of rocBLAS's optimized implementation for AMD GPUs.

*From Formulas to Implementation*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our project code demonstrates two approaches to implementing GPU-accelerated matrix multiplication, both leveraging the GEMM formula and rocBLAS:

`PyTorch Implementation <https://github.com/pebblesandweeds/gpu_matmul/blob/main/pytorch/pytorch_matmul.py>`_:
PyTorch's ``torch.matmul`` function simplifies GPU programming by abstracting the complexities of the rocBLAS API (assuming PyTorch is installed with ROCm support). It internally uses the GEMM formula and rocBLAS on AMD GPUs, automatically managing memory allocation, data transfers, and API calls. This high-level approach allows developers to focus on algorithm design without dealing with low-level GPU details.

`Direct C Implementation with rocBLAS <https://github.com/pebblesandweeds/gpu_matmul/blob/main/c/src/main.c>`_:
Our C implementation directly interfaces with the rocBLAS API, providing greater control over the entire computation process. In this case, we manually handle rocBLAS API calls, GPU memory management, and matrix operations. We translate the GEMM formula:

:math:`C = \alpha \cdot \text{op}(A) \cdot \text{op}(B) + \beta \cdot C`

into the following rocBLAS function call:

.. code-block:: c

   CHECK_ROCBLAS(rocblas_sgemm(handle,
                               rocblas_operation_none, rocblas_operation_none,
                               N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));

In this example, matrices `A`, `B`, and `C` are initially in host memory and need to be `moved to GPU memory <https://github.com/pebblesandweeds/gpu_matmul/blob/12a4b4cad727afe1b0fe2cb633933d4af1cfaab1/c/src/timer.c#L4>`_ as `d_A`, `d_B`, and `d_C`. These device pointers are then passed to the `rocblas_sgemm` function instead of the host pointers.

We work with square matrices of size N x N, which is why we use 'N' for the dimensions in the rocBLAS API call. Similarly, the leading dimensions `lda`, `ldb`, and `ldc` are all set to 'N' since the matrices are stored contiguously.

To optimize performance, we transpose matrices A and B before passing them to GEMM. While matrices in C are typically initialized in row-major order, rocBLAS performs better with column-major order. We use a separate function to handle the transposition, as this consistently outperforms using the transpose flags during the `rocblas_sgemm` call.

Key variables in the API call:

- ``handle``: The rocBLAS library handle.
- ``rocblas_operation_none``: Specifies no transposition for input matrices.
- ``N``: The dimensions of our square matrices.
- ``alpha`` and ``beta``: Scalar multipliers in the GEMM formula.
- ``d_A``, ``d_B``, ``d_C``: Pointers to device (GPU) memory for matrices A, B, and C.

The GEMM formula serves as the foundation for both our PyTorch and C implementations. PyTorch abstracts the complexity of GPU programming, enabling rapid development, while our C implementation offers finer control, demonstrating performance improvements by pre-transposing matrices. These approaches illustrate how the same underlying formula can be applied across different programming paradigms to meet specific performance needs in GPU-accelerated matrix multiplication.

Matrix Setup and Code Breakdown
-------------------------------

Matrix Setup For Benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our matrix multiplication operates on square matrices `A` and `B`, both of size N × N. For benchmarking, we've set N to 16,384, which provides a significant workload to demonstrate GPU performance. This configuration is defined using a preprocessor macro (``#define N 16384``), enabling consistent behavior and compiler optimizations.

With N = 16,384, each matrix has 268,435,456 elements. Using 32-bit floating-point precision (FP32), the size of each matrix is:

.. math::

       268,435,456 \times 4 \text{ bytes} = 1,073,741,824 \text{ bytes} \approx 1.07 \text{ GB}

Thus, the total memory requirement for three matrices (A, B, and C) is around 3.21 GB.

The computation involved in multiplying two matrices of this size is intensive. The total number of floating-point operations (FLOPs) required is:

.. math::

       \text{Total FLOPs} = 2N^3 = 2 \times 16,384^3 = 8,796,093,022,208 \approx 8.8 \text{ TFLOPs}

It's important to note that our benchmarks focus solely on the GPU performance during matrix multiplication. We are **not** including the time spent on matrix initialization, the transfer of matrices between host and device memory, or the transfer of results back to the host. This isolation ensures a more accurate representation of the GPU's computational performance.

We conducted benchmarks on a system with dual AMD EPYC 7713 64-Core Processors, 1 TB RAM, and a single AMD MI250 GPU to handle the matrix multiplication. Although the CPU handles tasks like matrix initialization and transposition, the benchmarks focus exclusively on the GPU's contribution during the matrix multiplication phase. This approach allows us to achieve consistent comparisons between different implementations, reporting the achieved TFLOPs for the multiplication step.

Project Structure and Code Organization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our project includes both a low-level C implementation using rocBLAS and a high-level PyTorch implementation, enabling a clear comparison between the two approaches.

In the C implementation, the code is divided into the following key components:

- ``main.c``: Contains the primary logic for benchmarking and running the multiplication.
- ``matrix_operations.c``: Implements the matrix multiplication logic using rocBLAS.
- ``utils.c``: Provides functions for memory management and data initialization.
- ``timer.c``: Includes functions for accurate timing of matrix operations.
- ``spot_check.c``: Verifies the correctness of the matrix multiplication results.

Header files in the ``include/`` directory define the interfaces for these components, ensuring modularity and reusability.

The PyTorch implementation is contained in a single file, ``pytorch_matmul.py``, which abstracts away the complexities of GPU memory management and API calls. This high-level framework simplifies the process of performing matrix multiplication on GPUs, making development faster and more convenient.

The project structure highlights the trade-offs between the detailed control offered by the C implementation and the simplicity and ease of PyTorch. Both approaches utilize GPU acceleration, but they offer different levels of flexibility depending on the user’s needs.

PyTorch Implementation: Abstracting rocBLAS
-------------------------------------------

Key Implementation Details
^^^^^^^^^^^^^^^^^^^^^^^^^^

The PyTorch implementation showcases the simplicity of using a high-level framework for GPU-accelerated matrix multiplication. In this approach, rocBLAS is abstracted away, allowing us to focus on the core computation without dealing with low-level GPU programming details.

Matrix Setup
^^^^^^^^^^^^

.. code-block:: python

   N = 16384
   device = torch.device(f"cuda:{gpu_id}")
   A = torch.empty(N, N, dtype=torch.float32, device=device).uniform_(-1,1)
   B = torch.empty(N, N, dtype=torch.float32, device=device).uniform_(-1,1)

This code initializes two 16384x16384 matrices with random values directly on the GPU by specifying the `device=device` argument. PyTorch internally handles allocating and transferring these matrices to the GPU, so `A` and `B` reside in GPU memory right from the start. No explicit host-to-device memory transfer is needed, as would be required in lower-level frameworks.

Matrix Multiplication
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   torch.matmul(A, B)

This single line performs the entire matrix multiplication operation, leveraging PyTorch's optimized backend (which uses rocBLAS for AMD GPUs).

FLOPS Calculation
^^^^^^^^^^^^^^^^^

.. code-block:: python

   torch.cuda.synchronize()
   start = time.perf_counter()
   torch.matmul(A, B)
   torch.cuda.synchronize()
   end = time.perf_counter()
   run_time = end - start
   tflops = (2 * N**3 / run_time) / 1e12

To accurately measure `run_time`, we use `torch.cuda.synchronize()` to ensure that the matrix multiplication is fully completed on the GPU before and after calling `torch.matmul`. This prevents asynchronous execution from affecting the timing. We use `time.perf_counter()` from the Python standard library for high-resolution timing, but it must be combined with GPU synchronization to reflect only the time spent on the actual computation, not the queuing of operations.

Benchmark Strategy
^^^^^^^^^^^^^^^^^^

The benchmark runs the matrix multiplication 25 times to get a stable performance number. The first run is typically slower because PyTorch needs to load and compile the rocBLAS kernel. Subsequent runs benefit from this initialization and show more consistent performance.

Results Summary
^^^^^^^^^^^^^^^

The benchmark results show:

- First run: 1.74 TFLOPS (5.066478 seconds)
- Subsequent runs: Consistently around 37.5 TFLOPS (0.234 seconds)

Example output:

.. code-block:: text

   Run     Time (s)        TFLOPS
   ------------------------------
   1       5.066478        1.74
   2       0.234706        37.48
   3       0.234577        37.50
   ...
   25      0.234543        37.50

The stark difference between the first run and subsequent runs clearly demonstrates the overhead of initializing the GPU kernel. After initialization, we see stable performance at about 37.5 TFLOPS, showcasing the impressive computational capabilities of the AMD Instinct MI250X GPU for large-scale matrix multiplication tasks.

This PyTorch implementation demonstrates how high-level frameworks can abstract away the complexities of GPU programming while still delivering excellent performance for computational tasks like matrix multiplication.

C Implementation: Direct rocBLAS Integration
--------------------------------------------

Key Implementation Details
^^^^^^^^^^^^^^^^^^^^^^^^^^

The C implementation provides a lower-level approach, directly integrating with rocBLAS for GPU-accelerated matrix multiplication. This method offers more control over the computation process but requires more detailed management of GPU resources.

Matrix Setup
^^^^^^^^^^^^

.. code-block:: c

   size_t size = N * N * sizeof(float);
   float *h_A, *h_B, *h_C, *h_A_trans, *h_B_trans, *h_C_trans;
   float *d_A, *d_B, *d_C;
   
   // Allocate host memory
   h_A = (float*)malloc(size);
   h_B = (float*)malloc(size);
   h_C = (float*)malloc(size);
   
   // Initialize matrices
   initialize_matrices(h_A, h_B, N);
   
   // Allocate device memory
   CHECK_HIP(hipMalloc(&d_A, size));
   CHECK_HIP(hipMalloc(&d_B, size));
   CHECK_HIP(hipMalloc(&d_C, size));

This code allocates memory for matrices on both the host and device, and initializes the input matrices.

Matrix Multiplication
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   rocblas_handle handle;
   CHECK_ROCBLAS(rocblas_create_handle(&handle));
   perform_matrix_multiplication(handle, d_A, d_B, d_C, N, NUM_RUNS);

The matrix multiplication is performed using rocBLAS's `rocblas_sgemm` function, which is called within the `perform_matrix_multiplication` function.

FLOPS Calculation
^^^^^^^^^^^^^^^^^

.. code-block:: c

   double total_flops = 2.0 * N * N * N;
   double tflops = total_flops / (seconds * 1e12);

Similar to the PyTorch implementation, we calculate FLOPS as 2N³, accounting for N³ multiplications and N³ additions.

Benchmark Strategy
^^^^^^^^^^^^^^^^^^

The benchmark runs the matrix multiplication 25 times, with the first run typically being slower due to the initial loading and compilation of the rocBLAS kernel. Subsequent runs show more consistent performance.

Results Summary
^^^^^^^^^^^^^^^

The benchmark results show:

- First run: 2.40 TFLOPS (3669.096191 ms)
- Subsequent runs: Consistently around 37.5 TFLOPS (234 ms)

Example output:

.. code-block:: text

   Run 1: Matrix multiplication time: 3669.096191 ms, Performance: 2.40 TFLOPS
   Run 2: Matrix multiplication time: 234.542786 ms, Performance: 37.50 TFLOPS
   Run 3: Matrix multiplication time: 234.463577 ms, Performance: 37.52 TFLOPS
   ...
   Run 25: Matrix multiplication time: 234.464218 ms, Performance: 37.52 TFLOPS

The performance difference between the first and subsequent runs demonstrates the overhead of initializing the GPU kernel. After initialization, we see stable performance at about 37.5 TFLOPS, matching the performance of the PyTorch implementation.

Accuracy Verification
^^^^^^^^^^^^^^^^^^^^^

Unlike the PyTorch implementation, this C implementation includes a spot-checking mechanism to verify the accuracy of the GPU computations:

.. code-block:: c

   spot_check(h_A, h_B, h_C_trans, N);

This function performs random spot checks, comparing the GPU results with CPU-computed results to ensure accuracy within a specified threshold.

The output confirms the accuracy:

.. code-block:: text

   Performing random spot checks between CPU and GPU results...
   Success: All 50 spot checks passed within the relative error threshold.

This C implementation with direct rocBLAS integration offers fine-grained control over the matrix multiplication process while achieving performance equivalent to the high-level PyTorch implementation. The addition of accuracy verification provides an extra layer of confidence in the results.

Conclusion
----------

Our exploration of GPU-accelerated matrix multiplication using AMD's rocBLAS library has demonstrated the impressive performance capabilities of modern GPUs. We achieved consistent performance of about 37.5 TFLOPS for a 16384x16384 matrix multiplication, showcasing the power of GPU acceleration for large-scale computational tasks.

Both our PyTorch and C implementations reached similar performance levels, highlighting that low-level C programming with rocBLAS can match the efficiency of high-level frameworks like PyTorch. This comparison underscores the value of understanding both high-level abstractions and low-level GPU programming concepts.

The C implementation, while more complex, offers greater control and insight into the GPU computation process. It allowed us to directly manage memory allocation, data transfer, and rocBLAS function calls, providing a deeper understanding of GPU programming principles. The addition of accuracy verification through spot-checking adds an extra layer of confidence in our results.

This journey from CPU to GPU optimization showcases the significant performance gains possible with GPU acceleration. While our previous CPU optimizations achieved 3,000 GFLOPS, the GPU implementation reached 37,500 GFLOPS - a further 12.5x improvement. This leap in performance illustrates the transformative potential of GPU computing for matrix multiplication and, by extension, for deep learning and scientific computing applications.

Thanks for reading! For more details, check out our `gpu_matmul GitHub repo <https://github.com/pebblesandweeds/gpu_matmul>`_. Stay tuned for future blogs where we'll dive deeper into GPU optimizations and explore more advanced topics in high-performance computing.

Further Reading
---------------

* `GEMM Optimization Tutorial <https://github.com/flame/how-to-optimize-gemm>`_ and `BLISlab Tutorial <https://github.com/flame/blislab/blob/master/tutorial.pdf>`_
* `Beating NumPy in 150 lines of C Code <https://salykova.github.io/matmul-cpu>`_ plus the `repo <https://github.com/salykova/matmul.c>`_
