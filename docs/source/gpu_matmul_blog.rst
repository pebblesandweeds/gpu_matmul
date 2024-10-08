Accelerating Matrix Multiplication on AMD GPUs with rocBLAS in C
================================================================

.. admonition:: Highlights 

 Matrix multiplication is the core operation behind deep learning, driving the computations in neural networks for model training and inference. This blog post demonstrates how AMD's rocBLAS library can be used in C to achieve matrix multiplication performance comparable to PyTorch's implementation, leveraging low-level control for efficient use of AMD GPUs.

 - **Problem Scale**: We perform multiplication of two 16,384 x 16,384 matrices, requiring ~3.21 GB of memory and ~8.8 TFLOPs of computation.

 - **PyTorch Baseline**: Achieves **~37.5 TFLOPS** using `this simple code <https://github.com/pebblesandweeds/gpu_matmul/blob/main/pytorch/pytorch_matmul.py>`_. PyTorch's high-level API (``torch.matmul``) abstracts the underlying rocBLAS operations, providing ease of use without sacrificing performance.

 - **rocBLAS Implementation in C**: Matches PyTorch at **~37.5 TFLOPS** with `this C implementation <https://github.com/pebblesandweeds/gpu_matmul/blob/main/c/src/matrix_operations.c>`_. By directly calling ``rocblas_sgemm()``, we expose GPU programming concepts like memory allocation, data transfer, and operation parameters which provide insight into the underlying processes that high-level APIs abstract away.

 - **Performance Gain**: Our GPU implementation achieves a 12.5x speedup over our `optimized CPU version <https://github.com/pebblesandweeds/cpu_matmul/blob/main/c/src/matmul_lib.c>`_ (3 TFLOPS to 37.5 GFLOPS).

 - **Accuracy Verification**: The C implementation includes spot-checking to verify GPU computation accuracy against CPU results.

 This comparison showcases how low-level C programming with rocBLAS can achieve performance parity with high-level frameworks like PyTorch. The C implementation offers a valuable learning opportunity, introducing developers to GPU programming concepts while maintaining high performance. It serves as a bridge between high-level APIs and custom GPU kernel development, providing a deeper understanding of GPU computing without sacrificing efficiency.

 All benchmarks were run on an AMD Instinct MI250X GPU, demonstrating the capabilities of AMD's high-performance hardware for deep learning.

 Get all of the code `in this repo <https://github.com/pebblesandweeds/gpu_matmul>`_.

Introduction
------------

Matrix multiplication is a cornerstone operation in machine learning and deep learning, powering critical computations in neural networks such as forward and backward propagation. In `our previous blog post <https://blog.pebblesandweeds.com/cpu_matmul_blog.html#why-is-matrix-multiplication-important>`_, we explored this fundamental operation by implementing matrix multiplication from scratch in C on `AMD EPYC CPUs <https://aws.amazon.com/ec2/instance-types/c7a/>`_. This exploration laid the groundwork for understanding the core principles behind matrix multiplication, setting the stage for our journey into GPU accelerated computations. 

Building on that foundation, this blog post extends our exploration to GPU acceleration. We demonstrate how to harness the power of AMD GPUs for high-performance matrix multiplication using the `rocBLAS library <https://github.com/rocm/rocBLAS>`_ in C. While rocBLAS is not as low-level as building custom GPU kernels from scratch, it provides a middle ground, offering more granular control than high-level libraries like PyTorch. Our goal is to showcase how this C implementation with rocBLAS can achieve performance parity with high-level libraries, while offering developers greater insight into GPU resource management and a deeper understanding of GPU programming concepts without the complexity of writing kernels from scratch.

Matrix Multiplication: CPUs vs. GPUs
------------------------------------

Implementing matrix multiplication differs significantly between CPUs and GPUs. Understanding these differences sheds light on why GPU acceleration is crucial for deep learning computations.

- **CPUs** are optimized for general-purpose, sequential tasks. They excel at handling smaller workloads, complex operations, and situations requiring low latency for individual operations or when working with sparse matrices. Efficient CPU matrix multiplication in C focuses on cache utilization and instruction-level parallelism. While CPUs can leverage multiple cores through threading, their parallelism is limited by core count, making them less ideal for very large matrix operations.

- **GPUs**, by contrast, are designed for massively parallel computation, making them well-suited for the dense arithmetic operations required by matrix multiplication. GPUs contain thousands of lightweight cores that can perform many matrix operations simultaneously, leading to substantial performance gains for large workloads.

Modern GPUs offer different computation units, such as SIMD (Single Instruction, Multiple Data) processors and dedicated matrix multiplication units (often called Tensor Cores in NVIDIA GPUs or Matrix Cores in AMD GPUs). While SIMD units are versatile, the specialized matrix multiplication units offer significantly higher performance for operations like matrix multiplication. Writing efficient GPU code typically involves leveraging specialized libraries that are optimized to fully exploit these architectural features. Libraries such as AMD's rocBLAS for AMD GPUs or cuBLAS for NVIDIA GPUs are designed to utilize these specialized units, providing implementations that are far more efficient than what most developers could achieve with general-purpose GPU code.

In this blog, we move from CPU-based matrix multiplication to implementing it on GPUs using AMD's rocBLAS library in C. GPUs handle data differently, relying on parallel execution and optimized memory transfers to achieve high throughput. Understanding these differences is crucial when writing C code that takes advantage of GPU acceleration, allowing us to fully harness the capabilities of GPUs for deep learning tasks.

AMD GPU Programming in C
------------------------

**Why use C instead of PyTorch?**

Using PyTorch offers a high-level, user-friendly interface to perform matrix multiplication on GPUs, abstracting away most of the complexity. However, by writing matrix multiplication in C, we gain direct, low-level control over the GPU and understand the internal workings behind the scenes. This approach is crucial for learning how to optimize and fully exploit the hardware for maximum performance. It also offers an educational perspective, helping us understand what PyTorch does automatically, and provides insights for those who want to go beyond existing frameworks.

**Why rocBLAS?**

rocBLAS is a high-level library provided by AMD that offers efficient GPU implementations of BLAS operations, including matrix multiplication. This is an ideal starting point for programming GPUs with C, as it abstracts many of the complexities of directly writing GPU kernels while still providing a hands-on experience with GPU programming. Writing custom GPU kernels is a complex task and out of scope for this blog. Starting with rocBLAS allows us to learn the fundamentals of GPU programming and gain performance improvements without diving into the intricacies of kernel development right away.

**Why AMD?**

Because AMD is awesome! While there are an abundance of CUDA (NVIDIA) resources available online, there are fewer guides for programming on AMD GPUs, and we wanted to fill that gap. AMD’s ROCm platform provides a powerful environment for GPU programming, and this blog aims to showcase how to effectively use it. Plus, working with AMD GPUs provides a broader perspective for GPU programming, going beyond the NVIDIA-centric focus that is currently common in the industry.

GPU Matrix Multiplication with rocBLAS
--------------------------------------

Writing efficient GPU kernels can be challenging, as it requires careful handling of memory access patterns, synchronization, and the coordination of thousands of parallel threads to exploit modern GPU architectures. For tasks like matrix multiplication, starting with optimized libraries such as rocBLAS is beneficial, as it provides high-level APIs that abstract away much of the complexity, enabling developers to focus on leveraging GPU acceleration without diving into the intricacies of kernel development.

rocBLAS offers a set of optimized linear algebra routines specifically designed for AMD GPUs, making it an ideal choice for efficient matrix multiplication. By using rocBLAS, developers can achieve high performance without manually managing low-level GPU features, which can be time-consuming and error-prone. This guide will walk through how to use rocBLAS for implementing matrix multiplication in C, highlighting how to achieve efficient results by utilizing this powerful library.

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

*Putting It All Together: From Formulas to Implementation*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our project demonstrates two approaches to implementing GPU-accelerated matrix multiplication: a high-level implementation using PyTorch and a lower-level implementation in C using rocBLAS directly. The PyTorch implementation abstracts away the complexities of GPU programming and the rocBLAS API. When we perform matrix multiplication using PyTorch's ``torch.matmul`` function, we're indirectly utilizing the rocBLAS library on AMD GPUs. PyTorch's backend automatically handles the intricate details of memory allocation, data transfer between CPU and GPU, and the construction of the appropriate rocBLAS function calls. This abstraction allows developers to focus on the higher-level aspects of their algorithms without worrying about the underlying GPU operations. However, this convenience comes at the cost of some flexibility and fine-grained control over the exact operations being performed.

In contrast, our C implementation provides a more direct interface to the rocBLAS library, offering greater control but requiring more manual management. In this approach, we explicitly construct the rocBLAS API calls, handling details such as creating rocBLAS handles, specifying matrix operations (like transposition), and managing GPU memory directly. This lower-level implementation allows us to fine-tune parameters and potentially optimize performance for specific use cases. It also provides a clearer view of how the GEMM formula translates into actual GPU operations. While this method requires more code and a deeper understanding of GPU programming and the rocBLAS API, it offers the potential for highly optimized, application-specific implementations of matrix multiplication.

Both approaches ultimately leverage the power of rocBLAS and the underlying GEMM formula to perform efficient matrix multiplications on the GPU. The choice between them depends on the specific needs of the project, balancing factors such as development time, required performance optimizations, and the level of control needed over the GPU operations. Whether using the high-level abstractions provided by PyTorch or the direct control offered by our C implementation, the end goal remains the same: harnessing the computational power of GPUs to perform fast, efficient matrix multiplications using the optimized algorithms provided by rocBLAS.

Benchmarking Setup and Code Organization
----------------------------------------

*Benchmarking Setup and Matrix Configuration*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our implementation performs matrix multiplication using the formula C = A x B, where A and B are square matrices of size N × N. We've set N to 16,384, which provides a substantial workload to showcase GPU performance. This configuration is defined using a preprocessor C macro (``#define N 16384``), allowing for compiler optimizations and consistent runtime behavior.

Memory Requirements
'''''''''''''''''''

With N = 16,384, each matrix contains 268,435,456 elements. Using 32-bit floating-point precision (FP32), the size of each matrix is:

 .. math::

       268,435,456 \times 4 \text{ bytes} = 1,073,741,824 \text{ bytes} \approx 1.07 \text{ GB}

This results in a total memory requirement of approximately 3.21 GB for all three matrices.

Computational Complexity
''''''''''''''''''''''''

The computational effort for matrix multiplication of this size is substantial. The total number of floating point operations (FLOPs) is approximated by:

    .. math::

       \text{Total FLOPs} = 2N^3 = 2 \times 16,384^3 = 8,796,093,022,208 \approx 8.8 \text{ TFLOPs}

This immense number of operations underscores the computational intensity of large-scale matrix multiplication and highlights the importance of GPU acceleration.

Benchmarking Environment
''''''''''''''''''''''''

Our benchmarks are conducted on AWS instances equipped with AMD GPUs. This setup allows us to fully utilize the massive parallel processing capabilities of GPUs, which are particularly well-suited for the highly parallelizable task of matrix multiplication. By using GPUs, we can efficiently handle the large datasets and intensive computations required by our 16,384 × 16,384 matrix multiplication task.

*Code Structure and Organization*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our project is structured to provide both a low-level C implementation using rocBLAS and a high-level PyTorch implementation. The full project structure can be found in the `README.md file <https://github.com/pebblesandweeds/gpu_matmul?tab=readme-ov-file#project-structure>`_.

The C implementation is organized into several key components:

- ``main.c``: Contains the primary program logic and benchmarking code.
- ``matrix_operations.c``: Implements the core matrix multiplication functions using rocBLAS.
- ``utils.c``: Provides utility functions for memory management and data initialization.
- ``timer.c``: Offers functions for precise timing of operations.
- ``spot_check.c``: Includes functions for verifying the correctness of matrix multiplication results.

Header files in the ``include/`` directory declare the interfaces for these components, promoting modularity and ease of use.

The PyTorch implementation is contained in a single file, ``pytorch_matmul.py``, showcasing the simplicity and conciseness of high-level frameworks for GPU computations.

This organization allows for a clear comparison between the low-level, fine-grained control offered by the C implementation and the high-level abstraction provided by PyTorch, both leveraging GPU acceleration for matrix multiplication.

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

This code initializes two 16384x16384 matrices with random values on the GPU.

Matrix Multiplication
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   torch.matmul(A, B)

This single line performs the entire matrix multiplication operation, leveraging PyTorch's optimized backend (which uses rocBLAS for AMD GPUs).

FLOPS Calculation
^^^^^^^^^^^^^^^^^

.. code-block:: python

   flops = 2 * N**3
   tflops = (flops / run_time) / 1e12

We calculate the number of floating-point operations as 2N³, where N is the matrix dimension. This accounts for N³ multiplications and N³ additions. We then convert this to TFLOPS (Tera FLOPS) by dividing by the runtime and 10¹².

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

The stark difference between the first run and subsequent runs clearly demonstrates the overhead of initializing the GPU kernel. After initialization, we see stable performance at about 37.5 TFLOPS, showcasing the impressive computational capabilities of the AMD Instinct MI250X/MI250 GPU for large-scale matrix multiplication tasks.

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
