# Matrix Multiplication on GPUs

This repository demonstrates matrix multiplication on AMD GPUs using Python/PyTorch and C. The focus is on implementing matrix multiplication in C with the AMD rocBLAS library to achieve similar TFLOPS performance as Python/PyTorch. While both implementations use rocBLAS, this code provides an educational example of how to implement it in C.

## Getting Started

### Prerequisites

* Python 3.x
* AMD ROCm 6.x
* Pytorch 2.x+rocm6.x (Pytorch installed with ROCm 6.x support, can be a container or installed via `pip`)

### Installation

Installation of AMD ROCm 6.x and Pytorch with ROCm support are out of scope for this repo, see [AMD documentation](https://github.com/ROCm/ROCm) for instructions.  The only dependencies needed are PyTorch for Python, making requirements.txt unnecessary, and the C libraries installed by ROCm 6.x.

## Usage

### Running the Pytorch Script

Run the python script using `python pytorch_matmul.py`

### Running the C code

1.  Change directories `cd c`
2.  Compile and run the benchmark `make && ./gpu_matmul`

### Performance Output

The PyTorch implementation uses `torch.matmul`, which abstracts away the complexities of GPU interaction, making the code concise and straightforward. When running this version, users can expect it to set up the input tensors and perform matrix multiplication efficiently using the GPU, with minimal manual intervention. The output will provide a summary of the computation time and performance achieved across multiple runs.

The C implementation uses the AMD rocBLAS library directly, requiring more hands-on setup, including GPU context initialization and data transfers. Since we are implementing the rocBLAS framework ourselves, it also includes accuracy checks to ensure the results are correct. This version outputs GPU specifications, memory transfer times, matrix multiplication times, and the performance for each run. Users will also see results of spot checks to confirm the numerical correctness of the GPU computations compared to CPU expectations, which is important to ensure that high performance isn't accompanied by incorrect results.

## Project Structure

```
matrix-multiplication/
├── README.md
├── c/
│   ├── Makefile
│   ├── include/
│   │   ├── timer.h
│   │   ├── matrix_operations.h
│   │   ├── spot_check.h
│   │   └── utils.h
│   ├── src/
│   │   ├── main.c
│   │   ├── timer.c
│   │   ├── matrix_operations.c
│   │   ├── spot_check.c
│   │   └── utils.c
│   └── obj/
│       └── (object files will be placed here)
└── pytorch/
    └── pytorch_matmul.py
```
