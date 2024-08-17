import torch
from torch.utils.benchmark import Timer
import argparse

def main(gpu_id):
    N = 16384
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # Clear the CUDA cache before running the benchmark
    torch.cuda.empty_cache()
    
    A = torch.empty(N, N, dtype=torch.float32, device=device).uniform_(-1,1)
    B = torch.empty(N, N, dtype=torch.float32, device=device).uniform_(-1,1)

    timer = Timer(stmt='torch.matmul(A, B); torch.cuda.synchronize()',
                  globals={'A': A, 'B': B, 'torch': torch})

    result = timer.timeit(number=10)
    flops = 2 * N**3
    gflops = (flops / result.mean) / 1e9

    print(f"Matrix size: {N}x{N}")
    print(f"Mean execution time: {result.mean:.6f} seconds")
    print(f"Performance: {gflops:.2f} GFLOPS")
    print(f"GPU used: {gpu_id} - {torch.cuda.get_device_name(gpu_id)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Matrix Multiplication Benchmark")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use for the benchmark")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        exit(1)
    elif args.gpu >= torch.cuda.device_count():
        print(f"GPU {args.gpu} is not available. Available GPUs: 0 to {torch.cuda.device_count()-1}")
        exit(1)
    else:
        print(f"Using GPU: {args.gpu}")

    main(args.gpu)
