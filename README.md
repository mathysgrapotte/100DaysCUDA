# Building Deep Learning in CUDA! 🚀

from [Umar Jamil's challenge](https://github.com/hkproj/100-days-of-gpu)

## Goal

Goal is to reproduce some of the results of [The Principles of Deep Learning Theory](https://arxiv.org/abs/2106.10165) using CUDA and benchmark against pytorch on the way. 

## Roadmap

| Roadmap | Status |
|---------|--------|
| Vector Addition in CUDA | 🤗 |
| Matrix Addition in CUDA | 🤗 |
| MatMul Level 0: Naive Implementation | 🤗 |
| MatMul Level 1: Memory Coalescing | 🤗 |
| MatMul Level 2: Shared Memory Tiling | 🤗 |
| MatMul Level 3: Loop Unrolling | 🫠 |
| MatMul Level 4: Double Buffering | 🫠 |
| MatMul Level 5: Advanced Block Tiling | 🫠 |
| MatMul Level 6: Warp-level Optimizations | 🫠 |
| MatMul Level 7: Register Tiling & Vectorization | 🫠 |
| MatMul Level 8: Tensor Cores (Optional) | 🫠 |
| MatMul Level 9: Multi-GPU & CUDA Streams | 🫠 |
| MatMul Level 10: Final Optimized Kernel | 🫠 |
| One Layer MLP Forward Pass | 🫠 |
| One Layer MLP Backward Pass | 🫠 |
| Softmax Activation Function | 🫠 |
| SGD Optimizer in CUDA | 🫠 |
| Train on Titanic Dataset | 🫠 |
| VR1 Network Width -> ∞ = Gaussian distribution | 🫠 |

## Side Quests

### CUDA Propr

Propr is a package for differential expression analysis in R. It is based on the log-variance ratio (LRV) method.
The main backend is written in Rcpp, so quite optimized for CPU.

Unfortunately, this is still mega slow, CUDA comes to the rescue as it is an embarrassingly parallel problem.

See Day4 for the implementation and notes for the algorithm.

| Roadmap | Status |
|---------|--------|
| Implement Log Variance Ratio in CUDA | 🤗 |
| Implement Propr in CUDA | 🤗 |
| benchmark both and make sure we get the same results | 🤗 |



## Progress Log

| Day      | Progress & Notes |
|----------|-----------------|
| Day 1    | Implemented vector addition in CUDA. Learned about CUDA thread hierarchy (grids/blocks/threads), memory management (cudaMalloc/cudaMemcpy), and kernel execution. Created a simple vector addition program that adds two arrays of 1024 elements in parallel using 256 threads per block. Focused on proper memory allocation, transfers between host/device, and error handling. |
| Day 2    | Implemented matrix addition in CUDA. Deep dive into memory coalescing, thread synchronization, and grid organization. Learned about memory layout, coalesced access patterns, 2D grid/block organization, CUDA synchronization, and thread execution model. Created a matrix addition program handling 1024x1024 matrices using 16x16 thread blocks. |
| Day 3    | Started matrix multiplication optimization journey with Level 0 (naive implementation). Created comprehensive performance benchmarking infrastructure for measuring kernel execution, memory transfers, GFLOPS, bandwidth and CPU vs GPU comparisons. Created detailed notes on CUDA block size selection, covering hardware constraints, matrix size considerations, memory access patterns and optimization strategies. Implemented naive matrix multiplication with 16x16 thread blocks for 1024x1024 matrices. |
| Day 4    | Implemented Log Variance Ratio in CUDA. Learned about shared memory and about sum reduction. Yields 200x speedup over Rcpp log ratio function !! |
| Day 5    | Implemented MatMul tiling, took a long time to get right, especially vey messy in my dealing with the indices at first, but I feel like I understand the idea now. Benchmarked against naive implementation. |
| Day 6    | Implemented Coalesced access through transposing input matrix at GPU load time. |
| Day 7    | Recap: rewrote naive matmul from scratch (so no llm or internet, just pen&paper). |
| Day 8    | Recap: implemented profiling toolkit. |
| Day 9    | Recap: re-implemented matmul memory coalescing and added debugging tools to understand that memory was indeed coalesced. |
| Day 10   | Recap: re-implemented shared-memory and tiling.|
| Day 11   |  |
| Day 12   |  |
| Day 13   |  |
| Day 14   |  |
| Day 15   |  |
| Day 16   |  |
| Day 17   |  |
| Day 18   |  |
| Day 19   |  |
| Day 20   |  |
| Day 21   |  |
| Day 22   |  |
| Day 23   |  |
| Day 24   |  |
| Day 25   |  |
| Day 26   |  |
| Day 27   |  |
| Day 28   |  |
| Day 29   |  |
| Day 30   |  |
| Day 31   |  |
| Day 32   |  |
| Day 33   |  |
| Day 34   |  |
| Day 35   |  |
| Day 36   |  |
| Day 37   |  |
| Day 38   |  |
| Day 39   |  |
| Day 40   |  |
| Day 41   |  |
| Day 42   |  |
| Day 43   |  |
| Day 44   |  |
| Day 45   |  |
| Day 46   |  |
| Day 47   |  |
| Day 48   |  |
| Day 49   |  |
| Day 50   |  |
| Day 51   |  |
| Day 52   |  |
| Day 53   |  |
| Day 54   |  |
| Day 55   |  |
| Day 56   |  |
| Day 57   |  |
| Day 58   |  |
| Day 59   |  |
| Day 60   |  |
| Day 61   |  |
| Day 62   |  |
| Day 63   |  |
| Day 64   |  |
| Day 65   |  |
| Day 66   |  |
| Day 67   |  |
| Day 68   |  |
| Day 69   |  |
| Day 70   |  |
| Day 71   |  |
| Day 72   |  |
| Day 73   |  |
| Day 74   |  |
| Day 75   |  |
| Day 76   |  |
| Day 77   |  |
| Day 78   |  |
| Day 79   |  |
| Day 80   |  |
| Day 81   |  |
| Day 82   |  |
| Day 83   |  |
| Day 84   |  |
| Day 85   |  |
| Day 86   |  |
| Day 87   |  |
| Day 88   |  |
| Day 89   |  |
| Day 90   |  |
| Day 91   |  |
| Day 92   |  |
| Day 93   |  |
| Day 94   |  |
| Day 95   |  |
| Day 96   |  |
| Day 97   |  |
| Day 98   |  |
| Day 99   |  |
| Day 100  |  |
