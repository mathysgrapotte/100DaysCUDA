

## What the current program does (Unweighted, Non‑transformed Mode)

See [propr](https://github.com/tpq/propr/blob/master/src/lrv.cpp) for the current implementation.

In the non‐weighted branch—i.e. when the parameter `a` is NA and `weighted` is false—the program processes the input matrix **Y** (where rows are samples and columns are features, for example gene expression values) as follows:

1. **For every pair of features (columns) (i, j) with i > j:**
   - It extracts the full vector of values for each feature (i.e. for column i and column j).
   - It computes the element‑wise log‑ratio:
     
     $v = \log\left(\frac{Y_{\ast, i}}{Y_{\ast, j}}\right)$
     
     where $Y_{\ast, i}$ denotes all rows (samples) in column i.
     
2. **Computes the sample variance of the log‑ratio vector $v$:**  
   This is done using the standard variance formula:
   
   $\text{var}(v) = \frac{1}{N - 1} \sum_{k=1}^{N} \left( v_k - \overline{v} \right)^2,$
   
   where $\overline{v}$ is the mean of the log‑ratio values, and $N$ is the number of samples (the number of rows in Y).

3. **Stores the result:**  
   For each unique pair (i, j), the computed variance is stored in a one‑dimensional output vector. (Since there are $\frac{p(p-1)}{2}$ pairs for p features, the result is a half‑matrix flattened into a vector.)

In the context of the paper, this variance of the log-ratios (often termed "log-ratio variance" or lrv) is used as a normalization‐free measure to assess differential proportionality between gene expression values.

---

## Plan to Implement the Unweighted Mode in CUDA

Because the variance computations for each pair are independent, the task is highly parallelizable. Below is a step-by-step plan to design a CUDA version:

### 1. Data Organization and Transfer
- **Host Data:**  
  Your input matrix **Y** (dimensions $N \times p$) resides in host (CPU) memory.
- **Device Memory:**  
  Allocate device (GPU) memory for:
  - The matrix **Y** (transfer it from host to device).
  - An output array (of length $\frac{p(p-1)}{2}$) to hold the computed variances.

### 2. Parallelization Strategy
- **Mapping Pairs to Threads/Blocks:**  
  Each unique pair (i, j) can be computed independently.  
  We will use a 1d grid of threads since we are interested in pairs of vectors (1D elements).
    
i.e. 

```cpp
// Grid dimensions: one block per pair
int num_pairs = (nb_genes * (nb_genes - 1)) / 2;
int threads_per_block = 256; // tune this
int num_blocks = num_pairs;

computeLogRatioVariance<<<num_blocks, threads_per_block, shared_mem_size>>>(
    d_Y,           // input matrix
    d_variances,   // output array
    nb_samples,    // N
    nb_genes       // p
);
```

### 3. Kernel Design
For each pair (i, j), you need to:
- **Compute Log-Ratios:**  
  For each sample $k$ (from 0 to $N-1$), compute  
  $v_k = \log\left(\frac{Y[k][i]}{Y[k][j]}\right).$
- **Reduction to Compute the Mean:**  
  Use parallel reduction (e.g., using shared memory) within the thread block to compute the sum of $v_k$ and then the mean $\overline{v}$.
- **Reduction to Compute the Variance:**  
  In a second reduction pass (or combined with the first), compute the sum of squared differences:  
  $\sum_{k=1}^{N} (v_k - \overline{v})^2.$
  Then divide by $N-1$ to obtain the variance.
- **Store the Result:**  
  Write the computed variance for pair (i, j) to the appropriate location in the output array.




4. **Final Reduction:**  
   Perform a final reduction (within each block) to compute the complete sum and sum of squares.
5. **Store and Copy Back:**  
   Write the result for each pair to the output array in global memory, and then copy the entire output vector back to the host.

### 5. Testing and Validation
- **Validation:**  
  Compare the output from your CUDA implementation with the Rcpp version (using a small test matrix) to ensure that variances match.
- **Performance Tuning:**  
  Optimize memory accesses, use shared memory effectively, and consider using asynchronous kernel launches and streams if the number of pairs is very large. ((future work))

---
## optimisation tricks 

- sum reduction 

The idea of sum reduction is to have multiple threads compute sums at the same time and use sync inbetween to make sure we are synced.

Here is the implementation details (loop is unrolled)

```cpp
// Initial state of mean array with 8 elements:
// mean = [5, 2, 7, 1, 8, 3, 6, 4]
//         0  1  2  3  4  5  6  7  (indices)

// Reduction loop unrolled:
__syncthreads();
if (threadIdx.x < 4) {  // stride = 4
    mean[threadIdx.x] += mean[threadIdx.x + 4];
}
// After first iteration:
// mean = [13, 5, 13, 5, 8, 3, 6, 4]
//         0   1   2  3  4  5  6  7
// Thread 0 added indices 0+4
// Thread 1 added indices 1+5
// Thread 2 added indices 2+6
// Thread 3 added indices 3+7

__syncthreads();
if (threadIdx.x < 2) {  // stride = 2
    mean[threadIdx.x] += mean[threadIdx.x + 2];
}
// After second iteration:
// mean = [26, 10, 13, 5, 8, 3, 6, 4]
//         0   1   2  3  4  5  6  7
// Thread 0 added indices 0+2
// Thread 1 added indices 1+3

__syncthreads();
if (threadIdx.x < 1) {  // stride = 1
    mean[threadIdx.x] += mean[threadIdx.x + 1];
}
// Final result:
// mean = [36, 10, 13, 5, 8, 3, 6, 4]
//         0   1   2  3  4  5  6  7
// Thread 0 added indices 0+1
```

- shared memory

When two elements are not in the same warp, we can load them into shared memory to decrease the number of global memory access.
When we then transfer the data, each thread will load a datapoint and move it to shared memory. 

```cpp
__shared__ float shared_mem[32];

shared_mem[threadIdx.x] = Y[threadIdx.x];
```	

TODO: understand what are the limits of shared memory, and how to use it efficiently.



## misc : 

Current R implementation is compared to CPU implementation in [propr_compare.cpp](propr_compare.cpp) (run test.R to see the results).

## Results

```
Device: NVIDIA RTX A4500
Compute Capability: 8.6
Max threads per block: 1024
Max threads in X-dimension: 1024

Performance Metrics:
Matrix Size: 80x10000
  +-- Kernel Time:     447.00 ms
  +-- Memory Time:     0.71 ms
Total Time: 447.715 ms
Performance: 35.79 GFLOPs
Memory Bandwidth: 0.02 GB/s
Results: PASSED

=== Log Variance Ratio Benchmark Report ===
============================================
Performance Summary:
--------------------------------------------
Total Execution Time: 447.71 ms
  +-- Kernel Time:     447.00 ms
  +-- Memory Time:     0.71 ms

Compute Performance:
--------------------------------------------
GFLOP/s:             35.79
Memory Bandwidth:     0.02 GB/s
============================================

=== CPU vs GPU Comparison ===
--------------------------------------------
CPU Time:             95280.00 ms
GPU Time:             447.71 ms
Speedup:              212.81x

Compute Performance:
CPU GFLOP/s:          0.17
GPU GFLOP/s:          35.79
Performance Ratio:     213.15x
============================================
```