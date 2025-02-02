Below is an explanation of what the unweighted (non‐alpha-transformed) branch of the C++ code does and a high-level plan for implementing that functionality in CUDA.

---

## What the Program Does (Unweighted, Non‑transformed Mode)

In the non‐weighted branch—i.e. when the parameter `a` is NA and `weighted` is false—the program processes the input matrix **Y** (where rows are samples and columns are features, for example gene expression values) as follows:

1. **For every pair of features (columns) (i, j) with i > j:**
   - It extracts the full vector of values for each feature (i.e. for column i and column j).
   - It computes the element‑wise log‑ratio:
     
     \[
     v = \log\left(\frac{Y_{\ast, i}}{Y_{\ast, j}}\right)
     \]
     
     where \( Y_{\ast, i} \) denotes all rows (samples) in column i.
     
2. **Computes the sample variance of the log‑ratio vector \(v\):**  
   This is done using the standard variance formula:
   
   \[
   \text{var}(v) = \frac{1}{N - 1} \sum_{k=1}^{N} \left( v_k - \overline{v} \right)^2,
   \]
   
   where \( \overline{v} \) is the mean of the log‑ratio values, and \(N\) is the number of samples (the number of rows in Y).

3. **Stores the result:**  
   For each unique pair (i, j), the computed variance is stored in a one‑dimensional output vector. (Since there are \(\frac{p(p-1)}{2}\) pairs for p features, the result is a half‑matrix flattened into a vector.)

In the context of the paper, this variance of the log-ratios (often termed “log-ratio variance” or lrv) is used as a normalization‐free measure to assess differential proportionality between gene expression values.

---

## Plan to Implement the Unweighted Mode in CUDA

Because the variance computations for each pair are independent, the task is highly parallelizable. Below is a step-by-step plan to design a CUDA version:

### 1. Data Organization and Transfer
- **Host Data:**  
  Your input matrix **Y** (dimensions \(N \times p\)) resides in host (CPU) memory.
- **Device Memory:**  
  Allocate device (GPU) memory for:
  - The matrix **Y** (transfer it from host to device).
  - An output array (of length \(\frac{p(p-1)}{2}\)) to hold the computed variances.

### 2. Parallelization Strategy
- **Mapping Pairs to Threads/Blocks:**  
  Each unique pair (i, j) can be computed independently.  
  There are two common approaches:
  - **2D Grid Approach:**  
    Launch a 2D grid of threads (with dimensions p × p) and let each thread check if its indices satisfy i > j.  
  - **One Block per Pair:**  
    Enumerate the \(\frac{p(p-1)}{2}\) pairs and assign one thread block to each pair.  
    Within each block, use multiple threads to perform a parallel reduction over the N samples.
    
  The second approach is usually preferred because each pair’s variance calculation involves a reduction (summing values and summing squares).

### 3. Kernel Design
For each pair (i, j), you need to:
- **Compute Log-Ratios:**  
  For each sample \( k \) (from 0 to \( N-1 \)), compute  
  \[
  v_k = \log\left(\frac{Y[k][i]}{Y[k][j]}\right).
  \]
- **Reduction to Compute the Mean:**  
  Use parallel reduction (e.g., using shared memory) within the thread block to compute the sum of \(v_k\) and then the mean \( \overline{v} \).
- **Reduction to Compute the Variance:**  
  In a second reduction pass (or combined with the first), compute the sum of squared differences:  
  \[
  \sum_{k=1}^{N} (v_k - \overline{v})^2.
  \]
  Then divide by \(N-1\) to obtain the variance.
- **Store the Result:**  
  Write the computed variance for pair (i, j) to the appropriate location in the output array.

_Note:_ You may use CUDA libraries such as Thrust or CUB to simplify parallel reductions.

### 4. Implementation Steps
1. **Memory Allocation:**  
   Allocate device memory for matrix Y and the output vector.
2. **Data Transfer:**  
   Copy the host matrix Y to device memory.
3. **Kernel Launch:**  
   Launch a kernel where each block is responsible for one pair (i, j).  
   - Within the kernel, each thread processes a subset of the N samples.
   - Use shared memory to accumulate partial sums for the mean and variance.
4. **Final Reduction:**  
   Perform a final reduction (within each block) to compute the complete sum and sum of squares.
5. **Store and Copy Back:**  
   Write the result for each pair to the output array in global memory, and then copy the entire output vector back to the host.
6. **Post-Processing:**  
   Optionally, post-process or convert the result vector to a half‑matrix format if needed by your application.

### 5. Testing and Validation
- **Validation:**  
  Compare the output from your CUDA implementation with the Rcpp version (using a small test matrix) to ensure that variances match.
- **Performance Tuning:**  
  Optimize memory accesses, use shared memory effectively, and consider using asynchronous kernel launches and streams if the number of pairs is very large.

---

## Summary

- **Description:**  
  In the unweighted, non-transformed mode, the program computes the variance of the log ratios of every pair of columns (features) in the input matrix Y.
- **CUDA Implementation Plan:**  
  Transfer Y to the GPU, then for each pair (i, j) use a CUDA kernel (with block-level parallel reductions) to compute the variance of \(\log(Y_{\ast, i} / Y_{\ast, j})\) across all samples, and finally copy the result back to the host.

This high-level plan leverages the inherent parallelism of the problem (each pair can be computed independently) and uses efficient reduction techniques on the GPU to handle the per-pair computations.

If you need further details or code snippets for any of these steps, please let me know!