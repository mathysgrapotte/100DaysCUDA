## Coal matmul optimisation

The kernel uses a 1D thread index (threadIdx.x) and converts it to a 2D coordinate in a 32x32 tile (just like the previous implementation).
We do this using bitwise operations because I thought modulo / integer division was causing overhead but it only yields a .01% improvement in compute efficiency.

Code Snippet:
```cpp
   const int tid = threadIdx.x;
   const int localRow = tid >> 5;              // Equivalent to: tid / 32
   const int localCol = tid & (BLOCKSIZE - 1);   // Equivalent to: tid % 32
   const int globalRow = blockIdx.y * BLOCKSIZE + localRow;
   const int globalCol = blockIdx.x * BLOCKSIZE + localCol;
```

Each block handles one tile of output and maps 1024 threads (32x32) to the tile's rows and columns.

```
   Thread indices in a block (flattened 1D order):

           [ 0] [ 1] [ 2] ... [31]    --> Maps to row 0 (localRow = 0)
           [32] [33] [34] ... [63]    --> Maps to row 1 (localRow = 1)
           ...       ...    ... 
           [992] ...          [1023]   --> Maps to row 31 (localRow = 31)
```

The kernel divides the task along the K-dimension into tiles. For each tile, it loads submatrices
from A and B into shared memory. Shared memory is much faster than global memory, so it is useful if we need to re-use the data (which is our case, since we have an inner for loop).

```cpp
   // Allocate shared memory for tiles of A and B
   __shared__ float sharedA[BLOCKSIZE][BLOCKSIZE];
   __shared__ float sharedB[BLOCKSIZE][BLOCKSIZE];

   // Loop over tiles in the K dimension
   for (int tile = 0; tile < (K + BLOCKSIZE - 1) / BLOCKSIZE; tile++) {
       int tiledA_col = tile * BLOCKSIZE + localCol;
       int tiledB_row = tile * BLOCKSIZE + localRow;
       
       // Load A tile with bounds checking and zero-padding
       if (globalRow < M && tiledA_col < K) {
           sharedA[localRow][localCol] = A[globalRow * K + tiledA_col];
       } else {
           sharedA[localRow][localCol] = 0.0f;
       }
       
       // Load B tile with bounds checking and zero-padding
       if (tiledB_row < K && globalCol < N) {
           sharedB[localRow][localCol] = B[tiledB_row * N + globalCol];
       } else {
           sharedB[localRow][localCol] = 0.0f;
       }
       
       __syncthreads();  // Ensure the entire tile is loaded

       // Compute partial products for the current tile
       for (int k = 0; k < BLOCKSIZE; k++) {
           sum += sharedA[localRow][k] * sharedB[k][localCol];
       }
       
       __syncthreads();  // Prepare for loading the next tile
   }
```


Recap :
- The matrix multiplication is broken into tiles (submatrices) of dimension 32x32.
- Each tile from matrices A and B is loaded into shared memory (sharedA and sharedB) by the block.
- Each thread is responsible for loading one element of the tile, ensuring that global memory accesses are
  coalesced (i.e., consecutive threads access consecutive memory locations).
- __syncthreads() is called before computation to ensure the complete tile is loaded and again after
  the computation to prepare for the next tile.
  
```
   Global Matrix (either A or B)
   +---------------------------+
   |       Tile 0 (32x32)      |
   +---------------------------+
   |       Tile 1 (32x32)      |
   +---------------------------+
   |           ...             |
   +---------------------------+
```

In cases where the matrix dimensions are not an exact multiple of the tile size, the kernel incorporates bounds
checking. If a thread attempts to access an element beyond the matrix bounds, it writes a zero into shared memory.
This ensures that the extra computations do not affect the final result.

```
   if (globalRow < M && tiledA_col < K) {
       sharedA[localRow][localCol] = A[globalRow * K + tiledA_col];
   } else {
       sharedA[localRow][localCol] = 0.0f;
   }
```

After processing all tiles, each thread has computed a partial sum for its designated output element.
A final bounds check ensures that the thread's computed row and column indices fall within the valid
output range before writing the result back to global memory.

   if (globalRow < M && globalCol < N) {
       C[globalRow * N + globalCol] = sum;
   }




