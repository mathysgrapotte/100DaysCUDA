## Global memory access

In a typical matrix multiplication, computing an element of the result involves accessing entire rows of matrix A and entire columns of matrix B. Without tiling, every thread might load the same data repeatedly from global memory—even though global memory access is both relatively high latency (hundreds of cycles) and lower bandwidth compared to on-chip memory (shared memory or registers).
Consider a simple global memory layout for a matrix in row-major order:

```
Global Memory for Matrix A (Row-major):
+------------------------+
|  a00  a01  a02  a03 ...|
|  a10  a11  a12  a13 ...|
|  a20  a21  a22  a23 ...|
|  a30  a31  a32  a33 ...|
+------------------------+
```

Here, if every thread were to access each needed element separately from global memory, then many of the same elements would be fetched multiple times. This redundancy wastes precious bandwidth. 

## Tiling 

Tiling is the technique of dividing matrices into small sub-blocks (or tiles) that can fit into shared memory. Each thread block will cooperatively load these sub-blocks into the faster, on-chip shared memory—where data can be reused many times. This approach greatly reduces access to the slower global memory.
Let’s imagine we are multiplying two matrices A(MxK) and B(KxN) to produce C(MxN). With tiling, the operation is split into smaller tiles. 

```
Global Matrix A:
+----------------------------------------+
| a00  a01  a02  a03 | a04  a05  ...      |
| a10  a11  a12  a13 | a14  a15           |
| a20  a21  a22  a23 | a24  a25           |
| a30  a31  a32  a33 | a34  a35           |
|  ...                ...                |
+----------------------------------------+

Extracting a tile (e.g., a 4x4 block):
+-----------------+
| a00  a01  a02  a03 |
| a10  a11  a12  a13 |
| a20  a21  a22  a23 |
| a30  a31  a32  a33 |
+-----------------+
```

Once this tile is loaded into shared memory, all threads within that block can access it quickly and perform multiple computations without reloading the same data repeatedly.

## GPU architecture, or why is this fast

The GPU is designed for massive parallelism, with thousands of threads running concurrently. However, these threads have:
- Fast-access, low-capacity memory: Shared memory (on-chip)
- Slow-access, high-capacity memory: Global memory

Using tiling, we take advantage of the fast shared memory by loading the tile once and reusing it for many arithmetic operations.

```
          Global Memory (High latency, high capacity)
                   │
                   ▼  <-- Load tile (cooperative, once-per-tile)
            +------------------+
            |  Shared Memory   |  <-- Fast access for all threads in the block
            | (Tile of A & B)  |
            +------------------+
                   │
                   ▼  <-- Each thread works using the data loaded in shared memory
            +------------------+
            |  ALUs (Compute)  |
            +------------------+
```

Importantly, memory bandwiths limits our transfers between Global Memory and Shared Memory.
Because the shared memory tile is used many times for computing the output tile, the total number of global memory accesses is greatly reduced. In essence, tiling increases arithmetic intensity—the ratio of computations to memory accesses—making the algorithm more efficient on architectures that are memory-bandwidth bound.

```
Thread Block Layout:
+----+----+----+----+
| t0 | t1 | t2 | t3 |
+----+----+----+----+
| t4 | t5 | t6 | t7 |
+----+----+----+----+
| t8 | t9 |t10 |t11 |
+----+----+----+----+
|t12 |t13 |t14 |t15 |
+----+----+----+----+
```

Each thread in this block loads a specific element from the global matrix into the shared memory tile (after proper synchronization).

## Results 

```
Device: Tesla T4
Compute Capability: 7.5
Max threads per block: 1024
Max threads in X-dimension: 1024

=== CPU Implementation ===
Time: 7241.844 ms

=== Naive Implementation Performance ===
Kernel Time: 9.253 ms
Memory Time: 3.072 ms
Total Time: 12.325 ms
Performance: 232.09 GFLOPs
Memory Bandwidth: 1.02 GB/s
Max difference: 0.000092 at index 166082
Results: PASSED

=== Tiled Implementation Performance ===
Kernel Time: 6.087 ms
Memory Time: 3.076 ms
Total Time: 9.163 ms
Performance: 352.80 GFLOPs
Memory Bandwidth: 1.37 GB/s
Max difference: 0.000092 at index 166082
Results: PASSED

=== Performance Comparison Report ===
============================================
Implementation      Time(ms)   GFLOP/s   Speedup
--------------------------------------------
Naive (Baseline)      9.25    232.09     1.00x
Tiling                6.09    352.80     1.52x
============================================
```