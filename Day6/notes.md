## Coalesced memory access

As we saw before, B is loaded into shared memory in a non-coalesced way, which limits our memory bandwith.

```
Thread 0 → B[0][0]
Thread 1 → B[1][0]  // Different row, same column
Thread 2 → B[2][0]
...
```
In theory, we could transpose the matrix on the CPU :

```
Original B       Transposed B'
[0 1 2]         [0 3 6]
[3 4 5]  →      [1 4 7]
[6 7 8]         [2 5 8]
```

Which would lead to :

```
   Thread 0 → B'[0][0]
   Thread 1 → B'[0][1]  // Same row, consecutive columns
   Thread 2 → B'[0][2]
```

But transposing on the CPU is slow, since our code runs in mear milliseconds, that would imply a bit too much overhead. 

Therefore we have to transpose whil loading on the GPU, which involves a different index mapping. 

For this part, I recommand taking a piece of paper and checking with a dummy matrix multiplication with A/B of different sizes. 

After transposing and with tiling, you should come to the conclusion that the loading row/loading column look like this : 

```cpp
int load_row = tile_idx * TILE_SIZE + threadIdx.y;  // y-dim for rows
int load_col = blockIdx.x * TILE_SIZE + threadIdx.x; // x-dim for cols

if (load_row < K && load_col < N) {
    B_tile[threadIdx.y][threadIdx.x] = B[load_row * N + load_col];
} else {
    B_tile[threadIdx.y][threadIdx.x] = 0.0f;
}
```

rest of the code is unchanged. 

## Results 

We see an improvement of memory bandwith, as expected:

```

Device: Tesla T4
Compute Capability: 7.5
Max threads per block: 1024
Max threads in X-dimension: 1024

=== CPU Implementation ===
Time: 8099.220 ms
Performance: 0.27 GFLOPs

=== Naive GPU Implementation Performance ===
Kernel Time: 9.316 ms
Memory Time: 3.199 ms
Total Time: 12.515 ms
Performance: 230.52 GFLOPs
Memory Bandwidth: 1.01 GB/s
Max difference: 0.000092 at index 166082
Results: PASSED

=== Tiled Implementation Performance ===
Kernel Time: 6.089 ms
Memory Time: 3.199 ms
Total Time: 9.287 ms
Performance: 352.69 GFLOPs
Memory Bandwidth: 1.35 GB/s
Max difference: 0.000092 at index 166082
Results: PASSED

=== Tiled Coalesced Implementation Performance ===
Kernel Time: 6.096 ms
Memory Time: 0.909 ms
Total Time: 7.005 ms
Performance: 352.29 GFLOPs
Memory Bandwidth: 1.80 GB/s
Max difference: 0.000092 at index 166082
Results: PASSED

=== Performance Comparison Report ===
====================================================
Implementation         Time(ms)   GFLOP/s   Speedup
----------------------------------------------------
CPU                    8099.22      0.27    0.002x
Naive GPU (Baseline)     12.52    230.52     1.00x
Tiled                     9.29    352.69     1.35x
Tiled Coalesced           7.00    352.29     1.79x
====================================================

```