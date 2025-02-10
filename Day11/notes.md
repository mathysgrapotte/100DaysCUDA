### Day 11 understanding profiling & bank conflicts

Day 11 we implemented bank conflict management. 

Shared GPU memory is organised by banks, banks are access channels, meaning that they store addresses to share memory. If we would have a single bank, every thread would need to access that bank. To unable controled parallel acces, GPUs have 32 banks, each of them contains a piece of the address book. 

Banks are organised using a row-major convention. Therefore, threads within a warp calculate which bank they need to access using the following formula `Bank Index = (Byte Address / 4) % SHARED_ACCESS_SIZE[1]`.

Threads cannot access the same bank at the same time (i.e. a "lock" is applied to the bank when the thread accesses it). This means that if thread x is accessing bank 1 and thread y need to access bank 1, thread y needs to wait until thread x completes, which causes performance losses. 

In our code:

```cpp

__shared__ float sharedA[BLOCKSIZE][BLOCKSIZE];  
__shared__ float sharedB[BLOCKSIZE][BLOCKSIZE];

// Compute the partial product for this tile
for (int k = 0; k < BLOCKSIZE; k++) {
    sum += sharedA[localRow][k] * sharedB[k][localCol];
}
```

`sharedA[0][0]` and `sharedA[1][0]` are accessing the same bank:

```
[0][0]   | (0*32 + 0)%32 = 0
[1][0]   | (0*32 + 0)%32 = 0
```

which is why, if we apply define our shared memory to have an extra column, we change the way the access is computed an make sure threads are accessing different banks when the row of A changes: 

```cpp
__shared__ float sharedA[BLOCKSIZE][BLOCKSIZE+1]; //[32][33]
// [0][0]   | (0*33 + 0)%32 = 0
// [1][0]   | (1*33 + 0)%32 = 1
```

however, if we do that, our performance unexpectedly drops, 

```
Kernel matmul_naive: Results Passed!
Kernel matmul_naive execution time: 308.2091 ms
Kernel matmul_coal: Results Passed!
Kernel matmul_coal execution time: 209.0226 ms
Kernel matmul_coal_optim: Results Passed!
Kernel matmul_coal_optim execution time: 137.1723 ms
Kernel matmul_conflicts: Results Passed!
Kernel matmul_conflicts execution time: 175.8396 ms <- kernel with conflicts optimised
```

What is going on ?

To understand, we need to run proviling, I am running this command in colab to do this :
`!ncu --target-processes all ./matmul`

```
| Metric                  | matmul_coal_optim | matmul_conflicts |
| ----------------------- | ----------------- | ---------------- |
| DRAM Frequency          | 5.00 GHz          | 5.00 GHz         |
| SM Frequency            | 585.00 MHz        | 585.00 MHz       |
| Elapsed Cycles          | 195856422 cycles  | 254197615 cycles |
| Memory Throughput       | 75.41%            | 89.78%           |
| DRAM Throughput         | 11.31%            | 8.72%            |
| Duration                | 334.80 ms         | 434.53 ms        |
| L1/TEX Cache Throughput | 89.44%            | 90.03%           |
| L2 Cache Throughput     | 5.88%             | 4.53%            |
| SM Active Cycles        | 195586939.57 cycl | 253892040 cycles |
| Compute (SM) Throughput | 75.41%            | 89.78%           |
| Block Size              | 1024              | 1024             |
| Function Cache Conf.    | CachePreferNone   | CachePreferNone  |
| Grid Size               | 16384             | 16384            |
| Registers/Thread        | 41                | 36               |
| Shared Mem Conf. Size   | 32.77 Kbyte       | 32.77 Kbyte      |
| Driver Shared Memory    | 0                 | 0                |
| Dynamic Shared Memory   | 0                 | 0                |
| Static Shared Memory    | 8.19 Kbyte/block  | 8.45 Kbyte/block |
| # SMs                   | 40                | 40               |
| Threads                 | 16777216          | 16777216         |
| Green Context           | 0                 | 0                |
| Waves/SM                | 409.60            | 409.60           |
| Block Limit SM          | 16                | 16               |
| Block Limit Registers   | 1                 | 1                |
| Block Limit Shared Mem  | 4                 | 3                |
| Block Limit Warps       | 1                 | 1                |
| Theoretical Ac Warps/SM | 32                | 32               |
| Theoretical Occupancy   | 100%              | 100%             |
| Achieved Occupancy      | 99.98%            | 99.99%           |
| Achieved Active Warps/SM| 32.00             | 32.00            |
| Avg DRAM Active Cycles  | 189220351.50 cycl | 189343963.50 cycl|
| Total DRAM Cycles       | 13380792320 cycles| 17366819840 cycl |
| Avg L1 Active Cycles    | 195586939.57 cycle| 253892040 cycles |
| Total L1 Cycles         | 7833692448 cycles | 10167926760 cycle|
| Avg L2 Active Cycles    | 76506660.69 cycles| 70395522.34 cycle|
| Total L2 Cycles         | 9160054208 cycles | 11888626848 cycle|
| Avg SM Active Cycles    | 195586939.57 cycle| 253892040 cycles |
| Total SM Cycles         | 7833692448 cycles | 10167926760 cycle|
| Avg SMSP Active Cycles  | 195587690.47 cycle| 253876400.93 cycl|
| Total SMSP Cycles       | 31334769792 cycle | 40671707040 cycle|
| Opt. Est. Speedup       | 8.321%            | 7.497%           |
```

We will provide an analysis of this in day 12!