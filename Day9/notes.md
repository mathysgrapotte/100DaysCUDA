## matrice sizes matter :

bus size is 128, 4092 for instance is not divisible by 128.

### size 4092 : 

=== Performance Analysis (matmul_coal) ===
Basic Metrics:
  Kernel Time: 208.421 ms
  GFLOPS: 657.50
  Memory Throughput: 0.96 GB/s
  SM Occupancy: 100.00%

Advanced Metrics:
  Arithmetic Intensity: 682.00 FLOPS/byte
  Theoretical Peak: 4070.40 GFLOPS
  Theoretical Bandwidth: 320.06 GB/s
  Compute Utilization: 16.15%
  Memory Bandwidth Utilization: 0.30%
  Memory/Compute Bound Ratio: 0.02 (Compute Bound)

but 4096 is ! (shave 7ms of compute)

### size 4096 :

  === Performance Analysis (matmul_coal) ===
Basic Metrics:
  Kernel Time: 201.570 ms
  GFLOPS: 681.84
  Memory Throughput: 1.00 GB/s
  SM Occupancy: 100.00%

Advanced Metrics:
  Arithmetic Intensity: 682.67 FLOPS/byte
  Theoretical Peak: 4070.40 GFLOPS
  Theoretical Bandwidth: 320.06 GB/s
  Compute Utilization: 16.75%
  Memory Bandwidth Utilization: 0.31%
  Memory/Compute Bound Ratio: 0.02 (Compute Bound)

## debuging for coalescing access 

At this stage, it is fairly understood that global memory should be accessed in a coalesced way. We can actually debug for this with a bunch of print statements (by adding this at the start of your kernel): 

```cpp 
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Memory access pattern for first warp:\n");
        for (int k = 0; k < 3; k++) {
            printf("\nIteration %d:\n", k);
            for (int t = 0; t < 32; t++) {
                printf("Thread %d: A[%d], B[%d]\n", 
                    t, 
                    (blockIdx.y * BLOCKSIZE + t / BLOCKSIZE) * K + k,
                    t + k * BLOCKSIZE  //replace by your access scheme
                );
            }
        }
    }
```

in that case we can see this in the output :

```
Iteration 0:
Thread 0: A[0], B[0]
Thread 1: A[0], B[1]
Thread 2: A[0], B[2]
Thread 3: A[0], B[3]
Thread 4: A[0], B[4]
Thread 5: A[0], B[5]
...
Thread 31: A[0], B[31]

Iteration 1:
Thread 0: A[1], B[32]
Thread 1: A[1], B[33]
Thread 2: A[1], B[34]
Thread 3: A[1], B[35]
Thread 4: A[1], B[36]
Thread 5: A[1], B[37]
...
Thread 31: A[1], B[63]

Iteration 2:
Thread 0: A[2], B[64]
Thread 1: A[2], B[65]
Thread 2: A[2], B[66]
Thread 3: A[2], B[67]
Thread 4: A[2], B[68]
Thread 5: A[2], B[69]
...
```

meaning that access is nicely coalesced !!! Those debugging statements are very useful when we need to understand if our memory access is well done (one day, I will write about how cuda is mostly understanding memory access).

