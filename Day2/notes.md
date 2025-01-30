# Threads

```cpp
__global__ void kernel(float* data) {
    int idx = threadIdx.x;
    
    // What physically happens:
    // 1. Thread loads instruction from instruction cache
    // 2. Thread requests data[idx] from memory
    // 3. Memory controller fetches data
    // 4. Data goes through cache hierarchy
    // 5. Data arrives in thread's registers
    float value = data[idx];
    // Process in registers
    value = value 2;
    // Write back to memory
    data[idx] = value;
```

Think of threads like workers at a library:
- The library is the global memory
- Workers (threads) don't "hold" books (data)
- They temporarily look at books (load into registers)
- Then put them back (store to memory)


# Memory layout 

## Physical Memory Layout

- GPU memory is organized in segments of 32/64/128 bytes (depending on architecture)
- Memory transactions happen in these segments, not individual bytes
- For example, on many NVIDIA GPUs, L1 cache line is 128 bytes

```
[128 bytes][128 bytes][128 bytes][128 bytes]
    ↑         ↑         ↑         ↑
 Segment1  Segment2  Segment3  Segment4
```

# Memory coalescing

A warp (32 threads) makes a transaction of a whole memory segment, which is composed of 31 individual memory locations of each 4 bytes. Therefore, if memory is aligned (i.e. each element of 4 bytes are contiguous on the same segment), we can fetch it in a single transaction for a warp.


Coalesced Access (Good):
Warp of 32 threads:
```
[T0][T1][T2]...[T31]
  ↓   ↓   ↓     ↓
[M0][M1][M2]...[M31] → Single 128-byte transaction

```
Memory Layout for floats:

```
[M0:4bytes][M1:4bytes][M2:4bytes]...[M31:4bytes]
    T0         T1         T2           T31
Total: 32 threads × 4 bytes = 128 bytes
```


Non-Coalesced Access (Bad):
Warp of 32 threads:

```
[T0][T1][T2]...[T31]
  ↓   ↓   ↓     ↓
[M0][MN][M2N]...[M31N] → Multiple transactions needed
```

Physical Process:
1. Memory Transaction 1 for T0
2. Memory Transaction 2 for T1
3. Memory Transaction 3 for T2
...and so on

Coalesced:    1 transaction × 128 bytes = 128 bytes transferred
Non-coalesced: 32 transactions × 4 bytes = 128 bytes transferred
              (But much slower due to multiple transactions)

In code, it translates this way :

```cpp
// Good: Threads in a warp access consecutive memory
// Thread 0 accesses element 0, Thread 1 accesses element 1, etc.
int idx = threadIdx.x + blockIdx.x * blockDim.x;
float element = input[idx];  // Coalesced access

// Bad: Threads access memory with stride
// Thread 0 accesses element 0, Thread 1 accesses element N, etc.
int idx = threadIdx.x * N;  // Strided access
float element = input[idx];  // Non-coalesced access
```

# Grid organization

```
Matrix (1024x1024)
[                     ]
[                     ]  ← Entire Problem Space
[                     ]

Split into Blocks:
[□□□□][□□□□][□□□□]    ← Grid of Blocks
[□□□□][□□□□][□□□□]    Each □ is a Block (16x16 threads)
[□□□□][□□□□][□□□□]
```

And the grid is a 2D array of blocks :

```
Grid (64x64 blocks):
[B0,0][B0,1][B0,2]...[B0,63]
[B1,0][B1,1][B1,2]...[B1,63]
[B2,0][B2,1][B2,2]...[B2,63]
  ...  ...  ...  ...
[B63,0][B63,1][B63,2]...[B63,63]
```

Where each block is a 2D organization of threads

```
One Block (16x16):
[T0,0][T0,1]...[T0,15]
[T1,0][T1,1]...[T1,15]
[T2,0][T2,1]...[T2,15]
  ...  ...  ...
[T15,0][T15,1]...[T15,15]
```

And coalescing is respected this way :
```
Block (0,0):              Block (0,1):
[T0,0][T0,1]...          [T0,0][T0,1]...
   ↓    ↓                   ↓    ↓
[M0,0][M0,1]...          [M0,16][M0,17]...
```

# Cuda sync 

```
Without Sync:
CPU: Launch kernel → Copy results → Continue
GPU:     →→→ Still computing →→→

With Sync:
CPU: Launch kernel → Wait for GPU → Copy results → Continue
GPU:     →→→ Computing →→→ Done ↑
                              Synchronization point
```

Without this, my code was freeing memory while the kernel was still computing !!