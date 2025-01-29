# CUDA Thread Hierarchy

Grids contain block that contain threads

```
Grid
├── Block 0 (blockIdx.x = 0)
│   ├── Thread 0 (threadIdx.x = 0)
│   ├── Thread 1 (threadIdx.x = 1)
│   └── Thread 2 (threadIdx.x = 2)
│      ... (blockDim.x = 256)
├── Block 1 (blockIdx.x = 1)
│   ├── Thread 0 (threadIdx.x = 0)
│   ├── Thread 1 (threadIdx.x = 1)
│   └── Thread 2 (threadIdx.x = 2)
│      ...
```

# CUDA Thread Spatial Organization

Cuda threads can also be indexed using 2D indexing

```
Block (0,0)
Block (1,0)
...
Block (1,2)
├── Thread (0,0)
├── Thread (1,0) 
├── Thread (2,0)
├── Thread (3,0)
├── Thread (0,1)
├── Thread (1,1)
├── Thread (2,1) 
├── Thread (3,1)
├── Thread (0,2)
├── Thread (1,2)
├── Thread (2,2)
└── Thread (3,2)

blockDim.x = 4 (width)
blockDim.y = 3 (height)
```

# CUDA Thread Numbering

Thread numbering depends if we are using 1D, 2D, or 3D indexing, for our simple problem of vector addition, we will use 1D indexing

```
                            Grid
┌───────────────────────────────────────────────────────┐
│                                                       │
│   Block 0         Block 1         Block 2             │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐          │
│  │ T0  T1  │     │ T0  T1  │     │ T0  T1  │          │
│  │ T2  T3  │     │ T2  T3  │     │ T2  T3  │          │
│  └─────────┘     └─────────┘     └─────────┘          │
│                                                       │
│   Global IDs:     Global IDs:     Global IDs:         │
│   0   1          4   5           8   9                │
│   2   3          6   7           10  11               │
│                                                       │
└───────────────────────────────────────────────────────┘
```

Formula for global thread ID:
globalIdx = blockIdx.x * blockDim.x + threadIdx.x

Example for thread T1 in Block 1:
globalIdx = 1 * 4 + 1 = 5

Where:
- blockDim.x = 4 (threads per block)
- blockIdx.x = block number (0, 1, or 2)
- threadIdx.x = local thread number within block (0 to 3)


# How do blocks work ? 

Block execute in parallel with no guaranteed order
Each block must be able to execute independently 
You can't synchronize between blocks (only threads within the same block)

For instancem this would be incorrect :

```cuda
__global__ void kernel() {
    // This is NOT possible
    if (blockIdx.x == 2 && blockIdx.y == 1) {
        // Do computation A
    } else if (blockIdx.x == 2 && blockIdx.y == 2) {
        // Do computation B
    }
}
```

Therefore, in our algorithms, any block should be able to execute any part of the computation, and block should determine what data to run not what instruction to un 

```cuda
__global__ void kernel() {
    // Correct usage: use block indices to determine which data to process
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        // Process data at (row, col)
        // All blocks do the same computation, just on different data
    }
}
```

This is correct for instance. 

# How do units of computations (grid, blocks and threads) synchronize ?

## Thread level (within a block)

Threads within the same block can be synchronized, meaning all threads in a block will reach the same point (waiting i.e. performance loss)

```cuda 
__global__ void kernel() {
    // Some computation by all threads
    __syncthreads();  // Wait for all threads in block
    // Next computation
}
```
## Block level (within the grid)

There is no direct sync between blocks, those execute independently and in any order

```cuda
__global__ void incorrect_kernel() {
    // THIS IS WRONG - No guarantee block 1 executes before block 2
    if (blockIdx.x == 1) {
        // Do something
        // Try to signal block 2
    }
    if (blockIdx.x == 2) {
        // Wait for block 1's signal
        // Do something else
    }
}
```

## Grid level

Kernels execute on the grid, and are synced in between kernels

```cuda 
// Split into two kernels if you need block synchronization
__global__ void kernel1() {
    // Operations for first phase
}

__global__ void kernel2() {
    // Operations for second phase
}

// In host code:
kernel1<<<grid, block>>>();
cudaDeviceSynchronize();  // Wait for all blocks to complete
kernel2<<<grid, block>>>();
```

# CUDA Memory Management

## Memory Spaces
- **Host Memory**: CPU accessible memory (RAM)
- **Device Memory**: GPU accessible memory (VRAM)
- Each has their own separate address space

## cudaMalloc
```cpp
cudaMalloc((void**)&device_pointer, size_in_bytes);
```

Key points:
- Allocates memory on the GPU (device)
- Takes pointer to pointer (to modify the pointer value)
- Size in bytes (like regular malloc)
- Memory is uninitialized after allocation
- Must be freed with cudaFree()

## cudaMemcpy
```cpp
cudaMemcpy(destination, source, size_in_bytes, direction);
```

Direction types:
- cudaMemcpyHostToDevice: CPU → GPU (Upload)
- cudaMemcpyDeviceToHost: GPU → CPU (Download)
- cudaMemcpyDeviceToDevice: GPU → GPU (GPU internal copy)

Best Practices:
1. Minimize transfers between CPU and GPU
2. Batch operations to reduce transfer frequency
3. Keep data on GPU as long as possible
4. Consider operation complexity vs transfer cost
   - Example: Matrix addition (O(1) operations per transfer) - less beneficial
   - Example: Matrix multiplication (O(N) operations per transfer) - more beneficial

## Memory Management Pattern
```cpp
// 1. Allocate host memory
float* h_data = (float*)malloc(size);

// 2. Allocate device memory
float* d_data;
cudaMalloc((void**)&d_data, size);

// 3. Transfer to GPU
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// 4. Perform GPU operations
kernel<<<blocks, threads>>>(d_data);

// 5. Transfer results back
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

// 6. Free memory
cudaFree(d_data);
free(h_data);
```

## Advanced Memory Options
- cudaMallocPitch(): For 2D arrays (ensures proper alignment)
- cudaMalloc3D(): For 3D arrays
- cudaMallocManaged(): Unified Memory (automatically managed CPU-GPU transfers)
- cudaMallocHost(): Pinned memory for faster transfers
