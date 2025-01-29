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

Cuda threads are organized in block in a 2d organization 

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
