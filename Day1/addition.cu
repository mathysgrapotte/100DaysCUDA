// stdio.h is a standard C header file that provides input/output functionality
// It contains functions like printf() for printing to console and scanf() for reading input
// We need it here to print debug information and results from our CUDA program
#include <stdio.h>

// cuda_runtime.h is a CUDA header file that provides runtime functionality
// It contains functions like cudaMalloc() for memory allocation and cudaFree() for memory deallocation
// We need it here to allocate memory for our CUDA program
#include <cuda_runtime.h>

// Let's define parameters for the kernel:
const int threadsPerBlock = 256; // 8 warps per block (so 32 * 8 = 256 threads per block)
const int N = 1024; // size of arrays
const int blocksNeeded = (N + threadsPerBlock - 1) / threadsPerBlock;  // Find out how many blocks are needed 

/* explanation for N = 1000:
N = 1000, threadsPerBlock = 256

Block 0: Threads 0-255    (handles elements 0-255)
Block 1: Threads 256-511  (handles elements 256-511)
Block 2: Threads 512-767  (handles elements 512-767)
Block 3: Threads 768-999  (handles elements 768-999)
         Threads 1000-1023 (idle due to boundary check) Aha! this is why arrays being of size ^2 is good

Why This Approach?
    Coalesced Memory Access:
        Adjacent threads access adjacent memory locations
        Maximizes memory bandwidth utilization
    Load Balancing:
        Even distribution of work across threads
        No thread does more work than others
    -> Efficient use of GPU resources
    Scalability:
        Works for any input size
        Automatically adjusts number of blocks
        Handles edge cases gracefully
*/

// Kernel definition 
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    // 1. Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /*
    blockIdx.x: The index of the block in the grid
    blockDim.x: The number of threads per block
    threadIdx.x: The index of the thread within its block

    Block 0: idx = 0*256 + (0 to 255)   → Threads 0-255
    Block 1: idx = 1*256 + (0 to 255)   → Threads 256-511
    Block 2: idx = 2*256 + (0 to 255)   → Threads 512-767
    Block 3: idx = 3*256 + (0 to 255)   → Threads 768-1023

    */
    
    // 2. Boundary check
    if (idx < N) {
        // 3. Perform addition
        C[idx] = A[idx] + B[idx];
    }

    /* vectorAdd is only modifying the elements of C
    This function returns nothing because otherwise each thread would need to return a value
    which would be inefficient.

    Instead, those functions write to the global memory (hence the __global__ keyword) and the void keyword

    Cuda philosophy:
        Memory Efficiency:
            Output memory is pre-allocated
            Results are written directly to GPU memory
            No need for intermediate storage
        Parallel Processing Model:
            Each thread writes to its designated memory location
            No contention or synchronization needed
            Follows SIMT (Single Instruction, Multiple Thread) model
        Performance:
            Direct memory writes are more efficient
            No need to collect and manage return values
            Better utilization of GPU memory bandwidth
    */ 
}

int main() {
    // here we need to do : 
    // 1. Define pointers and size of arrays
    // host arrays (cpu pointers)
    float *h_A, *h_B, *h_C;
    // device arrays (gpu pointers)
    float *d_A, *d_B, *d_C;

    // Size of arrays
    size_t size = N * sizeof(float);

    // 2. Allocate memory for host arrays A, B, and C
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // 3. Allocate memory for device arrays A, B, and C
    cudaError_t err;
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        printf("Failed to allocate memory for d_A\n");
        return -1;
    }
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        printf("Failed to allocate memory for d_B\n");
        return -1;
    }
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        printf("Failed to allocate memory for d_C\n");
        return -1;
    }

    // 4. Initialize host arrays A and B (all elements to 1)
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // 5. Transfer data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 6. Launch kernel
    vectorAdd<<<blocksNeeded, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 7. Transfer data from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 8. Free memory on device
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 9. Verify results
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 2.0f) {
            printf("Error at index %d: h_C[%d] = %f\n", i, i, h_C[i]);
            return -1;
        }
    }

    // 10. Print results
    printf("Results: ");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    // 11. Free memory for h_C
    free(h_C);

    // 12. Return 0
    return 0;
}