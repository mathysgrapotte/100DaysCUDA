#include <stdio.h>
#include <cuda_runtime.h>

const int width = 1024;
const int height = 1024;
dim3 block(16, 16); // 16x16 threads per block (=256 threads)
dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

// For width=1024, block.x=16:
// (1024 + 16 - 1) / 16 = 64 blocks in x-direction
// (1024 + 16 - 1) / 16 = 64 blocks in y-direction
// Total grid size: 64x64 blocks
// So that is 64*64*256 threads = 1024*1024 matrix memory adresses.

__global__ void matrixAdd(const float *A, const float *B, float *C, int width, int height){
    // 1. Calculate global thread ID
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Convert 2D position to a 1D index 
    // We convert 2D to 1D because memory is physically laid out linearly (1D) in hardware

    int idx = row * width + col; // row-major ordering

    // Row-major memory layout visualization:
    //
    // Matrix in 2D:         Memory in 1D (row-major):
    // [A00 A01 A02]        [A00 A01 A02 A10 A11 A12 A20 A21 A22]
    // [A10 A11 A12]   =>    |-- row0 --||-- row1 --||-- row2 --|
    // [A20 A21 A22]
    //
    // idx = row * width + col maps (row,col) to memory offset
    // e.g. A11: row=1, col=1 => idx = 1*3 + 1 = 4


    // 2. boundary check and compute operation 
    if (col < width && row < height){
        C[idx] = A[idx] + B[idx];
    }

}

int main(){
        // Host pointers
    float *h_A, *h_B, *h_C;
    // Device pointers
    float *d_A, *d_B, *d_C;
    
    // Allocate host memory
    size_t size = width * height * sizeof(float);
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);


    // Initialize host arrays - we use a 1D loop since matrices are stored linearly in memory
    // Even though we conceptually work with 2D matrices, physically they are laid out as 1D arrays
    // where each row is placed consecutively after the previous one
    for (int i = 0; i < width * height; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

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

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    matrixAdd<<<grid, block>>>(d_A, d_B, d_C, width, height);
    
    // Add error checking
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    cudaDeviceSynchronize();

    // Copy result back
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy result back: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Print first few results (don't print all million elements!)
    printf("First few results: \n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    // Free memory after printing
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
