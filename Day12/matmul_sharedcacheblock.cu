#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Matrix dimensions 
const int M = 4096;
const int K = 4096;
const int N = 4096;

//dim3 grid((N + 32 - 1) / 32, (M + 32 - 1) / 32, 1);
//dim3 three_dblock(32, 32, 1);
//dim3 one_dblock(32*32);

// Update block and grid dimensions
const int BLOCKSIZE = 32;
dim3 grid((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
dim3 one_dblock(BLOCKSIZE * BLOCKSIZE);  // 1024 threads per block
dim3 three_dblock(BLOCKSIZE, BLOCKSIZE, 1);

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

bool verifyResults(float* C_gpu, const float* C_cpu, int M, int N) {
    const float epsilon = 1e-2;
    for(int i = 0; i<M*N; i++) {
        if(abs(C_gpu[i] - C_cpu[i]) > epsilon) {
            printf("Verification failed at index %d: GPU=%f, CPU=%f\n", i, C_gpu[i], C_cpu[i]);
            return false;
        }
    }
    return true;
}

void initializeMatrices(float* A, float* B, int M, int K, int N) {
    // Corrected parameter order in initialization
    // initialize A to random 
    for (int i=0; i<M*K; i++){
        A[i] = rand() / (float)RAND_MAX;
    }

    // initialize B to identity
    for (int i=0; i<K*N; i++){
        if (i % (N+1) == 0){
            B[i] = 1;
        } else {
            B[i] = 0;
        }
    }
}

// Performance metrics structure
struct PerformanceMetrics {
    float kernel_time;      // ms
    float gflops;          // GFLOPS achieved
    float sm_efficiency;    // Occupancy (0-1)
    float dram_throughput; // GB/s
    float arithmetic_intensity; // FLOPS/byte
    float bandwidth_utilization; // Percentage of theoretical
    float compute_utilization;   // Percentage of theoretical
    float memory_bound_ratio;    // >1 means memory bound, <1 means compute bound
    bool correct;
};


__global__ void matmul_naive(int M, int N, int K, const float *A,
                            const float *B, float *C) {
    int col_c = blockDim.x * blockIdx.x + threadIdx.x;
    int row_c = blockDim.y * blockIdx.y + threadIdx.y;

    if (col_c < N && row_c < M){
        float accu = 0.0f;
        for (int sum_index = 0; sum_index < K; sum_index+=1){
            accu += A[row_c * K + sum_index] * B[sum_index * N + col_c];
        }
        C[row_c * N + col_c] = accu;
    }
}
/*
__global__ void matmul_coal(int M, int N, int K, const float *A, 
                           const float* B, float *C) {
    const int BLOCKSIZE = 32;
    
    // Change indexing to minimize jumps between iterations
    const int tid = threadIdx.x;
    const int row = blockIdx.y * BLOCKSIZE + tid / BLOCKSIZE;
    const int col = blockIdx.x * BLOCKSIZE + tid % BLOCKSIZE;
    

    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Process elements in BLOCKSIZE chunks to maintain coalescing
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col + k * N];
        }
        
        C[row * N + col] = sum;
    }
}

*/

 // Start of Selection
__global__ void matmul_coal_optim(int M, int N, int K, const float *A, 
                                   const float* B, float *C) {
    const int BLOCKSIZE = 32;
    
    // Use a 1D thread index to derive 2D coordinates within the tile
    const int tid = threadIdx.x;
    const int localRow = tid >> 5;              // Equivalent to tid / 32
    const int localCol = tid & (BLOCKSIZE - 1);   // Equivalent to tid % 32
    const int globalRow = blockIdx.y * BLOCKSIZE + localRow;
    const int globalCol = blockIdx.x * BLOCKSIZE + localCol;
    
    // Allocate shared memory for A and B tiles
    __shared__ float sharedA[BLOCKSIZE][BLOCKSIZE];
    __shared__ float sharedB[BLOCKSIZE][BLOCKSIZE];
    
    float sum = 0.0f;
    
    // Loop over tiles in the K dimension
    // Loop over the tiles in the K dimension.
    // This loop divides the matrix multiplication task along the K dimension into smaller tiles,
    // each of size BLOCKSIZE. For example, if K = 100 and BLOCKSIZE = 32, then:
    //   (100 + 32 - 1) / 32 = 131 / 32, which equals 4 tiles.
    // This corresponds to:
    //   Tile 0: covers indices 0 to 31,
    //   Tile 1: covers indices 32 to 63,
    //   Tile 2: covers indices 64 to 95,
    //   Tile 3: covers indices 96 to 99 (with any extra elements padded with zeros).
    for (int tile = 0; tile < (K + BLOCKSIZE - 1) / BLOCKSIZE; tile++) {
        // Calculate the column index for A and row index for B in this tile
        int tiledA_col = tile * BLOCKSIZE + localCol;
        int tiledB_row = tile * BLOCKSIZE + localRow;
        
        // Load element into shared memory from A if within bounds; otherwise zero-pad
        if (globalRow < M && tiledA_col < K) {
            sharedA[localRow][localCol] = A[globalRow * K + tiledA_col];
        } else {
            sharedA[localRow][localCol] = 0.0f;
        }
        
        // Load element into shared memory from B if within bounds; otherwise zero-pad
        if (tiledB_row < K && globalCol < N) {
            sharedB[localRow][localCol] = B[tiledB_row * N + globalCol];
        } else {
            sharedB[localRow][localCol] = 0.0f;
        }
        
        __syncthreads(); // Ensure the tile is completely loaded
        
        // Compute the partial product for this tile
        for (int k = 0; k < BLOCKSIZE; k++) {
            sum += sharedA[localRow][k] * sharedB[k][localCol];
        }
        
        __syncthreads(); // Ensure all threads complete computation before next tile load
    }
    
    // Write the result back to global memory if within the valid output matrix range
    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = sum;
    }
}

/* Modified matmul_conflicts kernel to use 2D thread indices and improved debug prints to check memory coalescing. Note: This kernel requires a 2D block (e.g., dim3(32,32,1)) for correct operation. */
__global__ void matmul_conflicts(int M, int N, int K, const float *A, 
                                   const float* B, float *C) {
    const int BLOCKSIZE = 32;
    
    // Use 2D thread indices for proper mapping and coalescing
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    
    int globalRow = blockIdx.y * BLOCKSIZE + localRow;
    int globalCol = blockIdx.x * BLOCKSIZE + localCol;
    
    // Allocate shared memory with an extra column to avoid bank conflicts
    __shared__ float sharedA[BLOCKSIZE][BLOCKSIZE+1];
    __shared__ float sharedB[BLOCKSIZE][BLOCKSIZE+1];
    
    float sum = 0.0f;
    int numTiles = (K + BLOCKSIZE - 1) / BLOCKSIZE;
    
    for (int tile = 0; tile < numTiles; tile++) {
        int tiledA_col = tile * BLOCKSIZE + localCol;
        int tiledB_row = tile * BLOCKSIZE + localRow;
        
        // Load tile element from A
        if (globalRow < M && tiledA_col < K)
            sharedA[localRow][localCol] = A[globalRow * K + tiledA_col];
        else
            sharedA[localRow][localCol] = 0.0f;
        
        // Load tile element from B
        if (tiledB_row < K && globalCol < N)
            sharedB[localRow][localCol] = B[tiledB_row * N + globalCol];
        else
            sharedB[localRow][localCol] = 0.0f;
        
        __syncthreads();
        
        #ifdef DEBUG_PRINT
        // Detailed debug print for block (0,0) and tile 0 from thread (0,0)
        if (blockIdx.x == 0 && blockIdx.y == 0 && tile == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            int addrA = globalRow * K + tiledA_col;
            int addrB = tiledB_row * N + globalCol;
            printf("[DEBUG] Tile %d, thread(0,0): globalRow=%d, globalCol=%d, tiledA_col=%d, tiledB_row=%d, addrA=%d, A[addrA]=%f, addrB=%d, B[addrB]=%f\n",
                   tile, globalRow, globalCol, tiledA_col, tiledB_row, addrA, (globalRow < M && tiledA_col < K) ? A[addrA] : -1.0f, addrB, (tiledB_row < K && globalCol < N) ? B[addrB] : -1.0f);
        }
        #endif
        
        #pragma unroll
        for (int k = 0; k < BLOCKSIZE; k++) {
            sum += sharedA[localRow][k] * sharedB[k][localCol];
        }
        
        __syncthreads();
        
        #ifdef DEBUG_PRINT
        // Print partial sum from thread (0,0) for each tile
        if (globalRow == 0 && globalCol == 0 && tile == 0 && localRow == 0 && localCol == 0) {
            printf("[DEBUG] Tile %d: partial sum = %f\n", tile, sum);
        }
        #endif
    }
    
    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = sum;
    }
}

typedef void (*KernelFunction)(int, int, int, const float*, const float*, float*);

PerformanceMetrics runKernel(KernelFunction kernel, dim3 grid, dim3 block, const char* name, 
                            const float* h_A, const float* d_A, 
                            const float* d_B, float* d_C) 
{
    PerformanceMetrics metrics = {0};
    float* h_C = (float*)malloc(M * N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    // Run kernel
    kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics.kernel_time, start, stop);

    // Calculate basic metrics
    float operations = 2.0f * M * N * K;  // multiply-add counts as 2 operations

    // Calculate GFLOPS
    float seconds = metrics.kernel_time / 1000.0f;
    metrics.gflops = (operations / 1e9) / seconds;

    // Copy results back and verify
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    metrics.correct = verifyResults(h_C, h_A, M, N);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_C);
    if (metrics.correct) {
        printf("Kernel %s: Results Passed!\n", name);
    } else {
        printf("Kernel %s: Results Failed!\n", name);
    }
    printf("Kernel %s execution time: %.4f ms\n", name, metrics.kernel_time);

    return metrics;
}



int main() {
    // Initialize CUDA driver API
    cuInit(0);
    
    // define host pointers 
    float *h_A, *h_B, *h_C;
    // define device pointers 
    float *d_A, *d_B, *d_C;

    
    //initialise matrices sizes 
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    
    // allocate memory on the CPU
    h_A = (float*)malloc(size_a);
    h_B = (float*)malloc(size_b);
    h_C = (float*)malloc(size_c);
    
    // allocate memory on GPU
    CHECK_CUDA(cudaMalloc((void**)&d_A, size_a));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size_b));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size_c));

    initializeMatrices(h_A, h_B, M, K, N);
    // send data to GPU 
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

    PerformanceMetrics metrics = runKernel(matmul_naive, grid, three_dblock, "matmul_naive", h_A, d_A, d_B, d_C);
    //metrics = runKernel(matmul_coal, grid, one_dblock, "matmul_coal", h_A, d_A, d_B, d_C);
    metrics = runKernel(matmul_coal_optim, grid, one_dblock, "matmul_coal_optim", h_A, d_A, d_B, d_C);
    metrics = runKernel(matmul_conflicts, grid, three_dblock, "matmul_conflicts", h_A, d_A, d_B, d_C);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
}
