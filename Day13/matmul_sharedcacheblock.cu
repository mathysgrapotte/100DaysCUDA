#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

// This line tells the compiler to use the nvcuda::wmma namespace.
// By doing so, you can directly access WMMA types and functions (such as fragment, load_matrix_sync, store_matrix_sync, etc.)
// without having to prepend them with nvcuda::wmma::. These functions are used for performing fast matrix
// operations on Tensor Cores in modern NVIDIA GPUs.
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Matrix dimensions 
const int M = 4096;
const int K = 4096;
const int N = 4096;

dim3 grid((N + 32 - 1) / 32, (M + 32 - 1) / 32, 1);
dim3 three_dblock(32, 32, 1);
dim3 one_dblock(32*32);

// Should be calculated based on WMMA tile dimensions:
dim3 grid_half((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
// Should use 1 warp per tile (32 threads per block):
dim3 wmma_block(32, 1, 1); // 32 threads = 1 warp

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

__global__ void matmul_naive_tensor_core(int M, int N, int K, const __half *A, const __half *B, float *C) {
    // Compute linear thread index for the current block (2D block: blockDim.x x blockDim.y threads)
    int tid_in_block = threadIdx.x + threadIdx.y * blockDim.x;
    
    // Calculate number of warps per block.
    int warps_per_block = (blockDim.x * blockDim.y) / warpSize;
    
    // Warp ID within the block.
    int warpId_in_block = tid_in_block / warpSize;
    
    // Compute global warp ID across the entire grid.
    int global_warp_id = (((blockIdx.y * gridDim.x) + blockIdx.x) * warps_per_block) + warpId_in_block;
    
    // Map the global warp ID to an output tile (row, col).
    // Ensure that the grid is configured such that the total number of warps equals (M/WMMA_M)*(N/WMMA_N)
    int tiles_per_row = N / WMMA_N;
    int tileRow = global_warp_id / tiles_per_row; // which tile row
    int tileCol = global_warp_id % tiles_per_row;   // which tile column

    // Declare the WMMA fragments (A and B are __half, accumulator is float)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
    
    // Initialize the output fragment to zero.
    wmma::fill_fragment(cFrag, 0.0f);

    // Loop over the K dimension (tile multiplication and accumulation)
    for (int k = 0; k < K; k += WMMA_K) {
        // Compute memory pointers for the current WMMA tile.
        int aRow = tileRow * WMMA_M;
        int bCol = tileCol * WMMA_N;
        
        // Get pointer to A tile
        const __half *tileA = A + aRow * K + k;
        // Get pointer to B tile
        const __half *tileB = B + k * N + bCol;
        
        // Load the tile from global memory into fragments.
        wmma::load_matrix_sync(aFrag, tileA, K);
        wmma::load_matrix_sync(bFrag, tileB, N);
        
        // Multiply and accumulate into cFrag using tensor core operations.
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }
    
    // Compute the starting position of the tile in the output matrix C.
    int cRow = tileRow * WMMA_M;
    int cCol = tileCol * WMMA_N;
    float *tileC = C + cRow * N + cCol;
    
    // Store the resulting 16x16 tile back to global memory.
    wmma::store_matrix_sync(tileC, cFrag, N, wmma::mem_row_major);
}

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

void convertFloatMatrixToHalf(const float* src, __half* dst, int numElements) {
    for (int i = 0; i < numElements; i++) {
        dst[i] = __float2half(src[i]);
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

// New typedef for kernels that take __half pointers (for tensor cores)
typedef void (*KernelFunctionHalf)(int, int, int, const __half*, const __half*, float*);

// Runner for tensor core kernels using __half input pointers.
PerformanceMetrics runKernelHalf(KernelFunctionHalf kernel, dim3 grid, dim3 block, const char* name,
                                 const float* h_A, const __half* d_A, const __half* d_B, float* d_C) {
    PerformanceMetrics metrics = {0};
    float* h_C = (float*)malloc(M * N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaEventElapsedTime(&metrics.kernel_time, start, stop);

    float operations = 2.0f * M * N * K;
    float seconds = metrics.kernel_time / 1000.0f;
    metrics.gflops = (operations / 1e9f) / seconds;

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    metrics.correct = verifyResults(h_C, h_A, M, N);

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

    // Allocate host memory for half precision matrices
    __half *h_A_half = (__half*)malloc(M * K * sizeof(__half));
    __half *h_B_half = (__half*)malloc(K * N * sizeof(__half));

    // Convert float matrices to half precision.
    convertFloatMatrixToHalf(h_A, h_A_half, M * K);
    convertFloatMatrixToHalf(h_B, h_B_half, K * N);

    __half *d_A_half; 
    __half *d_B_half;

    // allocate memory on GPU for half matrices
    CHECK_CUDA(cudaMalloc((void**)&d_A_half, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc((void**)&d_B_half, K * N * sizeof(__half)));

    // send data to GPU 
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

    // Copy the host half precision data to GPU
    CHECK_CUDA(cudaMemcpy(d_A_half, h_A_half, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_half, h_B_half, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    PerformanceMetrics metrics = runKernel(matmul_naive, grid, three_dblock, "matmul_naive", h_A, d_A, d_B, d_C);
    metrics = runKernel(matmul_coal_optim, grid, one_dblock, "matmul_coal_optim", h_A, d_A, d_B, d_C);
    metrics = runKernelHalf(matmul_naive_tensor_core, grid_half, wmma_block, "matmul_naive_tensor_core", h_A, d_A_half, d_B_half, d_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
}
