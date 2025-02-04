#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <vector>
#include <string>

// Matrix dimensions 
const int M = 1024; // first matrix is size (M,K), second matrix is size (K,N)
const int K = 1024; // output matrix is size (M,N) 
const int N = 1024; 
const int TILE_SIZE = 16;

dim3 block(16, 16); // 16x16 threads per block (=256 threads)
dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

// Error checking macro
// Macros in C are text substitution directives processed by the preprocessor
// They follow the syntax: #define MACRO_NAME replacement_text

// This macro takes a value parameter and expands to a function call
// Example usage: CHECK_CUDA_ERROR(cudaMalloc(...))
// The preprocessor will replace it with: check(cudaMalloc(...), "cudaMalloc(...)", __FILE__, __LINE__)
// #val becomes the string literal of the parameter
// __FILE__ and __LINE__ are built-in macros for current file and line number
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

// This is the actual function that does the error checking
// It's a template to handle different CUDA API return types
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at: %s:%d\n", file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        exit(1);
    }
}

// Performance metrics structure
struct PerformanceMetrics {
    float kernel_time;      // milliseconds
    float memory_time;      // milliseconds
    float total_time;       // milliseconds
    float gflops;          // Floating point operations per second
    float bandwidth;       // GB/s
    bool correct;          // Whether results match baseline
};

// Kernel function pointer type
typedef void (*KernelFunction)(const float*, const float*, float*, int, int, int);

__global__ void MatrixMulNaive(const float* A, const float* B, float* C, int M, int N, int K){

    // Calculate global thread indices:
    //
    // For a 1024x1024 matrix with 16x16 thread blocks:
    // blockIdx.y ranges from 0 to (1024/16)-1 = 63  (block row)
    // blockDim.y = 16                               (threads per block in y)
    // threadIdx.y ranges from 0 to 15               (thread row within block)
    //
    // Example for row index 272:
    // blockIdx.y = 17, blockDim.y = 16, threadIdx.y = 0
    // row = 17 * 16 + 0 = 272

    // Similarly for column index:
    // blockIdx.x ranges from 0 to 63                (block column) 
    // blockDim.x = 16                              (threads per block in x)
    // threadIdx.x ranges from 0 to 15              (thread col within block)
    //
    // Example for col index 131:
    // blockIdx.x = 8, blockDim.x = 16, threadIdx.x = 3  
    // col = 8 * 16 + 3 = 131

    // To access element (272,131) in a 1D array representing a 2D matrix:
    // We need to convert 2D coordinates (row,col) to 1D index
    // For a matrix of width N, the formula is: index = row * N + col
    //
    // Example for element (272,131) in 1024x1024 matrix:
    // index = 272 * 1024 + 131 = 278,659
    //
    // This mapping preserves row-major order where elements in same row
    // are contiguous in memory for coalesced access:
    // row 0: [0,1,2,...,1023]
    // row 1: [1024,1025,1026,...,2047]
    // row 2: [2048,2049,2050,...,3071]
    // etc.


    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void MatrixMulTiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float A_tile[16][16];
    __shared__ float B_tile[16][16];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;  // This is the 'accu' from pseudocode

    // Accumulate C tile by tile
    for (int tileIdx = 0; tileIdx < K; tileIdx+=16){
    
    /*
     * This tiling approach works for non-square matrices because:
     * 
     * 1. The loop iterates over dimension K in steps of TILE_SIZE (16)
     *    - K is the common dimension between matrices (width of A, height of B)
     *    - Tiling K works regardless of M×K and K×N matrix shapes
     * 
     * 2. The dot product is computed along dimension K:
     *    - Each tile processes TILE_SIZE elements of this dot product
     *    - Final result combines all tiles independently of matrix shapes
     * 
     * 3. Tiling optimizes memory access patterns:
     *    - Loads TILE_SIZE x TILE_SIZE blocks into shared memory
     *    - Reduces global memory accesses by TILE_SIZE factor
     */

        // load one tile of A and one tile of B into shared memory
        if (row < M && (tileIdx + threadIdx.x) < K) {
            // We use [y][x] because:
            // 1. threadIdx.y maps to rows (y dimension) 
            // 2. threadIdx.x maps to columns (x dimension) (so essentially loading A[i,j])
            A_tile[threadIdx.y][threadIdx.x] = A[row * K + tileIdx + threadIdx.x]; // global mem coalesced
            // here tileIdx is essentially tileid*16 but we are incrementing tileIdx 16 by 16 (tile size)
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        

        // Match pseudocode exactly: B_tile(threadIdx.x, threadIdx.y) <= B_gpu(j,i)
        if (col < N && (tileIdx + threadIdx.y) < K) {
            B_tile[threadIdx.y][threadIdx.x] = B[(tileIdx + threadIdx.y) * N + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // compute tile partial sum 
        if (row < M && col < N) {
            for (int k = 0; k < 16 && (tileIdx + k) < K; k++) {
                sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // Store final result - matches pseudocode: C_gpu(i,j) <= accu
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void MatrixMulTiledCoalesced(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float A_tile[16][16];
    __shared__ float B_tile[16][16];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;

    for (int tileIdx = 0; tileIdx < K; tileIdx+=16){

        // load one tile of A and one tile of B into shared memory
        if (row < M && (tileIdx + threadIdx.x) < K) {
            // We use [y][x] because:
            // 1. threadIdx.y maps to rows (y dimension) 
            // 2. threadIdx.x maps to columns (x dimension) (so essentially loading A[i,j])
            A_tile[threadIdx.y][threadIdx.x] = A[row * K + tileIdx + threadIdx.x]; // global mem coalesced
            // here tileIdx is essentially tileid*16 but we are incrementing tileIdx 16 by 16 (tile size)
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int load_row = tileIdx + threadIdx.y;
        int load_col = blockIdx.x * 16 + threadIdx.x; //16 corresponds to tile size


        // Match pseudocode exactly: B_tile(threadIdx.x, threadIdx.y) <= B_gpu(j,i)
        if (col < N && (tileIdx + threadIdx.y) < K) {
            B_tile[threadIdx.y][threadIdx.x] = B[load_row * N + load_col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // compute tile partial sum 
        if (row < M && col < N) {
            for (int k = 0; k < 16 && (tileIdx + k) < K; k++) {
                sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Function to initialize matrices
void initializeMatrices(float* A, float* B, int M, int K, int N) {
    for(int i = 0; i < M * K; i++) A[i] = rand() / (float)RAND_MAX;
    for(int i = 0; i < K * N; i++) B[i] = rand() / (float)RAND_MAX;
}

// Function to verify results against baseline
bool verifyResults(float* C_test, float* C_baseline, int M, int N, const char* kernel_name) {
    const float epsilon = 1e-2;
    bool passed = true;
    float max_diff = 0.0f;
    int max_diff_idx = 0;
    
    for(int i = 0; i < M * N; i++) {
        float diff = abs(C_test[i] - C_baseline[i]);
        if(diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        if(diff > epsilon) {
            printf("Verification failed for %s at index %d: Test=%f, Baseline=%f, Diff=%f\n", 
                   kernel_name, i, C_test[i], C_baseline[i], diff);
            passed = false;
            break;
        }
    }
    
    printf("Max difference: %f at index %d\n", max_diff, max_diff_idx);
    return passed;
}

// CPU implementation for verification
void matrixMulCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0.0f;
            for(int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

PerformanceMetrics runKernel(KernelFunction kernel, const char* name, float* d_A, float* d_B, float* d_C, float* h_C_baseline) {
    PerformanceMetrics metrics;
    float* h_C = (float*)malloc(M * N * sizeof(float));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Kernel timing
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&metrics.kernel_time, start, stop));
    
    // Memory timing for copy back
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&metrics.memory_time, start, stop));
    
    // Update total time
    metrics.total_time = metrics.kernel_time + metrics.memory_time;
    
    // Calculate performance metrics
    float operations = 2.0f * M * N * K;
    metrics.gflops = (operations / 1e9) / (metrics.kernel_time / 1000.0f);
    metrics.bandwidth = (3.0f * M * N * sizeof(float)) / (metrics.total_time * 1e6);
    
    // Print performance metrics first
    printf("\n=== %s Performance ===\n", name);
    printf("Kernel Time: %.3f ms\n", metrics.kernel_time);
    printf("Memory Time: %.3f ms\n", metrics.memory_time);
    printf("Total Time: %.3f ms\n", metrics.total_time);
    printf("Performance: %.2f GFLOPs\n", metrics.gflops);
    printf("Memory Bandwidth: %.2f GB/s\n", metrics.bandwidth);
    
    // Verify results and print verification separately
    metrics.correct = verifyResults(h_C, h_C_baseline, M, N, name);
    printf("Results: %s\n", metrics.correct ? "PASSED" : "FAILED");
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    free(h_C);
    
    return metrics;
}

std::vector<PerformanceMetrics> benchmarkMatMul() {
    std::vector<PerformanceMetrics> all_metrics;
    
    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C_cpu = (float*)malloc(M * N * sizeof(float));  // CPU result (ground truth)
    
    // Initialize matrices
    initializeMatrices(h_A, h_B, M, K, N);
    
    // Generate CPU result first - this is our ground truth
    auto cpu_start = clock();
    matrixMulCPU(h_A, h_B, h_C_cpu, M, N, K);
    auto cpu_end = clock();
    float cpu_time = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0f;
    
    // Create CPU metrics
    PerformanceMetrics cpu_metrics;
    cpu_metrics.kernel_time = cpu_time;
    cpu_metrics.memory_time = 0;  // Not measuring for CPU
    cpu_metrics.total_time = cpu_time;
    float operations = 2.0f * M * N * K;
    cpu_metrics.gflops = (operations / 1e9) / (cpu_time / 1000.0f);
    cpu_metrics.bandwidth = 0;  // Not measuring for CPU
    cpu_metrics.correct = true;  // CPU is ground truth
    all_metrics.push_back(cpu_metrics);
    
    printf("\n=== CPU Implementation ===\n");
    printf("Time: %.3f ms\n", cpu_time);
    printf("Performance: %.2f GFLOPs\n", cpu_metrics.gflops);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run implementations with proper names
    PerformanceMetrics naive_metrics = runKernel(MatrixMulNaive, "Naive GPU Implementation", 
                                                d_A, d_B, d_C, h_C_cpu);
    all_metrics.push_back(naive_metrics);
    cudaDeviceSynchronize();
    
    PerformanceMetrics tiled_metrics = runKernel(MatrixMulTiled, "Tiled Implementation", 
                                                    d_A, d_B, d_C, h_C_cpu);
    all_metrics.push_back(tiled_metrics);
    cudaDeviceSynchronize();

    // Run coalesced implementation
    PerformanceMetrics coalesced_metrics = runKernel(MatrixMulTiledCoalesced, "Tiled Coalesced Implementation", 
                                                    d_A, d_B, d_C, h_C_cpu);  // Compare against CPU

    all_metrics.push_back(coalesced_metrics);
    cudaDeviceSynchronize();
    
    // Cleanup
    free(h_A); free(h_B); free(h_C_cpu);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    
    return all_metrics;
}

int main() {
    // Get device properties
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("\nDevice: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads in X-dimension: %d\n", prop.maxThreadsDim[0]);
    
    // Run benchmarks
    std::vector<PerformanceMetrics> metrics = benchmarkMatMul();
    
    // Print comparison report
    printf("\n=== Performance Comparison Report ===\n");
    printf("====================================================\n");
    printf("Implementation         Time(ms)   GFLOP/s   Speedup\n");
    printf("----------------------------------------------------\n");
    
    float cpu_time = metrics[0].total_time;
    float gpu_baseline_time = metrics[1].total_time;  // Naive GPU implementation
    
    // Print CPU performance (relative to GPU naive)
    printf("%-20s %9.2f %9.2f %8.3fx\n", 
           "CPU", 
           cpu_time,
           metrics[0].gflops,
           gpu_baseline_time / cpu_time);  // This will be < 1
           
    // Print GPU implementations (speedup relative to GPU naive)
    const char* names[] = {"Naive GPU (Baseline)", "Tiled", "Tiled Coalesced"};
    for(size_t i = 1; i < metrics.size(); i++) {
        float speedup = gpu_baseline_time / metrics[i].total_time;
        printf("%-20s %9.2f %9.2f %8.2fx\n", 
               names[i-1], 
               metrics[i].total_time,
               metrics[i].gflops,
               speedup);
    }
    printf("====================================================\n");
    
    return 0;
}
