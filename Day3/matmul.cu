#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <chrono>

// Matrix dimensions 
const int M = 1024; // first matrix is size (M,K), second matrix is size (K,N)
const int K = 1024; // output matrix is size (M,N) 
const int N = 1024; 

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
};

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

// Function to initialize matrices
void initializeMatrices(float* A, float* B, float* C, int M, int K, int N) {
    for(int i = 0; i < M * K; i++) A[i] = rand() / (float)RAND_MAX;
    for(int i = 0; i < K * N; i++) B[i] = rand() / (float)RAND_MAX;
    for(int i = 0; i < M * N; i++) C[i] = 0.0f;
}

// Function to verify results against CPU implementation
bool verifyResults(float* C_gpu, float* C_cpu, int M, int N) {
    const float epsilon = 1e-2;
    for(int i = 0; i < M * N; i++) {
        if(abs(C_gpu[i] - C_cpu[i]) > epsilon) {
            printf("Verification failed at index %d: GPU=%f, CPU=%f\n", i, C_gpu[i], C_cpu[i]);
            return false;
        }
    }
    return true;
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

PerformanceMetrics benchmarkMatMul() {
    PerformanceMetrics metrics;
    
    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    float *h_C_cpu = (float*)malloc(M * N * sizeof(float));
    
    // Initialize matrices
    initializeMatrices(h_A, h_B, h_C, M, K, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Memory transfer timing
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&metrics.memory_time, start, stop));
    
    // Kernel timing
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    MatrixMulNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&metrics.kernel_time, start, stop));
    
    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Calculate performance metrics
    metrics.total_time = metrics.kernel_time + metrics.memory_time;
    float operations = 2.0f * M * N * K;  // multiply-add per element
    metrics.gflops = (operations / 1e9) / (metrics.kernel_time / 1000.0f);
    metrics.bandwidth = (3.0f * M * N * sizeof(float)) / (metrics.total_time * 1e6);  // GB/s
    
    // Verify results
    matrixMulCPU(h_A, h_B, h_C_cpu, M, N, K);
    bool correct = verifyResults(h_C, h_C_cpu, M, N);
    
    // Print performance metrics
    printf("\nPerformance Metrics:\n");
    printf("Matrix Size: %dx%d * %dx%d\n", M, K, K, N);
    printf("Kernel Time: %.3f ms\n", metrics.kernel_time);
    printf("Memory Transfer Time: %.3f ms\n", metrics.memory_time);
    printf("Total Time: %.3f ms\n", metrics.total_time);
    printf("Performance: %.2f GFLOPs\n", metrics.gflops);
    printf("Memory Bandwidth: %.2f GB/s\n", metrics.bandwidth);
    printf("Results: %s\n", correct ? "PASSED" : "FAILED");
    
    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_C_cpu);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    
    return metrics;
}

int main() {
    // Get device properties
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("\nDevice: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads in X-dimension: %d\n", prop.maxThreadsDim[0]);
    
    // Run benchmark
    PerformanceMetrics metrics = benchmarkMatMul();

    // print the metrics to console 
    printf("\n=== Matrix Multiplication Benchmark Report ===\n");
    printf("============================================\n");
    printf("Performance Summary:\n");
    printf("--------------------------------------------\n");
    printf("Total Execution Time: %.2f ms\n", metrics.total_time);
    printf("  ├─ Kernel Time:     %.2f ms\n", metrics.kernel_time); 
    printf("  └─ Memory Time:     %.2f ms\n", metrics.memory_time);
    printf("\nCompute Performance:\n");
    printf("--------------------------------------------\n");
    printf("GFLOP/s:             %.2f\n", metrics.gflops);
    printf("Memory Bandwidth:     %.2f GB/s\n", metrics.bandwidth);
    printf("============================================\n");

    // Run CPU benchmark for comparison
    float cpu_time;
    {
        float *h_A = (float*)malloc(M * K * sizeof(float));
        float *h_B = (float*)malloc(K * N * sizeof(float));
        float *h_C_cpu = (float*)malloc(M * N * sizeof(float));
        
        initializeMatrices(h_A, h_B, h_C_cpu, M, K, N);
        
        auto start_time = clock();
        matrixMulCPU(h_A, h_B, h_C_cpu, M, N, K);
        auto end_time = clock();
        
        cpu_time = (float)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0f; // Convert to ms
        
        free(h_A);
        free(h_B); 
        free(h_C_cpu);
    }
    
    // Calculate CPU metrics
    float cpu_gflops = (2.0f * M * N * K) / (cpu_time * 1e6);
    
    printf("\n=== CPU vs GPU Comparison ===\n");
    printf("--------------------------------------------\n");
    printf("CPU Time:             %.2f ms\n", cpu_time);
    printf("GPU Time:             %.2f ms\n", metrics.total_time);
    printf("Speedup:              %.2fx\n", cpu_time / metrics.total_time);
    printf("\nCompute Performance:\n");
    printf("CPU GFLOP/s:          %.2f\n", cpu_gflops);
    printf("GPU GFLOP/s:          %.2f\n", metrics.gflops);
    printf("Performance Ratio:     %.2fx\n", metrics.gflops / cpu_gflops);
    printf("============================================\n");
    
    return 0;
}
