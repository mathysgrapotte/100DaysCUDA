#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>
#include <string.h>

// Matrix dimensions 
const int M = 4092;
const int K = 4092;
const int N = 4092;

// Add error checking macro at top
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Add error checking macro for CUPTI
#define CHECK_CUPTI(err) \
    do { \
        CUptiResult _err = (err); \
        if (_err != CUPTI_SUCCESS) { \
            const char *errstr; \
            cuptiGetResultString(_err, &errstr); \
            fprintf(stderr, "CUPTI error at %s:%d: %s\n", __FILE__, __LINE__, errstr); \
            exit(1); \
        } \
    } while(0)

// Update to use performance API metrics
const char* METRIC_NAMES[] = {
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",     // SM utilization
    "dram__bytes.sum.per_second",                             // Memory bandwidth
    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum"       // FMA instructions
};
const int NUM_METRICS = sizeof(METRIC_NAMES)/sizeof(METRIC_NAMES[0]);

dim3 grid((N + 32 - 1) / 32, (M + 32 - 1) / 32, 1);
dim3 block(32, 32, 1);

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
    float kernel_time;
    float memory_time;
    float total_time;
    float gflops;
    float sm_efficiency;
    float dram_throughput;
    float tensor_core_util;
    bool correct;
};

// CUPTI initialization function
void initCupti() {
    CUpti_SubscriberHandle subscriber;
    CHECK_CUPTI(cuptiSubscribe(&subscriber, 
        [](void* userdata, CUpti_CallbackDomain domain, 
           CUpti_CallbackId cbid, const void* cbdata) {}, 
        nullptr));
}

typedef void (*KernelFunction)(int, int, int, const float*, const float*, float*);

PerformanceMetrics runKernel(KernelFunction kernel, const char* name, 
                            const float* h_A, const float* d_A, 
                            const float* d_B, float* d_C) 
{
    PerformanceMetrics metrics = {0};
    float* h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize NVPW
    NVPW_InitializeHost_Params initializeHostParams = {NVPW_InitializeHost_Params_STRUCT_SIZE};
    NVPW_InitializeHost(&initializeHostParams);

    // Get current context and device
    CUcontext context;
    CUdevice device;
    cuCtxGetCurrent(&context);
    cuCtxGetDevice(&device);

    // Create CUDA events for timing
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

    // Calculate theoretical metrics
    float operations = 2.0f * M * N * K;  // multiply-add counts as 2 operations
    metrics.gflops = (operations / 1e9) / (metrics.kernel_time / 1000.0f);

    // Calculate memory throughput
    size_t bytes_read = M * K * sizeof(float) + K * N * sizeof(float);  // Reading A and B
    size_t bytes_written = M * N * sizeof(float);  // Writing to C
    float total_gb = (bytes_read + bytes_written) / (float)(1024*1024*1024);
    metrics.dram_throughput = total_gb / (metrics.kernel_time / 1000.0f);  // GB/s

    // Estimate SM efficiency based on occupancy
    int maxThreadsPerMultiProcessor;
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, 
                          cudaDevAttrMaxThreadsPerMultiProcessor, 
                          0);
    cudaDeviceGetAttribute(&maxThreadsPerBlock,
                          cudaDevAttrMaxThreadsPerBlock,
                          0);
    
    int numBlocks = grid.x * grid.y;
    int threadsPerBlock = block.x * block.y;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    
    // Calculate theoretical maximum blocks per SM
    int maxBlocksPerSM = maxThreadsPerMultiProcessor / threadsPerBlock;
    
    // Calculate blocks per SM (limited by both hardware and grid size)
    int blocksPerSM = min(numBlocks / numSMs, maxBlocksPerSM);
    
    // Calculate active threads per SM
    int activeThreadsPerSM = blocksPerSM * threadsPerBlock;
    
    // Calculate occupancy as a percentage
    metrics.sm_efficiency = (float)activeThreadsPerSM / maxThreadsPerMultiProcessor;

    // Clamp efficiency to 100%
    metrics.sm_efficiency = min(metrics.sm_efficiency, 1.0f);

    // Copy results back and verify
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    metrics.correct = verifyResults(h_C, h_A, M, N);

    // Print metrics
    printf("\n=== Performance Metrics (%s) ===\n", name);
    printf("Kernel Execution Time: %.3f ms\n", metrics.kernel_time);
    printf("GFLOPS: %.2f\n", metrics.gflops);
    printf("SM Efficiency (Occupancy): %.2f%%\n", metrics.sm_efficiency * 100.0f);
    printf("Memory Throughput: %.2f GB/s\n", metrics.dram_throughput);
    printf("Correctness: %s\n", metrics.correct ? "PASS" : "FAIL");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_C);

    return metrics;
}

int main() {
    // Initialize CUDA driver API
    cuInit(0);
    
    // define host pointers 
    float *h_A, *h_B, *h_C;
    // define device pointers 
    float *d_A, *d_B, *d_C;

    // initialise matrices sizes 
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
    printf("init matrices");
    // send data to GPU 
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

    printf("Going to run kernel\n");  // Add newline for proper output
    PerformanceMetrics metrics = runKernel(matmul_naive, "matmul_naive", h_A, d_A, d_B, d_C);
    printf("Done running kernel\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
}
