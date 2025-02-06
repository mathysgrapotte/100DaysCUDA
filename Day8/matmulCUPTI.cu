#include <stdio.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <vector>

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
    float kernel_time;      // milliseconds
    float gflops;           // Floating point operations per second
    float bandwidth;        // memory bandwidth in GB/s
    float load_efficiency;  // Global memory load efficiency (%)
    float store_efficiency; // Global memory store efficiency (%)
    bool correct;           // Whether results match baseline
};

typedef void (*KernelFunction)(int, int, int, const float*, const float*, float*);

PerformanceMetrics runKernel(KernelFunction kernel, const char* name, 
                            const float* h_A, const float* d_A, 
                            const float* d_B, float* d_C) 
{
    PerformanceMetrics metrics;
    float* h_C = (float*)malloc(M * N * sizeof(float));

    // 1. Simple timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 2. Simplified CUPTI metric setup
    CUpti_MetricID metric;
    CUpti_EventGroupSets* eventGroupSets = nullptr;
    
    // Get efficiency metric (works for CUDA 12.4+)
    cuptiMetricGetIdFromName(0, "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.lsu", &metric);
    
    // 3. Create event groups for this metric
    cuptiMetricCreateEventGroupSets(0, sizeof(metric), &metric, &eventGroupSets);

    // 4. Enable event collection
    for(uint32_t i=0; i<eventGroupSets->numSets; i++) {
        cuptiEventGroupSetEnable(&eventGroupSets->sets[i]);
    }


    // 5. Run kernel with timing
    cudaEventRecord(start);
    kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics.kernel_time, start, stop);

    // 6. Put data back to CPU
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 7. Check results 
    metrics.correct = verifyResults(h_C, h_A, M, N);
    /*
    // 6. Get metric value (simplified for single metric)
    CUpti_MetricValue metricValue;
    size_t valueSize = sizeof(metricValue);
    cuptiMetricGetValue(0, metric, 0, nullptr, 0, nullptr, valueSize, &metricValue);

    // Convert raw metric to percentage (check metric units in docs)
    metrics.load_efficiency = static_cast<float>(metricValue.metricValueDouble);
    */
    // 7. Calculate basic performance metrics
    float operations = 2.0f * M * N * K;
    metrics.gflops = (operations / 1e9) / (metrics.kernel_time / 1000.0f);

    // print metrics
    printf("\n=== Performance Metrics (%s) ===\n", name);
    printf("Kernel Execution Time: %.3f ms\n", metrics.kernel_time);
    printf("GFLOPS: %.2f\n", metrics.gflops);
    //printf("Load Efficiency: %.2f%%\n", metrics.load_efficiency * 100.0f);
    //printf("Results match: %.2f\n", metrics.correct);
    
    // 9. Cleanup
    //cuptiEventGroupSetsDestroy(eventGroupSets);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_C);

    return metrics;
}

int main(){
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
