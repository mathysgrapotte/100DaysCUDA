#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <chrono>

// Matrix dimensions
const int nb_genes = 10000; // number of gens (columns)
const int nb_samples = 80; // number of samples (rows)


int threads_per_block = 128;
int num_pairs = (nb_genes * (nb_genes - 1)) / 2;
int num_blocks = num_pairs;

#define BLOCK_SIZE 128

__global__ void computeLogRatioVariance(float *d_Y, float *d_variances, int nb_samples, int nb_genes) {
    // Compute log ratio for current pair
    int pair_index = blockIdx.x;
    int i = (int)((-1 + sqrt(1 + 8.0 * pair_index)) / 2.0) + 1;
    int j = pair_index - (i * (i - 1)) / 2;

    // Initialize shared memory for mean and variance
    __shared__ float mean[BLOCK_SIZE];
    __shared__ float variance[BLOCK_SIZE];
    __shared__ float col_i[BLOCK_SIZE];
    __shared__ float col_j[BLOCK_SIZE];
    __shared__ float log_ratios[BLOCK_SIZE];  // New array to store original values

    // Initialize shared memory for mean and variance
    mean[threadIdx.x] = 0.0f;
    variance[threadIdx.x] = 0.0f;

    if (threadIdx.x < nb_samples) {
        col_i[threadIdx.x] = d_Y[threadIdx.x + i * nb_samples];
        col_j[threadIdx.x] = d_Y[threadIdx.x + j * nb_samples];
    } // load data into shared memory (each thread loads one sample at a time, so we need to sync)

    __syncthreads();

    // First compute log ratios
    if (threadIdx.x < nb_samples) {
        float ratio = col_i[threadIdx.x] / col_j[threadIdx.x];
        log_ratios[threadIdx.x] = log(ratio);  // Store original values
        mean[threadIdx.x] = log_ratios[threadIdx.x];  // Copy for reduction
    }
    __syncthreads();

    // Reduce to get sum for mean
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (threadIdx.x < stride) {
            mean[threadIdx.x] += mean[threadIdx.x + stride];
        }
    }
    __syncthreads();

    // Compute mean
    float log_ratio_mean = mean[0] / nb_samples;

    // Compute squared differences
    if (threadIdx.x < nb_samples) {
        float diff = log_ratios[threadIdx.x] - log_ratio_mean;  // Use original values
        variance[threadIdx.x] = diff * diff;
    }
    __syncthreads();

    // Reduce again to get sum of squared differences
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) { // >>= 1 is bitwise right shift by 1 (divide by 2)
        __syncthreads();
        if (threadIdx.x < stride) {
            variance[threadIdx.x] += variance[threadIdx.x + stride];
        }
    }

    // Store final result with (N-1) correction
    if (threadIdx.x == 0) {
        d_variances[pair_index] = variance[0] / (nb_samples - 1);
    }
}

// CPU implementation for log variance ratio benchmark
float* compute_log_variance_ratio_cpu(const float* Y, int nb_samples, int nb_genes) {
    // Output array to store variances for each pair
    int num_pairs = (nb_genes * (nb_genes - 1)) / 2;
    float* variances = new float[num_pairs];
    int counter = 0;

    // For each pair of genes
    for(int i = 1; i < nb_genes; i++) {
        for(int j = 0; j < i; j++) {
            float mean = 0.0f;
            float variance = 0.0f;
            
            // First pass: compute mean of log ratios
            for(int k = 0; k < nb_samples; k++) {
                float ratio = Y[k + i * nb_samples] / Y[k + j * nb_samples];
                mean += log(ratio);
            }
            mean /= nb_samples;
            
            // Second pass: compute variance
            for(int k = 0; k < nb_samples; k++) {
                float ratio = Y[k + i * nb_samples] / Y[k + j * nb_samples];
                float diff = log(ratio) - mean;
                variance += diff * diff;
            }
            
            // Divide by (N-1) for sample variance
            variances[counter] = variance / (nb_samples - 1);
            counter++;
        }
    }

    return variances;
}

struct PerformanceMetrics {
    float kernel_time;      // milliseconds
    float memory_time;      // milliseconds
    float total_time;       // milliseconds
    float gflops;          // Floating point operations per second
    float bandwidth;       // GB/s
};

void initializeMatrice(float* Y, int nb_samples, int nb_genes) {
    for(int i = 0; i < nb_samples * nb_genes; i++) Y[i] = rand() / (float)RAND_MAX;
}

bool verifyResults(float* variances_gpu, float* variances_cpu, int num_pairs) {
    const float epsilon = 1e-2;
    for(int i = 0; i < num_pairs; i++) {
        if(abs(variances_gpu[i] - variances_cpu[i]) > epsilon) {
            printf("Verification failed at index %d: GPU=%f, CPU=%f\n", i, variances_gpu[i], variances_cpu[i]);
            return false;
        }
    }
    return true;
}


PerformanceMetrics benchmarkLogVarianceRatio() {
    PerformanceMetrics metrics;
    
    // Allocate host memory
    float *h_Y = (float*)malloc(nb_samples * nb_genes * sizeof(float));
    float *h_variances_cpu = (float*)malloc(num_pairs * sizeof(float));
    float *h_variances_gpu = (float*)malloc(num_pairs * sizeof(float));
    
    // Initialize matrices
    initializeMatrice(h_Y, nb_samples, nb_genes);
    
    // Allocate device memory
    float *d_Y, *d_variances_gpu;
    cudaMalloc(&d_Y, nb_samples * nb_genes * sizeof(float));
    cudaMalloc(&d_variances_gpu, num_pairs * sizeof(float));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Memory transfer timing
    cudaEventRecord(start);
    cudaMemcpy(d_Y, h_Y, nb_samples * nb_genes * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics.memory_time, start, stop);
    
    cudaEventRecord(start);
    computeLogRatioVariance<<<num_blocks, threads_per_block>>>(d_Y, d_variances_gpu, nb_samples, nb_genes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics.kernel_time, start, stop);
    
    // Copy result back
    cudaMemcpy(h_variances_gpu, d_variances_gpu, num_pairs * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate performance metrics
    metrics.total_time = metrics.kernel_time + metrics.memory_time;
    float operations = 2.0f * nb_samples * nb_genes * nb_genes;  // multiply-add per element
    metrics.gflops = (operations / 1e9) / (metrics.kernel_time / 1000.0f);
    metrics.bandwidth = (3.0f * nb_samples * nb_genes * sizeof(float)) / (metrics.total_time * 1e6);  // GB/s
    
    // Verify results
    h_variances_cpu = compute_log_variance_ratio_cpu(h_Y, nb_samples, nb_genes);
    bool correct = verifyResults(h_variances_gpu, h_variances_cpu, num_pairs);
    
    // Print performance metrics
    printf("\nPerformance Metrics:\n");
    printf("Matrix Size: %dx%d\n", nb_samples, nb_genes);
    printf("  +-- Kernel Time:     %.2f ms\n", metrics.kernel_time);
    printf("  +-- Memory Time:     %.2f ms\n", metrics.memory_time);
    printf("Total Time: %.3f ms\n", metrics.total_time);
    printf("Performance: %.2f GFLOPs\n", metrics.gflops);
    printf("Memory Bandwidth: %.2f GB/s\n", metrics.bandwidth);
    printf("Results: %s\n", correct ? "PASSED" : "FAILED");
    
    // Cleanup
    free(h_Y); free(h_variances_cpu); free(h_variances_gpu);
    cudaFree(d_Y);
    cudaFree(d_variances_gpu);
    
    return metrics;
}

int main() {
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads in X-dimension: %d\n", prop.maxThreadsDim[0]);
    
    // Run benchmark
    PerformanceMetrics metrics = benchmarkLogVarianceRatio();

    // print the metrics to console 
    printf("\n=== Log Variance Ratio Benchmark Report ===\n");
    printf("============================================\n");
    printf("Performance Summary:\n");
    printf("--------------------------------------------\n");
    printf("Total Execution Time: %.2f ms\n", metrics.total_time);
    printf("  +-- Kernel Time:     %.2f ms\n", metrics.kernel_time); 
    printf("  +-- Memory Time:     %.2f ms\n", metrics.memory_time);
    printf("\nCompute Performance:\n");
    printf("--------------------------------------------\n");
    printf("GFLOP/s:             %.2f\n", metrics.gflops);
    printf("Memory Bandwidth:     %.2f GB/s\n", metrics.bandwidth);
    printf("============================================\n");

    // Run CPU benchmark for comparison
    float cpu_time;
    {
        float *h_Y = (float*)malloc(nb_samples * nb_genes * sizeof(float));
        float *h_variances_cpu = (float*)malloc(num_pairs * sizeof(float));
        
        initializeMatrice(h_Y, nb_samples, nb_genes);
        
        auto start_time = clock();
        compute_log_variance_ratio_cpu(h_Y, nb_samples, nb_genes);
        auto end_time = clock();
        
        cpu_time = (float)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0f; // Convert to ms
        
        free(h_Y);
        free(h_variances_cpu);
    }
    
    // Calculate CPU metrics
    float cpu_gflops = (2.0f * nb_samples * nb_genes * nb_genes) / (cpu_time * 1e6);
    
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
