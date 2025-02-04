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
    // Use a 2D grid: thread indices map to gene indices.
    int i = blockIdx.x * blockDim.x + threadIdx.x; // gene i
    int j = blockIdx.y * blockDim.y + threadIdx.y; // gene j

    // We compute only for valid pairs: we want j < i.
    if (i < nb_genes && j < i) {
        float sum = 0.0f;
        float sumsq = 0.0f;
        // Loop over the sample dimension (nb_samples is small)
        for (int k = 0; k < nb_samples; k++) {
            // Access the matrix as: row k, column gene (i or j)
            // Assuming d_Y is stored in column-major
            float ratio = d_Y[k + i * nb_samples] / d_Y[k + j * nb_samples];
            float log_val = log(ratio);
            sum += log_val;
            sumsq += log_val * log_val;
        }
        float mean = sum / nb_samples;
        float variance = (sumsq - nb_samples * mean * mean) / (nb_samples - 1);
        // Compute 1D index for (i,j) with j < i:
        int pair_index = (i * (i - 1)) / 2 + j;
        d_variances[pair_index] = variance;
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
    dim3 blockDim(16, 16); // 16*16 = 256
    dim3 gridDim((nb_genes + blockDim.x - 1) / blockDim.x, (nb_genes + blockDim.y - 1) / blockDim.y);
    computeLogRatioVariance<<<gridDim, blockDim>>>(d_Y, d_variances_gpu, nb_samples, nb_genes);
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
