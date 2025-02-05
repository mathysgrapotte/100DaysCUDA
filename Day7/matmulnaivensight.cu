#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <vector>
#include <string>

// Matrix dimensions 
const int M = 4092; // first matrix is size (M,K), second matrix is size (K,N)
const int K = 4092; // output matrix is size (M,N) 
const int N = 4092; 

// create as many blocks as necessary to map all of C
dim3 gridDims((N + 32 - 1) / 32, (N + 32 - 1) / 32, 1);
// 32 * 32 = 1024 thread per block
dim3 blockDims(32, 32, 1);

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

bool verifyResults(float* C_gpu, float* C_cpu, int M, int N) {
    const float epsilon = 1e-2;
    for(int i = 0; i<M*N; i++) {
        if(abs(C_gpu[i] - C_cpu[i]) > epsilon) {
            printf("Verification failed at index %d: GPU=%f, CPU=%f\n", i, C_gpu[i], C_cpu[i]);
            return false;
        }
    }
    return true;
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
    cudaMalloc((void**)&d_A, size_a);
    cudaMalloc((void**)&d_B, size_b);
    cudaMalloc((void**)&d_C, size_c);

    // initialize A to random 
    for (int i=0; i<M*K; i++){
        h_A[i] = rand() / (float)RAND_MAX;
    }

    // initialize B to identity
    for (int i=0; i<K*N; i++){
        if (i % (N+1) == 0){
            h_B[i] = 1;
        } else {
            h_B[i] = 0;
        }
    }

    // send data to GPU 
    cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice);

    matmul_naive<<<gridDims, blockDims>>>(M, N, K, d_A, d_B, d_C);

    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size_c, cudaMemcpyDeviceToHost); // (destination, source, size, method)

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // check that results is exactly A: 
    bool result = verifyResults(h_C, h_A, M, N);
    printf("Verification Result: %s\n", result ? "Valid" : "Invalid");
    free(h_A);
    free(h_B);
    free(h_C);


}


