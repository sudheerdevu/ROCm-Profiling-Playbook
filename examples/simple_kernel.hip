/**
 * Simple kernel example for profiling demonstration
 */
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void matrixMul(const float* A, const float* B, float* C, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0.0f;
    
    for (int tile = 0; tile < (N + 15) / 16; ++tile) {
        if (row < N && tile * 16 + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + tile * 16 + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && tile * 16 + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < 16; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            
        __syncthreads();
    }
    
    if (row < N && col < N)
        C[row * N + col] = sum;
}

int main() {
    const int N = 1024;
    
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, N * N * sizeof(float));
    hipMalloc(&d_b, N * N * sizeof(float));
    hipMalloc(&d_c, N * N * sizeof(float));
    
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    
    // Warmup
    matrixMul<<<grid, block>>>(d_a, d_b, d_c, N);
    hipDeviceSynchronize();
    
    // Benchmark
    for (int i = 0; i < 100; ++i) {
        matrixMul<<<grid, block>>>(d_a, d_b, d_c, N);
    }
    hipDeviceSynchronize();
    
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    
    std::cout << "Done - profile with: rocprof --stats ./simple_kernel\n";
    return 0;
}
