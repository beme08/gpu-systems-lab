// labs/vector_add/vector_add.cu
// C[i] = A[i] + B[i] - the canonical memory-bound kernel.
// Try N = 1<<16, 1<<20, 1<<24 and block sizes 64, 128, 256, 1024.
// Report achieved GB/s = (3 * N * sizeof(float)) / kernel_time_seconds / 1e9.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 24);
    int block = (argc > 2) ? atoi(argv[2]) : 256;
    size_t bytes = (size_t)N * sizeof(float);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    std::vector<float> h(N, 1.0f);
    CUDA_CHECK(cudaMemcpy(dA, h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, h.data(), bytes, cudaMemcpyHostToDevice));

    int grid = (N + block - 1) / block;

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    vector_add<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);

    double gb = (3.0 * N * sizeof(float)) / 1e9;
    double bw = gb / (ms / 1000.0);
    printf("N=%d block=%d grid=%d time=%.3f ms  achieved=%.2f GB/s\n", N, block, grid, ms, bw);

    CUDA_CHECK(cudaMemcpy(h.data(), dC, bytes, cudaMemcpyDeviceToHost));
    printf("C[0]=%.1f C[%d]=%.1f\n", h[0], N-1, h[N-1]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
