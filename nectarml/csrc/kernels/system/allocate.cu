#include "kernels/common.h"

template<typename T>
__global__ void alloc_cuda_full_kernel(T* dst, size_t n, T fill_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = static_cast<T>(fill_value);
}

template<typename T>
void launch_alloc_cuda_full(T* dst, size_t n_elements, T fill_value) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    alloc_cuda_full_kernel<T><<<grid, block>>>(dst, n_elements, fill_value);
}

template void launch_alloc_cuda_full<float>(float*, size_t, float);
template void launch_alloc_cuda_full<half>(half*, size_t, half);
template void launch_alloc_cuda_full<uint8_t>(uint8_t*, size_t, uint8_t);
template void launch_alloc_cuda_full<int32_t>(int32_t*, size_t, int32_t);

template<typename T>
__global__ void alloc_cuda_random_kernel(
    T* dst, 
    size_t n, 
    curandState* random_state,
    unsigned long long seed,
    T min_value,
    T max_value
) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= n) return;

    curand_init(seed, idx, 0, &random_state[idx]);
    float min = static_cast<float>(min_value);
    float max = static_cast<float>(max_value);
    float rand_value = (curand_uniform(random_state+idx) - min) * (1.0 / (max - min));
    dst[idx] = static_cast<T>(rand_value);
}

template<typename T>
void launch_alloc_cuda_random(
    T* dst, 
    size_t n_elements,
    curandState* random_state,
    unsigned long long seed,
    T min_value, 
    T max_value
) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    alloc_cuda_random_kernel<T><<<grid, block>>>(
        dst, n_elements, random_state, seed, min_value, max_value);
}

template void launch_alloc_cuda_random<float>(float*, size_t, curandState*, unsigned long long, float, float);
template void launch_alloc_cuda_random<half>(half*, size_t, curandState*, unsigned long long, half, half);
template void launch_alloc_cuda_random<uint8_t>(uint8_t*, size_t, curandState*, unsigned long long, uint8_t, uint8_t);
template void launch_alloc_cuda_random<int32_t>(int32_t*, size_t, curandState*, unsigned long long, int32_t, int32_t);


