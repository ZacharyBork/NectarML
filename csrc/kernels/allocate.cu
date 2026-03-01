#include <common.h>

template<typename T>
__global__ void alloc_cuda_constant(T* dst, size_t n, T fill_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = static_cast<T>(fill_value);
}

template<typename T>
void launch_alloc_cuda_full(T* dst, size_t n_elements, T fill_value) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    alloc_cuda_constant<T><<<grid, block>>>(dst, n_elements, fill_value);
}

template void launch_alloc_cuda_full<float>(float*, size_t, float);
template void launch_alloc_cuda_full<half>(half*, size_t, half);
template void launch_alloc_cuda_full<uint8_t>(uint8_t*, size_t, uint8_t);
template void launch_alloc_cuda_full<int32_t>(int32_t*, size_t, int32_t);
