#include <common.h>

/* ADDITION */

template<typename T>
__global__ void add_kernel(T* a, T* b, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}

template<typename T>
void launch_add(T* a, T* b, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    add_kernel<T><<<grid, block>>>(a, b, out, n_elements);
}

template void launch_add<float>(float*, float*, float*, size_t);
template void launch_add<half>(half*, half*, half*, size_t);
template void launch_add<uint8_t>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_add<int32_t>(int32_t*, int32_t*, int32_t*, size_t);





