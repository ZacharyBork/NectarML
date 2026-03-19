#include "common.h"

template<typename SrcT, typename DstT>
__global__ void cast_kernel(SrcT* src, DstT* dst, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = static_cast<DstT>(src[idx]);
}

template<typename SrcT, typename DstT>
void launch_cast_kernel(SrcT* src, DstT* dst, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    cast_kernel<SrcT, DstT><<<grid, block>>>(src, dst, n_elements);
}

template void launch_cast_kernel<float, float>(float*, float*, size_t);
template void launch_cast_kernel<float, half>(float*, half*, size_t);
template void launch_cast_kernel<float, uint8_t>(float*, uint8_t*, size_t);
template void launch_cast_kernel<float, int32_t>(float*, int32_t*, size_t);

template void launch_cast_kernel<half, half>(half*, half*, size_t);
template void launch_cast_kernel<half, float>(half*, float*, size_t);
template void launch_cast_kernel<half, uint8_t>(half*, uint8_t*, size_t);
template void launch_cast_kernel<half, int32_t>(half*, int32_t*, size_t);

template void launch_cast_kernel<uint8_t, uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_cast_kernel<uint8_t, float>(uint8_t*, float*, size_t);
template void launch_cast_kernel<uint8_t, half>(uint8_t*, half*, size_t);
template void launch_cast_kernel<uint8_t, int32_t>(uint8_t*, int32_t*, size_t);

template void launch_cast_kernel<int32_t, int32_t>(int32_t*, int32_t*, size_t);
template void launch_cast_kernel<int32_t, half>(int32_t*, half*, size_t);
template void launch_cast_kernel<int32_t, float>(int32_t*, float*, size_t);
template void launch_cast_kernel<int32_t, uint8_t>(int32_t*, uint8_t*, size_t);

