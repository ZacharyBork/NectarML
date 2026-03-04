// REFERENCE: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

#include "common.h"

/* SUM */

template<typename T, unsigned int blockSize>
__device__ void reduce_sum_warp(volatile T* sdata, int tid) {
    auto add = [](volatile T& a, volatile T b) {
        if constexpr (std::is_same_v<T, half>) {
            a = __hadd(a, b);
        } else {
            a += b;
        }
    };
    
    if (blockSize >= 64) add(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) add(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) add(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8)  add(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4)  add(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2)  add(sdata[tid], sdata[tid + 1]);
}

template<typename T, unsigned int blockSize>
__global__ void reduce_sum_kernel(T* in_data, T* out_data, size_t n_elements) {
    extern __shared__ unsigned char shared_raw[];
    T* sdata = reinterpret_cast<T*>(shared_raw);

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while (idx < n_elements) {
        sdata[tid] += in_data[idx];
        if (idx + blockSize < n_elements)
            sdata[tid] += in_data[idx + blockSize];
        idx += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if(tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if(tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if(tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }
    if (tid < 32) reduce_sum_warp<T, blockSize>(sdata, tid);
    if (tid == 0) out_data[blockIdx.x] = sdata[0];
}

template<typename T>
void launch_reduce_sum(T* in_data, T* out_data, size_t n_elements) {
    int threads = (n_elements < BLOCK_SIZE_1D * 2) ? nextPow2((n_elements + 1) / 2) : BLOCK_SIZE_1D;
    int blocks = (n_elements + (threads * 2 - 1)) / (threads * 2);
    int smemSize = threads * sizeof(T);

    switch (threads) {
        case 512: reduce_sum_kernel<T, 512><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 256: reduce_sum_kernel<T, 256><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 128: reduce_sum_kernel<T, 128><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 64:  reduce_sum_kernel<T,  64><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 32:  reduce_sum_kernel<T,  32><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 16:  reduce_sum_kernel<T,  16><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 8:   reduce_sum_kernel<T,   8><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 4:   reduce_sum_kernel<T,   4><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 2:   reduce_sum_kernel<T,   2><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 1:   reduce_sum_kernel<T,   1><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
    }

    if (blocks > 1) { launch_reduce_sum<T>(out_data, out_data, blocks); }
}

template void launch_reduce_sum<float>(float*, float*, size_t);
template void launch_reduce_sum<half>(half*, half*, size_t);
template void launch_reduce_sum<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_reduce_sum<int32_t>(int32_t*, int32_t*, size_t);

