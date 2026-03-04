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
__global__ void reduce_sum_kernel(
    T* in_data, 
    T* out_data, 
    TensorIndex in_idx,
    TensorIndex out_idx,
    int reduce_dim
) {
    extern __shared__ unsigned char shared_raw[];
    T* sdata = reinterpret_cast<T*>(shared_raw);

    int out_flat = blockIdx.x;
    if(out_flat >= out_idx.n_elements) return;

    int coords[MAX_DIMS];
    out_idx.to_index(out_flat, coords);

    int in_coords[MAX_DIMS];
    for(int i = 0; i < reduce_dim; i++) in_coords[i] = coords[i];
    for(int i = reduce_dim; i < in_idx.ndim-1; i++) in_coords[i+1] = coords[i];

    unsigned int tid = threadIdx.x;
    sdata[tid] = 0;

    for(int i = tid; i < in_idx.shape[reduce_dim]; i += blockSize) {
        in_coords[reduce_dim] = i;
        sdata[tid] += in_data[in_idx.to_flat(in_coords)];
    }
    __syncthreads();

    if (blockSize >= 512) { if(tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if(tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if(tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }
    if (tid < 32) reduce_sum_warp<T, blockSize>(sdata, tid);
    if (tid == 0) out_data[blockIdx.x] = sdata[0];
}

template<typename T>
void launch_reduce_sum(
    T* in_data, 
    T* out_data, 
    TensorIndex in_idx, 
    TensorIndex out_idx,
    int reduce_dim
) {
    int blocks = out_idx.n_elements;
    int threads = min(nextPow2(in_idx.shape[reduce_dim]), BLOCK_SIZE_1D);
    int smemSize = threads * sizeof(T);

    switch (threads) {
        case 512: reduce_sum_kernel<T, 512><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 256: reduce_sum_kernel<T, 256><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 128: reduce_sum_kernel<T, 128><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 64:  reduce_sum_kernel<T,  64><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 32:  reduce_sum_kernel<T,  32><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 16:  reduce_sum_kernel<T,  16><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 8:   reduce_sum_kernel<T,   8><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 4:   reduce_sum_kernel<T,   4><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 2:   reduce_sum_kernel<T,   2><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 1:   reduce_sum_kernel<T,   1><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
    }
}

template void launch_reduce_sum<float>(float*, float*, TensorIndex, TensorIndex, int);
template void launch_reduce_sum<half>(half*, half*, TensorIndex, TensorIndex, int);
template void launch_reduce_sum<uint8_t>(uint8_t*, uint8_t*, TensorIndex, TensorIndex, int);
template void launch_reduce_sum<int32_t>(int32_t*, int32_t*, TensorIndex, TensorIndex, int);

