// REFERENCE: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

#include "common.h"

/* MIN */

template<typename T, unsigned int blockSize>
__device__ void reduce_min_warp(volatile T* sdata, int tid) {
    auto minimum = [](volatile T& a, volatile T b) {
        if constexpr (std::is_same_v<T, half>) { a = __hmin(a, b); } 
        else { a = min(a, b); }
    };
    
    if (blockSize >= 64) minimum(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) minimum(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) minimum(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8)  minimum(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4)  minimum(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2)  minimum(sdata[tid], sdata[tid + 1]);
}

template<typename T, unsigned int blockSize>
__global__ void reduce_min_kernel(T* in_data, T* out_data, size_t n_elements) {
    auto minimum = [](volatile T& a, volatile T b) {
        if constexpr (std::is_same_v<T, half>) { a = __hmin(a, b); } 
        else { a = min(a, b); }
    };

    extern __shared__ unsigned char shared_raw[];
    T* sdata = reinterpret_cast<T*>(shared_raw);

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = max_val<T>();

    while (idx < n_elements) {
        minimum(sdata[tid], in_data[idx]);
        if (idx + blockSize < n_elements)
            minimum(sdata[tid], in_data[idx + blockSize]);
        idx += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if(tid < 256) minimum(sdata[tid], sdata[tid + 256]); __syncthreads(); }
    if (blockSize >= 256) { if(tid < 128) minimum(sdata[tid], sdata[tid + 128]); __syncthreads(); }
    if (blockSize >= 128) { if(tid < 64)  minimum(sdata[tid], sdata[tid + 64]);  __syncthreads(); }
    if (tid < 32) reduce_min_warp<T, blockSize>(sdata, tid);
    if (tid == 0) out_data[blockIdx.x] = sdata[0];
}

template<typename T, unsigned int blockSize>
__global__ void reduce_min_dim_kernel(
    T* in_data, 
    T* out_data, 
    TensorIndex in_idx,
    TensorIndex out_idx,
    int reduce_dim
) {
    auto minimum = [](volatile T& a, volatile T b) {
        if constexpr (std::is_same_v<T, half>) { a = __hmin(a, b); } 
        else { a = min(a, b); }
    };

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
    sdata[tid] = max_val<T>();

    for(int i = tid; i < in_idx.shape[reduce_dim]; i += blockSize) {
        in_coords[reduce_dim] = i;
        minimum(sdata[tid], in_data[in_idx.to_flat(in_coords)]);
    }
    __syncthreads();

    if (blockSize >= 512) { 
        if(tid < 256) minimum(sdata[tid], sdata[tid + 256]); 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if(tid < 128) minimum(sdata[tid], sdata[tid + 128]); 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if(tid < 64)  minimum(sdata[tid], sdata[tid + 64]);  
        __syncthreads(); 
    }
    if (tid < 32) reduce_min_warp<T, blockSize>(sdata, tid);
    if (tid == 0) out_data[blockIdx.x] = sdata[0];
}

template<typename T>
void launch_reduce_min(T* in_data, T* out_data, size_t n_elements) {
    int threads = (n_elements < BLOCK_SIZE_1D * 2) ? nextPow2((n_elements + 1) / 2) : BLOCK_SIZE_1D;
    int blocks = (n_elements + (threads * 2 - 1)) / (threads * 2);
    int smemSize = threads * sizeof(T);

    switch (threads) {
        case 512: reduce_min_kernel<T, 512><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 256: reduce_min_kernel<T, 256><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 128: reduce_min_kernel<T, 128><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 64:  reduce_min_kernel<T,  64><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 32:  reduce_min_kernel<T,  32><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 16:  reduce_min_kernel<T,  16><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 8:   reduce_min_kernel<T,   8><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 4:   reduce_min_kernel<T,   4><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 2:   reduce_min_kernel<T,   2><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 1:   reduce_min_kernel<T,   1><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
    }

    if (blocks > 1) { launch_reduce_min<T>(out_data, out_data, blocks); }
}

template void launch_reduce_min<float>(float*, float*, size_t);
template void launch_reduce_min<half>(half*, half*, size_t);
template void launch_reduce_min<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_reduce_min<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
void launch_reduce_min_dim(
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
        case 512: reduce_min_dim_kernel<T, 512><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 256: reduce_min_dim_kernel<T, 256><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 128: reduce_min_dim_kernel<T, 128><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 64:  reduce_min_dim_kernel<T,  64><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 32:  reduce_min_dim_kernel<T,  32><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 16:  reduce_min_dim_kernel<T,  16><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 8:   reduce_min_dim_kernel<T,   8><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 4:   reduce_min_dim_kernel<T,   4><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 2:   reduce_min_dim_kernel<T,   2><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 1:   reduce_min_dim_kernel<T,   1><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
    }
}

template void launch_reduce_min_dim<float>(float*, float*, TensorIndex, TensorIndex, int);
template void launch_reduce_min_dim<half>(half*, half*, TensorIndex, TensorIndex, int);
template void launch_reduce_min_dim<uint8_t>(uint8_t*, uint8_t*, TensorIndex, TensorIndex, int);
template void launch_reduce_min_dim<int32_t>(int32_t*, int32_t*, TensorIndex, TensorIndex, int);

/* SUM */

template<typename T, unsigned int blockSize>
__device__ void reduce_sum_warp(volatile T* sdata, int tid) {
    auto add = [](volatile T& a, volatile T b) {
        if constexpr (std::is_same_v<T, half>) { a = __hadd(a, b); } 
        else { a += b; }
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

template<typename T, unsigned int blockSize>
__global__ void reduce_sum_dim_kernel(
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

template<typename T>
void launch_reduce_sum_dim(
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
        case 512: reduce_sum_dim_kernel<T, 512><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 256: reduce_sum_dim_kernel<T, 256><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 128: reduce_sum_dim_kernel<T, 128><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 64:  reduce_sum_dim_kernel<T,  64><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 32:  reduce_sum_dim_kernel<T,  32><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 16:  reduce_sum_dim_kernel<T,  16><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 8:   reduce_sum_dim_kernel<T,   8><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 4:   reduce_sum_dim_kernel<T,   4><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 2:   reduce_sum_dim_kernel<T,   2><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
        case 1:   reduce_sum_dim_kernel<T,   1><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim); break;
    }
}

template void launch_reduce_sum_dim<float>(float*, float*, TensorIndex, TensorIndex, int);
template void launch_reduce_sum_dim<half>(half*, half*, TensorIndex, TensorIndex, int);
template void launch_reduce_sum_dim<uint8_t>(uint8_t*, uint8_t*, TensorIndex, TensorIndex, int);
template void launch_reduce_sum_dim<int32_t>(int32_t*, int32_t*, TensorIndex, TensorIndex, int);


