// REFERENCE: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

#include "common.h"
#include "ops/policies/reductions.h"

template<typename T, unsigned int blockSize, template<typename> class Op>
__device__ void reduce_warp(volatile T* sdata, int tid) {
    if (blockSize >= 64) Op<T>::combine(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) Op<T>::combine(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) Op<T>::combine(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8)  Op<T>::combine(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4)  Op<T>::combine(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2)  Op<T>::combine(sdata[tid], sdata[tid + 1]);
}

template<typename T, unsigned int blockSize, template<typename> class Op>
__global__ void reduce_kernel(
    T* in_data, 
    T* out_data, 
    size_t n_elements,
    float identity
) {
    extern __shared__ unsigned char shared_raw[];
    T* sdata = reinterpret_cast<T*>(shared_raw);

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = isnan(identity) ? Op<T>::identity() : static_cast<T>(identity);

    while (idx < n_elements) {
        Op<T>::combine(sdata[tid], in_data[idx]);
        if (idx + blockSize < n_elements)
            Op<T>::combine(sdata[tid], in_data[idx + blockSize]);
        idx += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if(tid < 256) Op<T>::combine(sdata[tid], sdata[tid + 256]); __syncthreads(); }
    if (blockSize >= 256) { if(tid < 128) Op<T>::combine(sdata[tid], sdata[tid + 128]); __syncthreads(); }
    if (blockSize >= 128) { if(tid < 64)  Op<T>::combine(sdata[tid], sdata[tid + 64]);  __syncthreads(); }
    if (tid < 32) reduce_warp<T, blockSize, Op>(sdata, tid);
    if (tid == 0) out_data[blockIdx.x] = sdata[0];
}

template<typename T, unsigned int blockSize, template<typename> class Op>
__global__ void reduce_dim_kernel(
    T* in_data, 
    T* out_data, 
    TensorIndex in_idx,
    TensorIndex out_idx,
    int reduce_dim,
    float identity
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
    sdata[tid] = isnan(identity) ? Op<T>::identity() : static_cast<T>(identity);

    for(int i = tid; i < in_idx.shape[reduce_dim]; i += blockSize) {
        in_coords[reduce_dim] = i;
        Op<T>::combine(sdata[tid], in_data[in_idx.to_flat(in_coords)]);
    }
    __syncthreads();

    if (blockSize >= 512) { if(tid < 256) Op<T>::combine(sdata[tid], sdata[tid + 256]); __syncthreads(); }
    if (blockSize >= 256) { if(tid < 128) Op<T>::combine(sdata[tid], sdata[tid + 128]); __syncthreads(); }
    if (blockSize >= 128) { if(tid < 64)  Op<T>::combine(sdata[tid], sdata[tid + 64]);  __syncthreads(); }
    if (tid < 32) reduce_warp<T, blockSize, Op>(sdata, tid);
    if (tid == 0) out_data[blockIdx.x] = sdata[0];
}

template<typename T, template<typename> class Op>
void launch_reduce(
    T* in_data, 
    T* out_data, 
    size_t n_elements,
    float identity = std::numeric_limits<float>::quiet_NaN()
) {
    int threads = (n_elements < BLOCK_SIZE_1D * 2) ? nextPow2((n_elements + 1) / 2) : BLOCK_SIZE_1D;
    int blocks = (n_elements + (threads * 2 - 1)) / (threads * 2);
    int smemSize = threads * sizeof(T);

    switch (threads) {
        case 512: reduce_kernel<T, 512, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements, identity); break;
        case 256: reduce_kernel<T, 256, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements, identity); break;
        case 128: reduce_kernel<T, 128, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements, identity); break;
        case 64:  reduce_kernel<T,  64, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements, identity); break;
        case 32:  reduce_kernel<T,  32, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements, identity); break;
        case 16:  reduce_kernel<T,  16, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements, identity); break;
        case 8:   reduce_kernel<T,   8, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements, identity); break;
        case 4:   reduce_kernel<T,   4, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements, identity); break;
        case 2:   reduce_kernel<T,   2, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements, identity); break;
        case 1:   reduce_kernel<T,   1, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements, identity); break;
    }
    if (blocks > 1) launch_reduce<T, Op>(out_data, out_data, blocks, identity);
}

template void launch_reduce<float, SumOp>(float*, float*, size_t, float);
template void launch_reduce<float, MinOp>(float*, float*, size_t, float);
template void launch_reduce<float, MaxOp>(float*, float*, size_t, float);
template void launch_reduce<float, ProdOp>(float*, float*, size_t, float);

template void launch_reduce<half, SumOp>(half*, half*, size_t, float);
template void launch_reduce<half, MinOp>(half*, half*, size_t, float);
template void launch_reduce<half, MaxOp>(half*, half*, size_t, float);
template void launch_reduce<half, ProdOp>(half*, half*, size_t, float);

template void launch_reduce<uint8_t, SumOp>(uint8_t*, uint8_t*, size_t, float);
template void launch_reduce<uint8_t, MinOp>(uint8_t*, uint8_t*, size_t, float);
template void launch_reduce<uint8_t, MaxOp>(uint8_t*, uint8_t*, size_t, float);
template void launch_reduce<uint8_t, ProdOp>(uint8_t*, uint8_t*, size_t, float);

template void launch_reduce<int32_t, SumOp>(int32_t*, int32_t*, size_t, float);
template void launch_reduce<int32_t, MinOp>(int32_t*, int32_t*, size_t, float);
template void launch_reduce<int32_t, MaxOp>(int32_t*, int32_t*, size_t, float);
template void launch_reduce<int32_t, ProdOp>(int32_t*, int32_t*, size_t, float);

template<typename T, template<typename> class Op>
void launch_reduce_dim(
    T* in_data, 
    T* out_data, 
    TensorIndex in_idx, 
    TensorIndex out_idx,
    int reduce_dim,
    float identity = std::numeric_limits<float>::quiet_NaN()
) {
    int blocks = out_idx.n_elements;
    int threads = min(nextPow2(in_idx.shape[reduce_dim]), BLOCK_SIZE_1D);
    int smemSize = threads * sizeof(T);

    switch (threads) {
        case 512: reduce_dim_kernel<T, 512, Op><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim, identity); break;
        case 256: reduce_dim_kernel<T, 256, Op><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim, identity); break;
        case 128: reduce_dim_kernel<T, 128, Op><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim, identity); break;
        case 64:  reduce_dim_kernel<T,  64, Op><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim, identity); break;
        case 32:  reduce_dim_kernel<T,  32, Op><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim, identity); break;
        case 16:  reduce_dim_kernel<T,  16, Op><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim, identity); break;
        case 8:   reduce_dim_kernel<T,   8, Op><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim, identity); break;
        case 4:   reduce_dim_kernel<T,   4, Op><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim, identity); break;
        case 2:   reduce_dim_kernel<T,   2, Op><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim, identity); break;
        case 1:   reduce_dim_kernel<T,   1, Op><<< blocks, threads, smemSize >>>(in_data, out_data, in_idx, out_idx, reduce_dim, identity); break;
    }
}

template void launch_reduce_dim<float, SumOp>(float*, float*, TensorIndex, TensorIndex, int, float);
template void launch_reduce_dim<float, MinOp>(float*, float*, TensorIndex, TensorIndex, int, float);
template void launch_reduce_dim<float, MaxOp>(float*, float*, TensorIndex, TensorIndex, int, float);
template void launch_reduce_dim<float, ProdOp>(float*, float*, TensorIndex, TensorIndex, int, float);

template void launch_reduce_dim<half, SumOp>(half*, half*, TensorIndex, TensorIndex, int, float);
template void launch_reduce_dim<half, MinOp>(half*, half*, TensorIndex, TensorIndex, int, float);
template void launch_reduce_dim<half, MaxOp>(half*, half*, TensorIndex, TensorIndex, int, float);
template void launch_reduce_dim<half, ProdOp>(half*, half*, TensorIndex, TensorIndex, int, float);

template void launch_reduce_dim<uint8_t, SumOp>(uint8_t*, uint8_t*, TensorIndex, TensorIndex, int, float);
template void launch_reduce_dim<uint8_t, MinOp>(uint8_t*, uint8_t*, TensorIndex, TensorIndex, int, float);
template void launch_reduce_dim<uint8_t, MaxOp>(uint8_t*, uint8_t*, TensorIndex, TensorIndex, int, float);
template void launch_reduce_dim<uint8_t, ProdOp>(uint8_t*, uint8_t*, TensorIndex, TensorIndex, int, float);

template void launch_reduce_dim<int32_t, SumOp>(int32_t*, int32_t*, TensorIndex, TensorIndex, int, float);
template void launch_reduce_dim<int32_t, MinOp>(int32_t*, int32_t*, TensorIndex, TensorIndex, int, float);
template void launch_reduce_dim<int32_t, MaxOp>(int32_t*, int32_t*, TensorIndex, TensorIndex, int, float);
template void launch_reduce_dim<int32_t, ProdOp>(int32_t*, int32_t*, TensorIndex, TensorIndex, int, float);

