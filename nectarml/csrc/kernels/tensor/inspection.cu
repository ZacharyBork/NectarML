#include "kernels/common.h"
#include "common/functions.h"
#include "ops/policies/inspection.h"

template<unsigned int blockSize, class Op>
__device__ void combine_warp(volatile bool* sdata, int tid) {
    if (blockSize >= 64) Op::combine(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) Op::combine(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) Op::combine(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8)  Op::combine(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4)  Op::combine(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2)  Op::combine(sdata[tid], sdata[tid + 1]);
}

template<typename T, unsigned int blockSize, template<typename> class Pred, class Op>
__global__ void inspection_kernel(
    T* in_data, 
    bool* out_data, 
    size_t n_elements
) {
    extern __shared__ unsigned char shared_raw[];
    bool* sdata = reinterpret_cast<bool*>(shared_raw);
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = Op::identity();
        
    while (idx < n_elements) {
        Op::combine(sdata[tid], Pred<T>::inspect(in_data[idx]));
        if (idx + blockSize < n_elements)
            Op::combine(sdata[tid], Pred<T>::inspect(in_data[idx + blockSize]));
        idx += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if(tid < 256) Op::combine(sdata[tid], sdata[tid + 256]); __syncthreads(); }
    if (blockSize >= 256) { if(tid < 128) Op::combine(sdata[tid], sdata[tid + 128]); __syncthreads(); }
    if (blockSize >= 128) { if(tid < 64)  Op::combine(sdata[tid], sdata[tid + 64]);  __syncthreads(); }
    if (tid < 32) combine_warp<blockSize, Op>(sdata, tid);
    if (tid == 0) out_data[blockIdx.x] = sdata[0];
}

template<class Op>
void launch_inspection_reduce(
    bool* in_data,
    bool* out_data,
    size_t n_elements
) {
    int threads = (n_elements < BLOCK_SIZE_1D * 2) ? nextPow2((n_elements + 1) / 2) : BLOCK_SIZE_1D;
    int blocks = (n_elements + (threads * 2 - 1)) / (threads * 2);
    int smemSize = threads * sizeof(bool);

    switch (threads) {
        case 512: inspection_kernel<bool, 512, IdentityPred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 256: inspection_kernel<bool, 256, IdentityPred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 128: inspection_kernel<bool, 128, IdentityPred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 64:  inspection_kernel<bool,  64, IdentityPred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 32:  inspection_kernel<bool,  32, IdentityPred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 16:  inspection_kernel<bool,  16, IdentityPred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 8:   inspection_kernel<bool,   8, IdentityPred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 4:   inspection_kernel<bool,   4, IdentityPred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 2:   inspection_kernel<bool,   2, IdentityPred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 1:   inspection_kernel<bool,   1, IdentityPred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
    }
    if (blocks > 1) launch_inspection_reduce<Op>(out_data, out_data, blocks);
}

template<typename T, template<typename> class Pred, class Op>
void launch_inspection(
    T* in_data, 
    bool* out_data, 
    size_t n_elements
) {
    int threads = (n_elements < BLOCK_SIZE_1D * 2) ? nextPow2((n_elements + 1) / 2) : BLOCK_SIZE_1D;
    int blocks = (n_elements + (threads * 2 - 1)) / (threads * 2);
    int smemSize = threads * sizeof(bool);

    switch (threads) {
        case 512: inspection_kernel<T, 512, Pred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 256: inspection_kernel<T, 256, Pred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 128: inspection_kernel<T, 128, Pred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 64:  inspection_kernel<T,  64, Pred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 32:  inspection_kernel<T,  32, Pred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 16:  inspection_kernel<T,  16, Pred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 8:   inspection_kernel<T,   8, Pred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 4:   inspection_kernel<T,   4, Pred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 2:   inspection_kernel<T,   2, Pred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
        case 1:   inspection_kernel<T,   1, Pred, Op><<< blocks, threads, smemSize >>>(in_data, out_data, n_elements); break;
    }
    if (blocks > 1) launch_inspection_reduce<Op>(out_data, out_data, blocks);
}

template void launch_inspection<float, IsInfPred, AllOp>(float*, bool*, size_t);
template void launch_inspection<float, IsInfPred, AnyOp>(float*, bool*, size_t);
template void launch_inspection<float, IsFinitePred, AllOp>(float*, bool*, size_t);
template void launch_inspection<float, IsFinitePred, AnyOp>(float*, bool*, size_t);
template void launch_inspection<float, IsNanPred, AllOp>(float*, bool*, size_t);
template void launch_inspection<float, IsNanPred, AnyOp>(float*, bool*, size_t);

template void launch_inspection<half, IsInfPred, AllOp>(half*, bool*, size_t);
template void launch_inspection<half, IsInfPred, AnyOp>(half*, bool*, size_t);
template void launch_inspection<half, IsFinitePred, AllOp>(half*, bool*, size_t);
template void launch_inspection<half, IsFinitePred, AnyOp>(half*, bool*, size_t);
template void launch_inspection<half, IsNanPred, AllOp>(half*, bool*, size_t);
template void launch_inspection<half, IsNanPred, AnyOp>(half*, bool*, size_t);

template void launch_inspection<uint8_t, IsInfPred, AllOp>(uint8_t*, bool*, size_t);
template void launch_inspection<uint8_t, IsInfPred, AnyOp>(uint8_t*, bool*, size_t);
template void launch_inspection<uint8_t, IsFinitePred, AllOp>(uint8_t*, bool*, size_t);
template void launch_inspection<uint8_t, IsFinitePred, AnyOp>(uint8_t*, bool*, size_t);
template void launch_inspection<uint8_t, IsNanPred, AllOp>(uint8_t*, bool*, size_t);
template void launch_inspection<uint8_t, IsNanPred, AnyOp>(uint8_t*, bool*, size_t);

template void launch_inspection<int32_t, IsInfPred, AllOp>(int32_t*, bool*, size_t);
template void launch_inspection<int32_t, IsInfPred, AnyOp>(int32_t*, bool*, size_t);
template void launch_inspection<int32_t, IsFinitePred, AllOp>(int32_t*, bool*, size_t);
template void launch_inspection<int32_t, IsFinitePred, AnyOp>(int32_t*, bool*, size_t);
template void launch_inspection<int32_t, IsNanPred, AllOp>(int32_t*, bool*, size_t);
template void launch_inspection<int32_t, IsNanPred, AnyOp>(int32_t*, bool*, size_t);