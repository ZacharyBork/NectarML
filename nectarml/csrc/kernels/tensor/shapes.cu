#include "kernels/common.h"
#include "common/data_structures.h"

/* COMPARISON OPERATORS */

template<typename T>
__global__ void permute_kernel(
    T* in_data,
    T* out_data,
    TensorIndex in_idx,
    TensorIndex out_idx,
    Permutation inv_perm,
    size_t total_elements
) {
    int out_flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_flat >= total_elements) return;

    int out_coords[MAX_DIMS];
    out_idx.to_index(out_flat, out_coords);

    int in_coords[MAX_DIMS];
    for (int i = 0; i < out_idx.ndim; i++)
        in_coords[i] = out_coords[inv_perm.dims[i]];

    out_data[out_flat] = in_data[in_idx.to_flat(in_coords)];
}

template<typename T>
void launch_permute(
    T* in_data,
    T* out_data,
    TensorIndex in_idx,
    TensorIndex out_idx,
    Permutation inv_perm,
    size_t total_elements
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (total_elements + threads - 1) / threads;
    permute_kernel<T><<<blocks, threads>>>(in_data, out_data, in_idx, out_idx, inv_perm, total_elements);
}

template void launch_permute<float>(float*, float*, TensorIndex, TensorIndex, Permutation, size_t);
template void launch_permute<half>(half*, half*, TensorIndex, TensorIndex, Permutation, size_t);
template void launch_permute<uint8_t>(uint8_t*, uint8_t*, TensorIndex, TensorIndex, Permutation, size_t);
template void launch_permute<int32_t>(int32_t*, int32_t*, TensorIndex, TensorIndex, Permutation, size_t);

template<typename T>
__global__ void expand_kernel(
    T* in_data,
    T* out_data,
    TensorIndex in_idx,
    TensorIndex out_idx,
    size_t total_elements
) {
    int out_flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_flat >= total_elements) return;

    int out_coords[MAX_DIMS];
    out_idx.to_index(out_flat, out_coords);

    int in_coords[MAX_DIMS];
    for (int i = 0; i < out_idx.ndim; i++)
        in_coords[i] = (in_idx.shape[i] == 1) ? 0 : out_coords[i];

    out_data[out_flat] = in_data[in_idx.to_flat(in_coords)];
}

template<typename T>
void launch_expand(
    T* in_data,
    T* out_data,
    TensorIndex in_idx,
    TensorIndex out_idx,
    size_t total_elements
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (total_elements + threads - 1) / threads;
    expand_kernel<T><<<blocks, threads>>>(in_data, out_data, in_idx, out_idx, total_elements);
}

template void launch_expand<float>(float*, float*, TensorIndex, TensorIndex, size_t);
template void launch_expand<half>(half*, half*, TensorIndex, TensorIndex, size_t);
template void launch_expand<uint8_t>(uint8_t*, uint8_t*, TensorIndex, TensorIndex, size_t);
template void launch_expand<int32_t>(int32_t*, int32_t*, TensorIndex, TensorIndex, size_t);

template<typename T>
__global__ void flip_kernel(
    T* input, T* output,
    int outer, int dim_size, int inner
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * dim_size * inner) return;

    int i  = idx % inner;
    int k  = (idx / inner) % dim_size;
    int o  = idx / (inner * dim_size);

    int src_k   = dim_size - 1 - k;
    int src_idx = o * dim_size * inner + src_k * inner + i;

    output[idx] = input[src_idx];
}

template<typename T>
void launch_flip(
    T* input, T* output,
    int outer, int dim_size, int inner
) {
    int total   = outer * dim_size * inner;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    flip_kernel<T><<<blocks, threads>>>(
        input, output, outer, dim_size, inner);
}

template void launch_flip<float>(float*, float*, int, int, int);
template void launch_flip<half>(half*, half*, int, int, int);
template void launch_flip<int32_t>(int32_t*, int32_t*, int, int, int);
template void launch_flip<uint8_t>(uint8_t*, uint8_t*, int, int, int);

