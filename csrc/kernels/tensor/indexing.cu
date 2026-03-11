#include "common.h"

/* GATHER */

template<typename T>
__global__ void gather_kernel(
    T* in_data,
    TensorIndex in_idx,
    int32_t* indices,
    TensorIndex indices_idx,
    T* out_data,
    int dim
) {
    int out_flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_flat >= indices_idx.n_elements) return;

    int coords[MAX_DIMS];
    indices_idx.to_index(out_flat, coords);

    int idx_val = indices[out_flat];

    int in_coords[MAX_DIMS];
    for (int i = 0; i < in_idx.ndim; i++)
        in_coords[i] = coords[i];
    in_coords[dim] = idx_val;

    out_data[out_flat] = in_data[in_idx.to_flat(in_coords)];
}

template<typename T>
void launch_gather(
    T* in_data,
    TensorIndex in_idx,
    int32_t* indices,
    TensorIndex indices_idx,
    T* out_data,
    int dim
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (indices_idx.n_elements + threads - 1) / threads;
    gather_kernel<T><<<blocks, threads>>>(
        in_data, in_idx, indices, indices_idx, out_data, dim);
}

template void launch_gather<float>(float*, TensorIndex, int32_t*, TensorIndex, float*, int);
template void launch_gather<half>(half*, TensorIndex, int32_t*, TensorIndex, half*, int);
template void launch_gather<uint8_t>(uint8_t*, TensorIndex, int32_t*, TensorIndex, uint8_t*, int);
template void launch_gather<int32_t>(int32_t*, TensorIndex, int32_t*, TensorIndex, int32_t*, int);

/* SCATTER */

template<typename T>
__global__ void scatter_kernel(
    T* src_data,
    TensorIndex src_idx,
    int32_t* indices,
    TensorIndex indices_idx,
    T* out_data,
    int dim
) {
    int out_flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_flat >= indices_idx.n_elements) return;

    int coords[MAX_DIMS];
    indices_idx.to_index(out_flat, coords);

    int idx_val = indices[out_flat];

    int out_coords[MAX_DIMS];
    for (int i = 0; i < indices_idx.ndim; i++)
        out_coords[i] = coords[i];
    out_coords[dim] = idx_val;

    out_data[indices_idx.to_flat(out_coords)] = src_data[out_flat];
}

template<typename T>
void launch_scatter(
    T* src_data,
    TensorIndex src_idx,
    int32_t* indices,
    TensorIndex indices_idx,
    T* out_data,
    int dim
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (indices_idx.n_elements + threads - 1) / threads;
    scatter_kernel<T><<<blocks, threads>>>(
        src_data, src_idx, indices, indices_idx, out_data, dim);
}

template void launch_scatter<float>(float*, TensorIndex, int32_t*, TensorIndex, float*, int);
template void launch_scatter<half>(half*, TensorIndex, int32_t*, TensorIndex, half*, int);
template void launch_scatter<uint8_t>(uint8_t*, TensorIndex, int32_t*, TensorIndex, uint8_t*, int);
template void launch_scatter<int32_t>(int32_t*, TensorIndex, int32_t*, TensorIndex, int32_t*, int);



