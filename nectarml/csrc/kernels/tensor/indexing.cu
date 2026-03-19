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

template<typename T>
__global__ void scatter_add_kernel(
    TensorIndex in_idx,
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

    atomicAdd(&out_data[in_idx.to_flat(out_coords)], src_data[out_flat]);
}

template<typename T>
void launch_scatter_add(
    TensorIndex in_idx,
    T* src_data,
    TensorIndex src_idx,
    int32_t* indices,
    TensorIndex indices_idx,
    T* out_data,
    int dim
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (indices_idx.n_elements + threads - 1) / threads;
    scatter_add_kernel<T><<<blocks, threads>>>(
        in_idx, src_data, src_idx, indices, indices_idx, out_data, dim);
}

template void launch_scatter_add<float>(TensorIndex, float*, TensorIndex, int32_t*, TensorIndex, float*, int);
template void launch_scatter_add<half>(TensorIndex, half*, TensorIndex, int32_t*, TensorIndex, half*, int);
template void launch_scatter_add<int32_t>(TensorIndex, int32_t*, TensorIndex, int32_t*, TensorIndex, int32_t*, int);

/* SLICE */

template<typename T>
__global__ void slice_kernel(
    T* in_data,
    T* out_data,
    TensorIndex in_idx,
    TensorIndex out_idx,
    SliceIndex slice_idx
) {
    int out_flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_flat >= out_idx.n_elements) return;

    int out_coords[MAX_DIMS];
    out_idx.to_index(out_flat, out_coords);

    int in_coords[MAX_DIMS];
    for (int i = 0; i < slice_idx.ndim; i++)
        in_coords[i] = slice_idx.start[i] + out_coords[i] * slice_idx.step[i];

    out_data[out_flat] = in_data[in_idx.to_flat(in_coords)];
}

template<typename T>
void launch_slice(
    T* in_data,
    T* out_data,
    TensorIndex in_idx,
    TensorIndex out_idx,
    SliceIndex slice_index
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (out_idx.n_elements + threads - 1) / threads;
    slice_kernel<T><<<blocks, threads>>>(
        in_data, out_data, in_idx, out_idx, slice_index);
}

template void launch_slice<float>(float*, float*, TensorIndex, TensorIndex, SliceIndex);
template void launch_slice<half>(half*, half*, TensorIndex, TensorIndex, SliceIndex);
template void launch_slice<uint8_t>(uint8_t*, uint8_t*, TensorIndex, TensorIndex, SliceIndex);
template void launch_slice<int32_t>(int32_t*, int32_t*, TensorIndex, TensorIndex, SliceIndex);

/* INDEX PUT */

template<typename T>
__global__ void index_put_kernel(
    T* src_data,
    T* out_data,
    TensorIndex src_idx,
    TensorIndex out_idx,
    SliceIndex slice_idx
) {
    int src_flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (src_flat >= src_idx.n_elements) return;

    int src_coords[MAX_DIMS];
    src_idx.to_index(src_flat, src_coords);

    int out_coords[MAX_DIMS];
    for (int i = 0; i < slice_idx.ndim; i++)
        out_coords[i] = slice_idx.start[i] + src_coords[i] * slice_idx.step[i];

    out_data[out_idx.to_flat(out_coords)] = src_data[src_flat];
}

template<typename T>
void launch_index_put(
    T* src_data,
    T* out_data,
    TensorIndex src_idx,
    TensorIndex out_idx,
    SliceIndex slice_index
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (out_idx.n_elements + threads - 1) / threads;
    index_put_kernel<T><<<blocks, threads>>>(
        src_data, out_data, src_idx, out_idx, slice_index);
}

template void launch_index_put<float>(float*, float*, TensorIndex, TensorIndex, SliceIndex);
template void launch_index_put<half>(half*, half*, TensorIndex, TensorIndex, SliceIndex);
template void launch_index_put<uint8_t>(uint8_t*, uint8_t*, TensorIndex, TensorIndex, SliceIndex);
template void launch_index_put<int32_t>(int32_t*, int32_t*, TensorIndex, TensorIndex, SliceIndex);



