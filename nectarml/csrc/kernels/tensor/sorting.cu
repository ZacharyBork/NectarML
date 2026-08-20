#include "kernels/common.h"

template<typename T>
__global__ void sort_slices_kernel(
    T* values, int32_t* indices,
    int outer, int dim_size, int inner
) {
    int o = blockIdx.x;
    int i = blockIdx.y;
    if (o >= outer || i >= inner) return;

    extern __shared__ char shared[];
    T*       s_vals = reinterpret_cast<T*>(shared);
    int32_t* s_idx  = reinterpret_cast<int32_t*>(s_vals + dim_size);

    for (int k = threadIdx.x; k < dim_size; k += blockDim.x) {
        int global_idx = o * dim_size * inner + k * inner + i;
        s_vals[k] = values[global_idx];
        s_idx[k]  = k;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int a = 1; a < dim_size; a++) {
            T       key_val = s_vals[a];
            int32_t key_idx = s_idx[a];
            int b = a - 1;
            while (b >= 0 && s_vals[b] > key_val) {
                s_vals[b + 1] = s_vals[b];
                s_idx[b + 1]  = s_idx[b];
                b--;
            }
            s_vals[b + 1] = key_val;
            s_idx[b + 1]  = key_idx;
        }
    }
    __syncthreads();

    for (int k = threadIdx.x; k < dim_size; k += blockDim.x) {
        int global_idx = o * dim_size * inner + k * inner + i;
        values[global_idx]  = s_vals[k];
        indices[global_idx] = s_idx[k];
    }
}

template<typename T>
__global__ void reverse_slices_kernel(
    T* values, int32_t* indices,
    int outer, int dim_size, int inner
) {
    int o = blockIdx.x;
    int i = blockIdx.y;
    if (o >= outer || i >= inner) return;

    for (int k = threadIdx.x; k < dim_size / 2; k += blockDim.x) {
        int lo = o * dim_size * inner + k * inner + i;
        int hi = o * dim_size * inner + (dim_size - 1 - k) * inner + i;

        T tmp_val = values[lo];
        int32_t tmp_idx = indices[lo];

        values[lo]  = values[hi];
        indices[lo] = indices[hi];

        values[hi]  = tmp_val;
        indices[hi] = tmp_idx;
    }
}

template<typename T>
void sort_tensor(
    T* d_values, int32_t* d_indices,
    int dim_size, int outer, int inner,
    bool descending
) {
    dim3 blocks(outer, inner);
    int  threads = min(dim_size, 256);
    size_t shared_bytes = dim_size * (sizeof(T) + sizeof(int32_t));

    sort_slices_kernel<T><<<blocks, threads, shared_bytes>>>(
        d_values, d_indices, outer, dim_size, inner);

    if (descending) {
        int threads = min(dim_size / 2, 256);
        if (threads > 0) {
            reverse_slices_kernel<T><<<blocks, threads>>>(
                d_values, d_indices, outer, dim_size, inner);
        }
    }
}

template void sort_tensor<float>(float*, int32_t*, int, int, int, bool);
template void sort_tensor<half>(half*, int32_t*, int, int, int, bool);
template void sort_tensor<uint8_t>(uint8_t*, int32_t*, int, int, int, bool);
template void sort_tensor<int32_t>(int32_t*, int32_t*, int, int, int, bool);
