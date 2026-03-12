#include "common.h"


template<typename T>
__global__ void concatenate_kernel(
    ConcatInputs inputs, 
    TensorIndex out_idx, 
    T* out, 
    int dim, 
    size_t n_elements
) {
    int out_flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_flat >= n_elements) return;

    int out_coords[MAX_DIMS];
    out_idx.to_index(out_flat, out_coords);

    int input_idx = 0;
    while (input_idx < inputs.n_inputs-1 && out_coords[dim] >= inputs.offsets[input_idx+1]) input_idx++;

    int in_coords[MAX_DIMS];
    in_coords[dim] = out_coords[dim] - inputs.offsets[input_idx];

    TensorIndex in_idx = inputs.indices[input_idx];
    T* in_data = reinterpret_cast<T*>(inputs.ptrs[input_idx]);
    out[out_flat] = in_data[in_idx.to_flat(in_coords)];
}

template<typename T>
void launch_concatenate(ConcatInputs inputs, TensorIndex out_idx, T* out, int dim, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    concatenate_kernel<T><<<grid, block>>>(inputs, out_idx, out, dim, n_elements);
}

template void launch_concatenate<float>(ConcatInputs, TensorIndex, float*, int, size_t);
template void launch_concatenate<half>(ConcatInputs, TensorIndex, half*, int, size_t);
template void launch_concatenate<uint8_t>(ConcatInputs, TensorIndex, uint8_t*, int, size_t);
template void launch_concatenate<int32_t>(ConcatInputs, TensorIndex, int32_t*, int, size_t);




