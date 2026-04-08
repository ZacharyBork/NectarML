#include "common.h"

/* KERNELS */

template<typename T>
void launch_permute(
    T* in_data,
    T* out_data,
    TensorIndex in_idx,
    TensorIndex out_idx,
    Permutation inv_perm,
    size_t total_elements
);

template<typename T>
void launch_expand(
    T* in_data,
    T* out_data,
    TensorIndex in_idx,
    TensorIndex out_idx,
    size_t total_elements
);

template<typename T>
void launch_flip(
    T* input, T* output,
    int outer, int dim_size, int inner
);

namespace nectar {

    uintptr_t permute(
        uintptr_t in_ptr,
        std::vector<int> shape,
        std::vector<int> dims,
        DType dtype
    ) {
        TensorIndex in_idx(shape.data(), shape.size());

        std::vector<int> out_shape(shape.size());
        for (int i = 0; i < dims.size(); i++)
            out_shape[i] = shape[dims[i]];

        TensorIndex out_idx(out_shape.data(), out_shape.size());
        
        Permutation perm(dims.data(), dims.size());
        Permutation inv_perm = perm.inverse();
        
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, out_idx.n_elements * sizeof(T));
            launch_permute<T>(
                reinterpret_cast<T*>(in_ptr), d_out, 
                in_idx,out_idx, inv_perm, out_idx.n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t expand(
        uintptr_t in_ptr,
        std::vector<int> in_shape,
        std::vector<int> target_shape,
        DType dtype
    ) {
        TensorIndex in_idx(in_shape.data(), in_shape.size());
        TensorIndex out_idx(target_shape.data(), target_shape.size());

        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, out_idx.n_elements * sizeof(T));
            launch_expand<T>(
                reinterpret_cast<T*>(in_ptr), d_out,
                in_idx, out_idx, out_idx.n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t flip(
        uintptr_t in_ptr,
        int total,
        int dim_size,
        int outer,
        int inner,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_input  = reinterpret_cast<T*>(in_ptr);
            T* d_output;
            cudaMalloc(&d_output, total * sizeof(T));
            launch_flip<T>(d_input, d_output, outer, dim_size, inner);
            return reinterpret_cast<uintptr_t>(d_output);
        });
    }

}
