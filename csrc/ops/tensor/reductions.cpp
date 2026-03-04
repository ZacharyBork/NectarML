#include <pybind11/numpy.h>
#include "common.h"

/* KERNELS */

template<typename T>
void launch_reduce_sum(
    T* in_data, 
    T* out_data, 
    TensorIndex in_idx, 
    TensorIndex out_idx,
    int reduce_dim
);

namespace nectar {

    uintptr_t reduce_sum(
        uintptr_t in_ptr, 
        std::vector<int> shape,
        int reduce_dim,
        DType dtype
    ) {
        TensorIndex in_idx(shape.data(), shape.size());
        std::vector<int> out_shape;
        for(int i = 0; i < shape.size(); i++)
            if(i != reduce_dim) out_shape.push_back(shape[i]);
        
        TensorIndex out_idx(out_shape.data(), out_shape.size());

        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, out_idx.n_elements * sizeof(T));
            launch_reduce_sum<T>(
                reinterpret_cast<T*>(in_ptr), d_out,
                in_idx, out_idx, reduce_dim);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

}

