#include "common.h"
#include <iostream>
#include <string>

namespace py = pybind11;

/* KERNELS */

template<typename T>
void launch_gather(
    T* in_data,
    TensorIndex in_idx,
    int32_t* indices,
    TensorIndex indices_idx,
    T* out_data,
    int dim
); 

template<typename T>
void launch_scatter(
    T* src_data,
    TensorIndex src_idx,
    int32_t* indices,
    TensorIndex indices_idx,
    T* out_data,
    int dim
); 


/*
- scatter
- scatter_add
- where
- masked_fill
- index_select
*/

namespace nectar {

    uintptr_t gather(
        uintptr_t data_ptr,
        std::vector<int> data_shape,
        uintptr_t indices_ptr,
        std::vector<int> indices_shape,
        int dim,
        DType dtype
    ) {
        TensorIndex in_idx(data_shape.data(), data_shape.size());
        TensorIndex indices_idx(indices_shape.data(), indices_shape.size());

        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, indices_idx.n_elements * sizeof(T));
            launch_gather<T>(
                reinterpret_cast<T*>(data_ptr), in_idx,
                reinterpret_cast<int32_t*>(indices_ptr), indices_idx, 
                d_out, dim);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t scatter(
        uintptr_t input_ptr,
        std::vector<int> input_shape,
        uintptr_t source_ptr,
        std::vector<int> source_shape,
        uintptr_t indices_ptr,
        std::vector<int> indices_shape,
        int dim,
        DType dtype
    ) {
        TensorIndex in_idx(input_shape.data(), input_shape.size());
        TensorIndex source_idx(source_shape.data(), source_shape.size());
        TensorIndex indices_idx(indices_shape.data(), indices_shape.size());

        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            size_t memsize = in_idx.n_elements * sizeof(T);
            cudaMalloc(&d_out, memsize);
            cudaMemcpy(d_out, reinterpret_cast<void*>(input_ptr), memsize, cudaMemcpyDeviceToDevice);
            launch_scatter<T>(
                reinterpret_cast<T*>(source_ptr), source_idx,
                reinterpret_cast<int32_t*>(indices_ptr), indices_idx, 
                d_out, dim);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

}

