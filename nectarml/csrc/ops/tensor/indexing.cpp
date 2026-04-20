#include "ops/common.h"
#include "common/dtype.h"
#include "common/data_structures.h"
#include "allocator_pool/allocator_pool.h"

namespace py = pybind11;

/* KERNELS */

template<typename T>
void launch_gather(
    T* in_data, TensorIndex in_idx,
    int32_t* indices, TensorIndex indices_idx,
    T* out_data, int dim
); 

template<typename T>
void launch_scatter(
    T* src_data, TensorIndex src_idx,
    int32_t* indices, TensorIndex indices_idx,
    T* out_data, int dim
); 

template<typename T>
void launch_scatter_add(
    TensorIndex in_idx,
    T* src_data, TensorIndex src_idx,
    int32_t* indices, TensorIndex indices_idx,
    T* out_data, int dim
); 

template<typename T>
void launch_slice(
    T* in_data,
    T* out_data,
    TensorIndex in_idx,
    TensorIndex out_idx,
    SliceIndex slice_index
);

template<typename T>
void launch_index_put(
    T* in_data,
    T* out_data,
    TensorIndex in_idx,
    TensorIndex out_idx,
    SliceIndex slice_index
);

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
            T* d_out = static_cast<T*>(g_pool.alloc(indices_idx.n_elements * sizeof(T)));
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
            size_t memsize = in_idx.n_elements * sizeof(T);
            T* d_out = static_cast<T*>(g_pool.alloc(memsize));
            cudaMemcpy(d_out, reinterpret_cast<void*>(input_ptr), memsize, cudaMemcpyDeviceToDevice);
            launch_scatter<T>(
                reinterpret_cast<T*>(source_ptr), source_idx,
                reinterpret_cast<int32_t*>(indices_ptr), indices_idx, 
                d_out, dim);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t scatter_add(
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

        if (dtype == DType::Float32) {
            size_t memsize = in_idx.n_elements * sizeof(float);
            float* d_out = static_cast<float*>(g_pool.alloc(memsize));
            cudaMemcpy(d_out, reinterpret_cast<void*>(input_ptr), memsize, cudaMemcpyDeviceToDevice);
            launch_scatter_add<float>(
                in_idx,
                reinterpret_cast<float*>(source_ptr), source_idx,
                reinterpret_cast<int32_t*>(indices_ptr), indices_idx, 
                d_out, dim);
            return reinterpret_cast<uintptr_t>(d_out);
        }
        else if (dtype == DType::Float16) {
            size_t memsize = in_idx.n_elements * sizeof(half);
            half* d_out = static_cast<half*>(g_pool.alloc(memsize));
            cudaMemcpy(d_out, reinterpret_cast<void*>(input_ptr), memsize, cudaMemcpyDeviceToDevice);
            launch_scatter_add<half>(
                in_idx,
                reinterpret_cast<half*>(source_ptr), source_idx,
                reinterpret_cast<int32_t*>(indices_ptr), indices_idx, 
                d_out, dim);
            return reinterpret_cast<uintptr_t>(d_out);
        }
        else if (dtype == DType::Int32) {
            size_t memsize = in_idx.n_elements * sizeof(int32_t);
            int32_t* d_out = static_cast<int32_t*>(g_pool.alloc(memsize));
            cudaMemcpy(d_out, reinterpret_cast<void*>(input_ptr), memsize, cudaMemcpyDeviceToDevice);
            launch_scatter_add<int32_t>(
                in_idx,
                reinterpret_cast<int32_t*>(source_ptr), source_idx,
                reinterpret_cast<int32_t*>(indices_ptr), indices_idx, 
                d_out, dim);
            return reinterpret_cast<uintptr_t>(d_out);
        }
        else {
            throw std::runtime_error(
                "scatter_add does not support this dtype. "
                "uint8 tensors are automatically promoted "
                "to int32 on the Python side.");
        }
    }

    uintptr_t slice(
        uintptr_t input_ptr,
        std::vector<int> input_shape,
        std::vector<int> start,
        std::vector<int> count,
        std::vector<int> step,
        DType dtype
    ) {
        TensorIndex in_idx(input_shape.data(), input_shape.size());
        TensorIndex out_idx(count.data(), count.size());
        SliceIndex slice_idx(start.data(), step.data(), in_idx.ndim);
        
        DISPATCH_DTYPE(dtype, T, {
            T* d_out = static_cast<T*>(g_pool.alloc(out_idx.n_elements * sizeof(T)));
            launch_slice<T>(
                reinterpret_cast<T*>(input_ptr), d_out,
                in_idx, out_idx, slice_idx);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t index_put(
        uintptr_t input_ptr,
        std::vector<int> input_shape,
        uintptr_t src_ptr,
        std::vector<int> start,
        std::vector<int> count,
        std::vector<int> step,
        DType dtype
    ) {
        TensorIndex in_idx(input_shape.data(), input_shape.size());
        TensorIndex src_idx(count.data(), count.size());
        SliceIndex slice_idx(start.data(), step.data(), in_idx.ndim);

        DISPATCH_DTYPE(dtype, T, {
            size_t memsize = in_idx.n_elements * sizeof(T);
            T* d_out = static_cast<T*>(g_pool.alloc(memsize));
            cudaMemcpy(d_out, reinterpret_cast<void*>(input_ptr), 
                memsize, cudaMemcpyDeviceToDevice);
            launch_index_put<T>(
                reinterpret_cast<T*>(src_ptr), d_out,
                src_idx, in_idx, slice_idx);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

}

