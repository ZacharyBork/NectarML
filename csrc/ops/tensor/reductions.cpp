#include <pybind11/numpy.h>
#include "common.h"
#include "ops/policies/reductions.h"
#include "ops/policies/elementwise.h"

/* KERNELS */

template<typename T, template<typename> class Op>
void launch_elementwise_math_tensorscalar(T* x, T* out, float value, size_t n_elements);

template<typename T, template<typename> class Op>
void launch_reduce(
    T* in_data, T* out_data, size_t n_elements, 
    float identity = std::numeric_limits<float>::quiet_NaN()
);

template<typename T, template<typename> class Op>
void launch_reduce_dim(
    T* in_data, T* out_data, 
    TensorIndex in_idx, TensorIndex out_idx,
    int reduce_dim, float identity = std::numeric_limits<float>::quiet_NaN()
);

namespace nectar {

    /* MIN */

    uintptr_t reduce_min(uintptr_t in_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_reduce<T, MinOp>(reinterpret_cast<T*>(in_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t reduce_min_dim(
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
            launch_reduce_dim<T, MinOp>(
                reinterpret_cast<T*>(in_ptr), d_out,
                in_idx, out_idx, reduce_dim);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* MAX */

    uintptr_t reduce_max(uintptr_t in_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_reduce<T, MaxOp>(reinterpret_cast<T*>(in_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t reduce_max_dim(
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
            launch_reduce_dim<T, MaxOp>(
                reinterpret_cast<T*>(in_ptr), d_out,
                in_idx, out_idx, reduce_dim);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* MEAN */

    uintptr_t reduce_mean(uintptr_t in_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_reduce<T, SumOp>(reinterpret_cast<T*>(in_ptr), d_out, n_elements);
            launch_elementwise_math_tensorscalar<T, ElemWiseScalarDivOp>(
                reinterpret_cast<T*>(d_out), d_out,
                static_cast<float>(n_elements), n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t reduce_mean_dim(
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
            launch_reduce_dim<T, SumOp>(
                reinterpret_cast<T*>(in_ptr), d_out,
                in_idx, out_idx, reduce_dim);
            launch_elementwise_math_tensorscalar<T, ElemWiseScalarDivOp>(
                reinterpret_cast<T*>(d_out), d_out,
                static_cast<float>(shape[reduce_dim]), shape[reduce_dim]);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* SUM */

    uintptr_t reduce_sum(uintptr_t in_ptr, size_t n_elements, float initial, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_reduce<T, SumOp>(reinterpret_cast<T*>(in_ptr), d_out, n_elements, initial);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t reduce_sum_dim(
        uintptr_t in_ptr, 
        std::vector<int> shape,
        int reduce_dim,
        float initial,
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
            launch_reduce_dim<T, SumOp>(
                reinterpret_cast<T*>(in_ptr), d_out,
                in_idx, out_idx, reduce_dim, initial);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* PROD */

    uintptr_t reduce_prod(uintptr_t in_ptr, size_t n_elements, float initial, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_reduce<T, ProdOp>(reinterpret_cast<T*>(in_ptr), d_out, n_elements, initial);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t reduce_prod_dim(
        uintptr_t in_ptr, 
        std::vector<int> shape,
        int reduce_dim,
        float initial,
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
            launch_reduce_dim<T, ProdOp>(
                reinterpret_cast<T*>(in_ptr), d_out,
                in_idx, out_idx, reduce_dim, initial);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

}

