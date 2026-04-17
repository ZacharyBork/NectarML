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

template<typename T>
void launch_cumsum(
    T* input, T* output,
    int outer, int dim_size, int inner
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

    /* MEAN */

    uintptr_t reduce_mean(uintptr_t in_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_reduce<T, SumOp>(reinterpret_cast<T*>(in_ptr), d_out, n_elements);
            launch_elementwise_math_tensorscalar<T, ElemWiseDivTSOp>(
                d_out, d_out, static_cast<float>(n_elements), 1);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t reduce_mean_dim(
        uintptr_t in_ptr,
        std::vector<int> shape,
        int reduce_dim,
        DType dtype
    ) {
        uintptr_t sum_ptr = reduce_sum_dim(
            in_ptr, shape, reduce_dim, 0.0f, dtype);
        
        size_t n = shape[reduce_dim];
        size_t out_elements = 1;
        for (int i = 0; i < shape.size(); i++)
            if (i != reduce_dim) out_elements *= shape[i];
        
        DISPATCH_DTYPE(dtype, T, {
            launch_elementwise_math_tensorscalar<T, ElemWiseDivTSOp>(
                reinterpret_cast<T*>(sum_ptr), 
                reinterpret_cast<T*>(sum_ptr), 
                static_cast<float>(n),
                out_elements);
        });
        return sum_ptr;
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

    uintptr_t reduce_cumsum(
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
            launch_cumsum<T>(d_input, d_output, outer, dim_size, inner);
            return reinterpret_cast<uintptr_t>(d_output);
        });
    }

}

