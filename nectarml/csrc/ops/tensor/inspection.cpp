#include "common.h"
#include "ops/policies/inspection.h"
#include "allocator_pool/allocator_pool.h"

/* KERNELS */

template<typename T, template<typename> class Pred, class Op>
void launch_inspection(T* in_data, bool* out_data, size_t n_elements);

template<template<typename> class Pred, class Op>
bool run_inspect(uintptr_t in_ptr, size_t n_elements, DType dtype) {
    DISPATCH_DTYPE(dtype, T, {
        bool* d_out = static_cast<bool*>(g_pool.alloc(n_elements * sizeof(bool)));
        launch_inspection<T, Pred, Op>(
            reinterpret_cast<T*>(in_ptr), d_out, n_elements);
        
        bool result;
        cudaMemcpy(&result, d_out, sizeof(bool), cudaMemcpyDeviceToHost);
        g_pool.free(d_out, n_elements * sizeof(T));
        return result;
    });
}

namespace nectar {

    bool is_inf(uintptr_t in_ptr, size_t n_elements, DType dtype) {
        return run_inspect<IsInfPred, AllOp>(in_ptr, n_elements, dtype);
    }

    bool is_finite(uintptr_t in_ptr, size_t n_elements, DType dtype) {
        return run_inspect<IsFinitePred, AllOp>(in_ptr, n_elements, dtype);
    }

    bool is_nan(uintptr_t in_ptr, size_t n_elements, DType dtype) {
        return run_inspect<IsNanPred, AllOp>(in_ptr, n_elements, dtype);
    }

    bool has_inf(uintptr_t in_ptr, size_t n_elements, DType dtype) {
        return run_inspect<IsInfPred, AnyOp>(in_ptr, n_elements, dtype);
    }

    bool has_nan(uintptr_t in_ptr, size_t n_elements, DType dtype) {
        return run_inspect<IsNanPred, AnyOp>(in_ptr, n_elements, dtype);
    }

}