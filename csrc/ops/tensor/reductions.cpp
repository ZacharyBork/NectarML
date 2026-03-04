#include <pybind11/numpy.h>
#include "common.h"

/* KERNELS */

template<typename T>
void launch_reduce_sum(T* in_data, T* out_data, size_t n_elements);

namespace nectar {

    uintptr_t reduce_sum(uintptr_t in_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_reduce_sum<T>(
                reinterpret_cast<T*>(in_ptr),
                d_out,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

}

