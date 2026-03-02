#include <pybind11/numpy.h>
#include "common.h"

/* KERNELS */

template<typename T>
void launch_alloc_cuda_full(T* dst, size_t n_elements, T fill_value);

/* FUNCTIONS */

void free_cuda(uintptr_t ptr) { cudaFree(reinterpret_cast<void*>(ptr)); }

uintptr_t alloc_cuda_full(size_t n_elements, DType dtype, double fill_value) {
    DISPATCH_DTYPE(dtype, T, {
        T* d_ptr;
        size_t nbytes = n_elements * sizeof(T);
        cudaMalloc(&d_ptr, nbytes);
        launch_alloc_cuda_full<T>(d_ptr, n_elements, static_cast<T>(fill_value));
        return reinterpret_cast<uintptr_t>(d_ptr);
    });
}


