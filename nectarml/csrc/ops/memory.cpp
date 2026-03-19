#include "common.h"

#include <curand.h>
#include <curand_kernel.h>

namespace py = pybind11;

/* KERNELS */

template<typename T>
void launch_alloc_cuda_full(T* dst, size_t n_elements, T fill_value);

template<typename T>
void launch_alloc_cuda_random(
    T* dst, 
    size_t n_elements, 
    curandState* random_state, 
    unsigned long long seed,
    T min_value, 
    T max_value
);

/* FUNCTIONS */

void free_cuda(uintptr_t ptr) { cudaFree(reinterpret_cast<void*>(ptr)); }

py::tuple get_cuda_meminfo() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    size_t used = total - free;
    return py::make_tuple(total, free, used);
}

uintptr_t alloc_cuda_full(size_t n_elements, DType dtype, double fill_value) {
    DISPATCH_DTYPE(dtype, T, {
        T* d_ptr;
        size_t nbytes = n_elements * sizeof(T);
        cudaMalloc(&d_ptr, nbytes);
        launch_alloc_cuda_full<T>(d_ptr, n_elements, static_cast<T>(fill_value));
        return reinterpret_cast<uintptr_t>(d_ptr);
    });
}

uintptr_t alloc_cuda_random(
    size_t n_elements, 
    DType dtype, 
    unsigned long long seed,
    float min_value, 
    float max_value
) {
    DISPATCH_DTYPE(dtype, T, {
        T* d_ptr;
        curandState* d_state;

        cudaMalloc(&d_ptr, n_elements * sizeof(T));
        launch_alloc_cuda_random<T>(
            d_ptr, n_elements, d_state, seed,
            static_cast<T>(min_value), static_cast<T>(max_value));
        return reinterpret_cast<uintptr_t>(d_ptr);
    });
}

uintptr_t alloc_cuda_empty(size_t n_elements, DType dtype) {
    DISPATCH_DTYPE(dtype, T, {
        T* d_ptr;
        cudaMalloc(&d_ptr, n_elements * sizeof(T));
        return reinterpret_cast<uintptr_t>(d_ptr);
    });
}




