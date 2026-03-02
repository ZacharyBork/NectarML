#include <pybind11/numpy.h>
#include <common.h>

/* KERNELS */

template<typename T>
void launch_add(T* a, T* b, T* out, size_t n_elements);

/* FUNCTIONS */

uintptr_t add(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
    DISPATCH_DTYPE(dtype, T, {
        T* d_out;
        cudaMalloc(&d_out, n_elements * sizeof(T));
        launch_add<T>(
            reinterpret_cast<T*>(a_ptr),
            reinterpret_cast<T*>(b_ptr),
            d_out,
            n_elements
        );
        return reinterpret_cast<uintptr_t>(d_out);
    });
}

