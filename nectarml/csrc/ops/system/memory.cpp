#include "ops/common.h"
#include "include/common/dtype.h"
#include "pool/allocator_pool.h"

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

/* INSPECTION */

py::tuple get_cuda_meminfo() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    size_t used = total - free;
    return py::make_tuple(total, free, used);
}

/* UTILITIES */

void cuda_synchronize() { cudaDeviceSynchronize(); }

void free_cuda(uintptr_t ptr, size_t n_elements, DType dtype) { 
    if (ptr == 0) return;
    DISPATCH_DTYPE(dtype, T, {
        T* d_ptr = reinterpret_cast<T*>(ptr);
        g_pool.free(d_ptr, n_elements * sizeof(T));
    });
}

void memcpy_to_cuda(uintptr_t dst, uintptr_t host_ptr, size_t size_bytes) {
    cudaMemcpy(
        reinterpret_cast<void*>(dst), 
        reinterpret_cast<void*>(host_ptr), 
        size_bytes, cudaMemcpyHostToDevice);
}

uintptr_t alloc_cuda_empty_raw(size_t size_bytes) {
    void* ptr = static_cast<void*>(g_pool.alloc(size_bytes));
    return reinterpret_cast<uintptr_t>(ptr);
}

uintptr_t alloc_cuda_full(size_t n_elements, DType dtype, double fill_value) {
    DISPATCH_DTYPE(dtype, T, {
        size_t nbytes = n_elements * sizeof(T);
        T* d_ptr = static_cast<T*>(g_pool.alloc(nbytes));
        launch_alloc_cuda_full<T>(d_ptr, n_elements, static_cast<T>(fill_value));
        return reinterpret_cast<uintptr_t>(d_ptr);
    });
}

void fill(uintptr_t ptr, int fill_value, size_t n_elements, DType dtype) { 
    if (ptr == 0) return;
    size_t bytes = n_elements * dtype_itemsize(dtype);
    cudaMemset(reinterpret_cast<void*>(ptr), 0, bytes); 
}

uintptr_t alloc_cuda_random(
    size_t n_elements, 
    DType dtype, 
    unsigned long long seed,
    float min_value, 
    float max_value
) {
    DISPATCH_DTYPE(dtype, T, {
        curandState* d_state;
        T* d_ptr = static_cast<T*>(g_pool.alloc(n_elements * sizeof(T)));
        launch_alloc_cuda_random<T>(
            d_ptr, n_elements, d_state, seed,
            static_cast<T>(min_value), static_cast<T>(max_value));
        return reinterpret_cast<uintptr_t>(d_ptr);
    });
}

uintptr_t alloc_cuda_empty(size_t n_elements, DType dtype) {
    DISPATCH_DTYPE(dtype, T, {
        T* d_ptr = static_cast<T*>(g_pool.alloc(n_elements * sizeof(T)));
        return reinterpret_cast<uintptr_t>(d_ptr);
    });
}




