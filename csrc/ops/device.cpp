#include <pybind11/numpy.h>
#include "common.h"

namespace py = pybind11;

/* KERNELS */

template<typename SrcT, typename DstT>
void launch_cast_kernel(SrcT* src, DstT* dst, size_t n_elements);

/* UTILS */

uintptr_t cast_tensor(uintptr_t device_ptr, size_t n_elements, DType src_dtype, DType dst_dtype) {
    DISPATCH_DTYPE(src_dtype, SrcT, {
        DISPATCH_DTYPE(dst_dtype, DstT, {
            DstT* d_ptr;
            cudaMalloc(&d_ptr, n_elements * sizeof(DstT));
            launch_cast_kernel<SrcT, DstT>(
                reinterpret_cast<SrcT*>(device_ptr), d_ptr, n_elements);
            cudaFree(reinterpret_cast<void*>(device_ptr));
            return reinterpret_cast<uintptr_t>(d_ptr);
        });
    });
}

uintptr_t to_cuda(uintptr_t host_ptr, size_t n_elements, DType dtype) {
    DISPATCH_DTYPE(dtype, T, {
        T* d_ptr;
        cudaMalloc(&d_ptr, n_elements * sizeof(T));
        cudaMemcpy(d_ptr, reinterpret_cast<void*>(host_ptr), 
                   n_elements * sizeof(T), cudaMemcpyHostToDevice);
        return reinterpret_cast<uintptr_t>(d_ptr);
    });
}

py::array to_cpu(uintptr_t device_ptr, std::vector<size_t> shape, DType dtype) {
    size_t n_elements = 1;
    for (auto s : shape) n_elements *= s;

    DISPATCH_DTYPE(dtype, T, {
        if constexpr (std::is_same_v<T, half>) {
            float* d_float;
            cudaMalloc(&d_float, n_elements * sizeof(float));
            launch_cast_kernel<half, float>(
                reinterpret_cast<half*>(device_ptr), d_float, n_elements);

            auto result = py::array_t<float>(shape);
            auto buf = result.request();
            cudaMemcpy(buf.ptr, d_float, 
                       n_elements * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_float);
            return result;
        } else {
            auto result = py::array_t<T>(shape);
            auto buf = result.request();
            cudaMemcpy(buf.ptr, reinterpret_cast<void*>(device_ptr),
                       n_elements * sizeof(T), cudaMemcpyDeviceToHost);
            return result;
        }
    });
}
