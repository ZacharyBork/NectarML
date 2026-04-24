#pragma once

#include "include/common/dtype.h"
#include "ops/common.h"

cublasHandle_t get_cublas_handle();
void destroy_cublas_handle();

uintptr_t cast_tensor(uintptr_t device_ptr, size_t n_elements, DType src_dtype, DType dst_dtype);
uintptr_t to_cuda(uintptr_t host_ptr, size_t n_elements, DType dtype);
py::array to_cpu(uintptr_t device_ptr, std::vector<size_t> shape, DType dtype);
uintptr_t clone_tensor(uintptr_t device_ptr, size_t n_elements, DType dtype);
