#pragma once

#include <stdint.h>
#include <cuda_fp16.h>
#include <stdexcept>

/*
 * Defines a datatype structure for tensors in the host layer and on device.
 * 
 * The DType enum class is exposed directly to pybind, allowing it to be used
 * in Python. The Python-side dtype class (nectarml.typing._dtype) then creates
 * a constant mapping between the numpy dtypes used for CPU tensors and these
 * DTypes in the host layer.
 */
enum class DType {
    Float32,
    Float16,
    UInt8,
    Int32,
    Bool
};

/*
 * Dispatcher macro for DType dependent calculations. Allows for automatic 
 * mapping of DTypes to concrete C++ data types for function dispatch.
 * 
 * Example:
 *
 * void my_function(uintptr_t tensor_ptr, DType dtype) {
 *     DISPATCH_DTYPE(dtype, T, {
 *         T* d_out = static_cast<T*>(g_pool.alloc(sizeof(T)));
 *         run_templated_function<T>(
 *             reinterpret_cast<T*>(tensor_ptr), d_out);
 *         return reinterpret_cast<uintptr_t>(d_out);
 *     });
 * }
 */
#define DISPATCH_DTYPE(dtype, T, ...) \
switch (dtype) { \
    case DType::Float32: { using T = float;    __VA_ARGS__; break; } \
    case DType::Float16: { using T = half;     __VA_ARGS__; break; } \
    case DType::UInt8:   { using T = uint8_t;  __VA_ARGS__; break; } \
    case DType::Int32:   { using T = int32_t;  __VA_ARGS__; break; } \
    case DType::Bool:    { using T = uint8_t;  __VA_ARGS__; break; } \
    default: throw std::runtime_error("Unsupported dtype"); \
}

/*
 * Helper function to get the single-item size in bytes of a given DType.
 */
constexpr size_t dtype_itemsize(DType dtype) {
    DISPATCH_DTYPE(dtype, T, { return sizeof(T); });
}
