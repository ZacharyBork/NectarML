#pragma once

#include <stdint.h>
#include <cuda_fp16.h>
#include <stdexcept>

enum class DType {
    Float32,
    Float16,
    UInt8,
    Int32,
    Bool
};


#define DISPATCH_DTYPE(dtype, T, ...) \
switch (dtype) { \
    case DType::Float32: { using T = float;    __VA_ARGS__; break; } \
    case DType::Float16: { using T = half;     __VA_ARGS__; break; } \
    case DType::UInt8:   { using T = uint8_t;  __VA_ARGS__; break; } \
    case DType::Int32:   { using T = int32_t;  __VA_ARGS__; break; } \
    case DType::Bool:    { using T = uint8_t;  __VA_ARGS__; break; } \
    default: throw std::runtime_error("Unsupported dtype"); \
}

constexpr size_t dtype_itemsize(DType dtype) {
    DISPATCH_DTYPE(dtype, T, { return sizeof(T); });
}
