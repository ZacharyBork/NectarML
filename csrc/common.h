#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdexcept>

constexpr int BLOCK_SIZE_1D = 256;
constexpr int BLOCK_SIZE_2D = 16;

enum class DType {
    Float32,
    Float16,
    UInt8,
    Int32,
};

#define DISPATCH_DTYPE(dtype, T, ...) \
switch (dtype) { \
    case DType::Float32: { using T = float;    __VA_ARGS__; break; } \
    case DType::Float16: { using T = half;     __VA_ARGS__; break; } \
    case DType::UInt8:   { using T = uint8_t;  __VA_ARGS__; break; } \
    case DType::Int32:   { using T = int32_t;  __VA_ARGS__; break; } \
    default: throw std::runtime_error("Unsupported dtype"); \
}

