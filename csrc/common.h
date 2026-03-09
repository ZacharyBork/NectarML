#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdexcept>
#include <vector>
#include <limits>
#include <type_traits>

/* ALLOCATION CONSTANTS */

constexpr int BLOCK_SIZE_1D = 256;
constexpr int BLOCK_SIZE_2D = 16;

/* DTYPE */

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

/* TENSOR INDEX */

#define MAX_DIMS 6

struct TensorIndex {
    int shape[MAX_DIMS];
    int strides[MAX_DIMS];
    int ndim;
    int n_elements;

    __host__ __device__ TensorIndex() : ndim(0), n_elements((0)) {
        for(int i = 0; i < MAX_DIMS; i++) {
            shape[i] = 0;
            strides[i] = 0;
        }
    }

    __host__ __device__ TensorIndex(const int* shape_, int ndim_) : ndim(ndim_) {
        n_elements = 1;
        for (int i = 0; i < ndim; i++) {
            shape[i] = shape_[i];
            n_elements *= shape[i];
        }
        for (int i = 0; i < MAX_DIMS - ndim; i++) 
            shape[ndim + i] = 0;
        _compute_strides();
    }

    __host__ __device__ void _compute_strides() {
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
    }

    __host__ __device__ int to_flat(int* indices) {
        int flat = 0;
        for(int i = 0; i < ndim; i++) 
            flat += indices[i] * strides[i];
        return flat;
    }

    __host__ __device__ void to_index(int flat, int* out_indices) {
        for(int i = ndim-1; i >= 0; i--) {
            out_indices[i] = flat % shape[i];
            flat /= shape[i];
        }
    }
};

inline TensorIndex build_tensor_index(const std::vector<int>& shape) {
    return TensorIndex(shape.data(), shape.size());
}

/* COMBINATION */

#define MAX_CONCAT_INPUTS 32

struct ConcatInputs {
    uintptr_t ptrs[MAX_CONCAT_INPUTS];
    TensorIndex indices[MAX_CONCAT_INPUTS];
    int offsets[MAX_CONCAT_INPUTS];
    int n_inputs;
};

/* PERMUTATION */

#define MAX_PERMUTE_DIMS 6

struct Permutation {
    int dims[MAX_PERMUTE_DIMS];
    int ndim;
    
    __host__ __device__ Permutation() : ndim(0) {
        for (int i = 0; i < MAX_PERMUTE_DIMS; i++) dims[i] = 0;
    }
    
    __host__ __device__ Permutation(const int* dims_, int ndim_) : ndim(ndim_) {
        for (int i = 0; i < ndim; i++) dims[i] = dims_[i];
    }
    
    __host__ __device__ Permutation inverse() const {
        Permutation inv;
        inv.ndim = ndim;
        for (int i = 0; i < ndim; i++)
            inv.dims[dims[i]] = i;
        return inv;
    }
};

/* GLOBAL FUNCTIONS */

constexpr int nextPow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

template<typename T>
__host__ __device__ T max_val() {
    if constexpr (std::is_same_v<T, half>) { return __float2half(65504.0f); } 
    else { return std::numeric_limits<T>::max(); }
}

template<typename T>
__host__ __device__ T min_val() {
    if constexpr (std::is_same_v<T, half>) { return __float2half(-65504.0f); } 
    else { return std::numeric_limits<T>::lowest(); }
}

template<typename T>
__device__ void device_min(volatile T& a, volatile T b) {
    if constexpr (std::is_same_v<T, half>) { a = __hmin(a, b); } 
    else { a = min(a, b); }
}

template<typename T>
__device__ void device_max(volatile T& a, volatile T b) {
    if constexpr (std::is_same_v<T, half>) { a = __hmax(a, b); } 
    else { a = max(a, b); }
}

template<typename T>
__device__ void device_add(volatile T& a, volatile T b) {
    if constexpr (std::is_same_v<T, half>) { a = __hadd(a, b); } 
    else { a += b; }
}

template<typename T>
__device__ void device_sub(volatile T& a, volatile T b) {
    if constexpr (std::is_same_v<T, half>) { a = __hsub(a, b); } 
    else { a -= b; }
}

/* OP POLICIES */

template<typename T>
struct SumOp {
    __device__ static void combine(volatile T& a, volatile T b) { device_add(a, b); }
    __device__ static T identity() { return static_cast<T>(0); }
};

template<typename T>
struct MinOp {
    __device__ static void combine(volatile T& a, volatile T b) { device_min(a, b); }
    __device__ static T identity() { return max_val<T>(); }
};

template<typename T>
struct MaxOp {
    __device__ static void combine(volatile T& a, volatile T b) { device_max(a, b); }
    __device__ static T identity() { return min_val<T>(); }
};


