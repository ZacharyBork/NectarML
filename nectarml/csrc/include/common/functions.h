#pragma once

#include <cuda_fp16.h>
#include <pybind11/stl.h>
#include <limits>

/* GLOBAL FUNCTIONS */

constexpr int nextPow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* 
 * DEVICE FUNCTIONS
 *
 * These only exist due the the requirement for CUDA fp16 to use dedicated
 * functions for some core operations. These functions are DType templated 
 * and will automatically delegate to the dedicated functions if templated for
 * fp16, meaning you can call the same core function regardless of input DType.
 */

// Returns the maximum value that a given data type can represent.
template<typename T>
__host__ __device__ T max_val() {
    if constexpr (std::is_same_v<T, half>) { return __float2half(65504.0f); } 
    else { return std::numeric_limits<T>::max(); }
}

// Returns the minimum value that a given data type can represent.
template<typename T>
__host__ __device__ T min_val() {
    if constexpr (std::is_same_v<T, half>) { return __float2half(-65504.0f); } 
    else { return std::numeric_limits<T>::lowest(); }
}

// Returns the smaller of two provided values.
template<typename T>
__device__ void device_min(volatile T& a, volatile T b) {
    if constexpr (std::is_same_v<T, half>) { a = __hmin(a, b); } 
    else { a = min(a, b); }
}

// Returns the larger of two provided values.
template<typename T>
__device__ void device_max(volatile T& a, volatile T b) {
    if constexpr (std::is_same_v<T, half>) { a = __hmax(a, b); } 
    else { a = max(a, b); }
}

// Adds two provided values together and returns the result.
template<typename T>
__device__ void device_add(volatile T& a, volatile T b) {
    if constexpr (std::is_same_v<T, half>) { a = __hadd(a, b); } 
    else { a += b; }
}

// Subtracts b from a and returns the result.
template<typename T>
__device__ void device_sub(volatile T& a, volatile T b) {
    if constexpr (std::is_same_v<T, half>) { a = __hsub(a, b); } 
    else { a -= b; }
}

// Multiplies two provided values together and returns the result.
template<typename T>
__device__ void device_prod(volatile T& a, volatile T b) {
    if constexpr (std::is_same_v<T, half>) { a = __hmul(a, b); } 
    else { a *= b; }
}

/*
 * DType templated CUDA atomicAdd wrapper.
 * 
 * atomicAdd does not work for uint8_t. This device macro just wraps atomicAdd
 * with a DType template which will run the standard function for all DTypes,
 * save for uint8_t, where it will run the function with inputs cast to 32bit
 * integers.
 */
#ifdef __CUDACC__
template<typename T>
__device__ void atomic_add(T* address, T val) {
    if constexpr (std::is_same_v<T, uint8_t>) {
        static_cast<uint8_t>(
            atomicAdd(reinterpret_cast<int*>(address), static_cast<int>(val)));
    }
    else atomicAdd(address, val);
}
#endif


