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

template<typename T>
__device__ void device_prod(volatile T& a, volatile T b) {
    if constexpr (std::is_same_v<T, half>) { a = __hmul(a, b); } 
    else { a *= b; }
}

#ifdef __CUDACC__
template<typename T>
__device__ void atomic_add(T* address, T val) {
    if constexpr (std::is_same_v<T, float>) {
        atomicAdd(address, val);
    } else if constexpr (std::is_same_v<T, half>) {
        atomicAdd(address, val);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        atomicAdd(address, val);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        static_cast<uint8_t>(
            atomicAdd(reinterpret_cast<int*>(address), static_cast<int>(val)));
    }
}
#endif
