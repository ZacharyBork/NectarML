#pragma once

#include "common/functions.h"

/* OP POLICIES */

template<typename T>
struct SumOp {
    __device__ static void combine(T& a, T b) { device_add(a, b); }
    __device__ static T identity() { return static_cast<T>(0.0f); }
};

template<typename T>
struct MinOp {
    __device__ static void combine(T& a, T b) { device_min(a, b); }
    __device__ static T identity() { return max_val<T>(); }
};

template<typename T>
struct MaxOp {
    __device__ static void combine(T& a, T b) { device_max(a, b); }
    __device__ static T identity() { return min_val<T>(); }
};

template<typename T>
struct ProdOp {
    __device__ static void combine(T& a, T b) { device_prod(a, b); }
    __device__ static T identity() { return static_cast<T>(1.0f); }
};


