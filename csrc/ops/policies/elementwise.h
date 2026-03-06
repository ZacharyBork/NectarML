#include "common.h"

/* COMPARISON */

template<typename T>
struct ElemWiseEqOp {
    __device__ static bool operation(T x, T y) {
        if constexpr (std::is_same_v<T, half>) { return __heq(x, y); } 
        else { return x == y; }
    }
};

template<typename T>
struct ElemWiseLtOp {
    __device__ static bool operation(T x, T y) {
        if constexpr (std::is_same_v<T, half>) { return __hlt(x, y); } 
        else { return x < y; }
    }
};

template<typename T>
struct ElemWiseLeOp {
    __device__ static bool operation(T x, T y) {
        if constexpr (std::is_same_v<T, half>) { return __hle(x, y); } 
        else { return x <= y; }
    }
};

template<typename T>
struct ElemWiseGtOp {
    __device__ static bool operation(T x, T y) {
        if constexpr (std::is_same_v<T, half>) { return __hgt(x, y); } 
        else { return x > y; }
    }
};

template<typename T>
struct ElemWiseGeOp {
    __device__ static bool operation(T x, T y) {
        if constexpr (std::is_same_v<T, half>) { return __hge(x, y); } 
        else { return x >= y; }
    }
};

/* MATH (2 TENSOR) */

template<typename T>
struct ElemWiseAddOp {
    __device__ static T operation(T x, T y) {
        if constexpr (std::is_same_v<T, half>) { return __hadd(x, y); } 
        else { return x + y; }
    }
};

template<typename T>
struct ElemWiseSubOp {
    __device__ static T operation(T x, T y) {
        if constexpr (std::is_same_v<T, half>) { return __hsub(x, y); } 
        else { return x - y; }
    }
};

template<typename T>
struct ElemWiseMulOp {
    __device__ static T operation(T x, T y) {
        if constexpr (std::is_same_v<T, half>) { return __hmul(x, y); } 
        else { return x * y; }
    }
};

template<typename T>
struct ElemWiseDivOp {
    __device__ static T operation(T x, T y) {
        if constexpr (std::is_same_v<T, half>) { return __hdiv(x, y); } 
        else { return x / y; }
    }
};

template<typename T>
struct ElemWiseAtan2Op {
    __device__ static T operation(T y, T x) {
        return static_cast<T>(atan2f(static_cast<float>(y), static_cast<float>(x)));
    }
};

template<typename T>
struct ElemWiseFModOp {
    __device__ static T operation(T x, T y) {
        return static_cast<T>(fmodf(static_cast<float>(x), static_cast<float>(y)));
    }
};

template<typename T>
struct ElemWiseMinOp {
    __device__ static T operation(T x, T y) {
        return static_cast<T>(fminf(static_cast<float>(x), static_cast<float>(y)));
    }
};

template<typename T>
struct ElemWiseMaxOp {
    __device__ static T operation(T x, T y) {
        return static_cast<T>(fmaxf(static_cast<float>(x), static_cast<float>(y)));
    }
};

template<typename T>
struct ElemWiseCopysignOp {
    __device__ static T operation(T x, T y) {
        return static_cast<T>(copysignf(static_cast<float>(x), static_cast<float>(y)));
    }
};

/* MATH (1 TENSOR) */

template<typename T>
struct ElemWiseSqrtOp {
    __device__ static T operation(T x) { return static_cast<T>(sqrtf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseRSqrtOp {
    #ifdef __CUDACC__
    __device__ static T operation(T x) { return static_cast<T>(rsqrtf(static_cast<float>(x))); }
    #endif
};

template<typename T>
struct ElemWiseExpOp {
    __device__ static T operation(T x) { return static_cast<T>(expf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseLogOp {
    __device__ static T operation(T x) { return static_cast<T>(logf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseLog2Op {
    __device__ static T operation(T x) { return static_cast<T>(log2f(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseLog10Op {
    __device__ static T operation(T x) { return static_cast<T>(log10f(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseSinOp {
    __device__ static T operation(T x) { return static_cast<T>(sinf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseAsinOp {
    __device__ static T operation(T x) { return static_cast<T>(asinf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseSinhOp {
    __device__ static T operation(T x) { return static_cast<T>(sinhf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseAsinhOp {
    __device__ static T operation(T x) { return static_cast<T>(asinhf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseCosOp {
    __device__ static T operation(T x) { return static_cast<T>(cosf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseAcosOp {
    __device__ static T operation(T x) { return static_cast<T>(acosf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseCoshOp {
    __device__ static T operation(T x) { return static_cast<T>(coshf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseAcoshOp {
    __device__ static T operation(T x) { return static_cast<T>(acoshf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseTanOp {
    __device__ static T operation(T x) { return static_cast<T>(tanh(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseTahnOp {
    __device__ static T operation(T x) { return static_cast<T>(tanhf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseAtanOp {
    __device__ static T operation(T x) { return static_cast<T>(atanf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseAtanhOp {
    __device__ static T operation(T x) { return static_cast<T>(atanhf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseAbsOp {
    __device__ static T operation(T x) { return static_cast<T>(fabsf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseFloorOp {
    __device__ static T operation(T x) { return static_cast<T>(floorf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseCeilOp {
    __device__ static T operation(T x) { return static_cast<T>(ceilf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseRoundOp {
    __device__ static T operation(T x) { return static_cast<T>(roundf(static_cast<float>(x))); }
};

template<typename T>
struct ElemWiseTruncOp {
    __device__ static T operation(T x) { return static_cast<T>(truncf(static_cast<float>(x))); }
};

/* MATH (TENSOR/SCALAR) */

template<typename T>
struct ElemWisePowOp {
    __device__ static T operation(T x, float value) {
        return static_cast<T>(powf(static_cast<float>(x), value));
    }
};

template<typename T>
struct ElemWiseScalarAddOp {
    __device__ static T operation(T x, float value) {
        return static_cast<T>(static_cast<float>(x) + value);
    }
};

template<typename T>
struct ElemWiseScalarSubOp {
    __device__ static T operation(T x, float value) {
        return static_cast<T>(static_cast<float>(x) - value);
    }
};

template<typename T>
struct ElemWiseScalarMulOp {
    __device__ static T operation(T x, float value) {
        return static_cast<T>(static_cast<float>(x) * value);
    }
};

template<typename T>
struct ElemWiseScalarDivOp {
    __device__ static T operation(T x, float value) {
        return static_cast<T>(static_cast<float>(x) / value);
    }
};

template<typename T>
struct ElemWiseScalarMinOp {
    __device__ static T operation(T x, float value) {
        return static_cast<T>(fmin(static_cast<float>(x), value));
    }
};

template<typename T>
struct ElemWiseScalarMaxOp {
    __device__ static T operation(T x, float value) {
        return static_cast<T>(fmax(static_cast<float>(x), value));
    }
};



