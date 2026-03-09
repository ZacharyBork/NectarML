#include <pybind11/numpy.h>
#include "common.h"
#include "ops/policies/elementwise.h"

/* KERNELS */

template<typename T, template<typename> class Op>
void launch_elementwise_compare(T* x, T* y, bool* out, size_t n_elements);

template<typename T, template<typename> class Op>
void launch_elementwise_math_1tensor(T* x, T* out, size_t n_elements);

template<typename T, template<typename> class Op>
void launch_elementwise_math_2tensor(T* x, T* y, T* out, size_t n_elements);

template<typename T, template<typename> class Op>
void launch_elementwise_math_tensorscalar(T* x, T* out, float value, size_t n_elements);

/* WRAPPERS */

template<template<typename> class Op>
uintptr_t call_elemwise_compare(
    uintptr_t a_ptr, 
    uintptr_t b_ptr, 
    size_t n_elements, 
    DType dtype
) {
    DISPATCH_DTYPE(dtype, T, {
        bool* d_out;
        cudaMalloc(&d_out, n_elements * sizeof(bool));
        launch_elementwise_compare<T, Op>(
            reinterpret_cast<T*>(a_ptr),
            reinterpret_cast<T*>(b_ptr),
            d_out, n_elements);
        return reinterpret_cast<uintptr_t>(d_out);
    });
}

template<template<typename> class Op>
uintptr_t call_elemwise_1tensor(
    uintptr_t x_ptr, 
    size_t n_elements, 
    DType dtype
) {
    DISPATCH_DTYPE(dtype, T, {
        T* d_out;
        cudaMalloc(&d_out, n_elements * sizeof(T));
        launch_elementwise_math_1tensor<T, Op>(
            reinterpret_cast<T*>(x_ptr), d_out, n_elements);
        return reinterpret_cast<uintptr_t>(d_out);
    });
}

template<template<typename> class Op>
uintptr_t call_elemwise_2tensor(
    uintptr_t a_ptr, 
    uintptr_t b_ptr, 
    size_t n_elements, 
    DType dtype
) {
    DISPATCH_DTYPE(dtype, T, {
        T* d_out;
        cudaMalloc(&d_out, n_elements * sizeof(T));
        launch_elementwise_math_2tensor<T, Op>(
            reinterpret_cast<T*>(a_ptr),
            reinterpret_cast<T*>(b_ptr),
            d_out, n_elements);
        return reinterpret_cast<uintptr_t>(d_out);
    });
}

template<template<typename> class Op>
uintptr_t call_elemwise_tensorscalar(
    uintptr_t base_ptr,
    float value, 
    size_t n_elements, 
    DType dtype
) {
    DISPATCH_DTYPE(dtype, T, {
        T* d_out;
        cudaMalloc(&d_out, n_elements * sizeof(T));
        launch_elementwise_math_tensorscalar<T, Op>(
            reinterpret_cast<T*>(base_ptr), d_out, value, n_elements);
        return reinterpret_cast<uintptr_t>(d_out);
    });
}

namespace nectar {

    /* COMPARISON */

    uintptr_t equal(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_compare<ElemWiseEqOp>(a_ptr, b_ptr, n_elements, dtype);
    }

    uintptr_t less_than(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_compare<ElemWiseLtOp>(a_ptr, b_ptr, n_elements, dtype);
    }

    uintptr_t less_than_or_equal(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_compare<ElemWiseLeOp>(a_ptr, b_ptr, n_elements, dtype);
    }

    uintptr_t greater_than(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_compare<ElemWiseGtOp>(a_ptr, b_ptr, n_elements, dtype);
    }

    uintptr_t greater_than_or_equal(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_compare<ElemWiseGeOp>(a_ptr, b_ptr, n_elements, dtype);
    }

    /* BASIC */

    uintptr_t add(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_2tensor<ElemWiseAddOp>(a_ptr, b_ptr, n_elements, dtype);
    }

    uintptr_t subtract(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_2tensor<ElemWiseSubOp>(a_ptr, b_ptr, n_elements, dtype);
    }

    uintptr_t multiply(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_2tensor<ElemWiseMulOp>(a_ptr, b_ptr, n_elements, dtype);
    }

    uintptr_t divide(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_2tensor<ElemWiseDivOp>(a_ptr, b_ptr, n_elements, dtype);
    }

    /* SQRT */

    uintptr_t sqrt(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseSqrtOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t rsqrt(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseRSqrtOp>(x_ptr, n_elements, dtype);
    }

    /* EXPONENT */

    uintptr_t exp(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseExpOp>(x_ptr, n_elements, dtype); }

    /* LOG */

    uintptr_t log(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseLogOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t log2(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseLog2Op>(x_ptr, n_elements, dtype);
    }

    uintptr_t log10(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseLog10Op>(x_ptr, n_elements, dtype);
    }

    /* SIN / COS */

    uintptr_t sin(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseSinOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t asin(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseAsinOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t sinh(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseSinhOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t asinh(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseAsinhOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t cos(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseCosOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t acos(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseAcosOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t cosh(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseCoshOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t acosh(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseAcoshOp>(x_ptr, n_elements, dtype);
    }

    /* TAN / ATAN */

    uintptr_t tan(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseTanOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t tanh(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseTahnOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t atan(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseAtanOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t atanh(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseAtanhOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t atan2(uintptr_t y_ptr, uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_2tensor<ElemWiseAtan2Op>(y_ptr, x_ptr, n_elements, dtype);
    }

    /* POW */

    uintptr_t pow(uintptr_t base_ptr, float exponent, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWisePowOp>(base_ptr, exponent, n_elements, dtype);
    }

    /* ABS */

    uintptr_t abs(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseAbsOp>(x_ptr, n_elements, dtype);
    }

    /* ROUNDING */

    uintptr_t floor(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseFloorOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t ceil(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseCeilOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t round(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseRoundOp>(x_ptr, n_elements, dtype);
    }

    /* MODULO */

    uintptr_t fmod(uintptr_t y_ptr, uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_2tensor<ElemWiseFModOp>(x_ptr, y_ptr, n_elements, dtype);
    }

    /* MIN / MAX */

    uintptr_t min(uintptr_t y_ptr, uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_2tensor<ElemWiseMinOp>(x_ptr, y_ptr, n_elements, dtype);
    }

    uintptr_t max(uintptr_t y_ptr, uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_2tensor<ElemWiseMaxOp>(x_ptr, y_ptr, n_elements, dtype);
    }

    /* COPYSIGN */

    uintptr_t copysign(uintptr_t y_ptr, uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_2tensor<ElemWiseCopysignOp>(x_ptr, y_ptr, n_elements, dtype);
    }

    /* TRUNCATE */

    uintptr_t trunc(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseTruncOp>(x_ptr, n_elements, dtype);
    }

    /* TENSOR/SCALAR MATH */

    uintptr_t scalaradd(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseScalarAddOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t scalarsub(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseScalarSubOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t scalarmul(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseScalarMulOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t scalardiv(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseScalarDivOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t scalarmin(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseScalarMinOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t scalarmax(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseScalarMaxOp>(base_ptr, value, n_elements, dtype);
    }

    /* MASKING */

    uintptr_t equal_mask(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseEqMaskkOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t less_than_mask(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseLtMaskOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t less_than_or_equal_mask(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseLeMaskOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t greater_than_mask(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseGtMaskOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t greater_than_or_equal_mask(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseGeMaskOp>(base_ptr, value, n_elements, dtype);
    }

    /* CLAMP */

    template<template<typename> class Op>
    uintptr_t clamp(
        uintptr_t base_ptr,
        float min_value, 
        float max_value,
        size_t n_elements, 
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_tensorscalar<T, ElemWiseScalarMaxOp>(
                reinterpret_cast<T*>(base_ptr), d_out, min_value, n_elements);
            launch_elementwise_math_tensorscalar<T, ElemWiseScalarMinOp>(
                d_out, d_out, max_value, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

}



