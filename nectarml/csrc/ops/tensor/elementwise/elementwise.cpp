#include "ops/tensor/elementwise/elementwise.h"

/* KERNELS */

template<typename T, template<typename> class Op>
void launch_elementwise_compare(
    T* x, T* y, bool* out, 
    BroadcastIndex x_index, BroadcastIndex y_index,
    ShapeArray out_shape, int ndim,
    size_t n_elements
);

template<typename T, template<typename> class Op>
void launch_elementwise_compare_ts(T* x, float value, bool* out, size_t n_elements);

template<typename T, template<typename> class Op>
void launch_elementwise_math_1tensor(T* x, T* out, size_t n_elements);

template<typename T, template<typename> class Op>
void launch_elementwise_math_2tensor(
    T* x, T* y, T* out, 
    BroadcastIndex x_index, BroadcastIndex y_index,
    ShapeArray out_shape, int ndim,
    size_t n_elements
);

template<typename T, template<typename> class Op>
void launch_elementwise_math_tensorscalar(T* x, T* out, float value, size_t n_elements);

/* WRAPPERS */

template<template<typename> class Op>
uintptr_t call_elemwise_compare(
    uintptr_t a_ptr, uintptr_t b_ptr,
    std::vector<int> a_shape,
    std::vector<int> b_shape,
    std::vector<int> out_shape,
    DType dtype
) {
    int ndim = out_shape.size();
    size_t n_elements = 1;
    for (int s : out_shape) n_elements *= s;

    BroadcastIndex a_idx;
    a_idx.ndim = ndim;
    int a_offset = ndim - a_shape.size();
    for (int i = 0; i < ndim; i++) {
        a_idx.shape[i] = (i < a_offset) ? 1 : a_shape[i - a_offset];
    }
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        a_idx.strides[i] = (a_idx.shape[i] == 1) ? 0 : stride;
        stride *= a_idx.shape[i];
    }

    BroadcastIndex b_idx;
    b_idx.ndim = ndim;
    int b_offset = ndim - b_shape.size();
    for (int i = 0; i < ndim; i++) {
        b_idx.shape[i] = (i < b_offset) ? 1 : b_shape[i - b_offset];
    }
    stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        b_idx.strides[i] = (b_idx.shape[i] == 1) ? 0 : stride;
        stride *= b_idx.shape[i];
    }

    ShapeArray out_shape_arr;
    out_shape_arr.ndim = ndim;
    for (int i = 0; i < ndim; i++) out_shape_arr.dims[i] = out_shape[i];

    DISPATCH_DTYPE(dtype, T, {
        bool* d_out = static_cast<bool*>(g_pool.alloc(n_elements * sizeof(bool)));
        launch_elementwise_compare<T, Op>(
            reinterpret_cast<T*>(a_ptr),
            reinterpret_cast<T*>(b_ptr),
            d_out, a_idx, b_idx, out_shape_arr, ndim, n_elements);
        return reinterpret_cast<uintptr_t>(d_out);
    });
}

template<template<typename> class Op>
uintptr_t call_elemwise_compare_ts(
    uintptr_t a_ptr, 
    float value, 
    size_t n_elements, 
    DType dtype
) {
    DISPATCH_DTYPE(dtype, T, {
        bool* d_out = static_cast<bool*>(g_pool.alloc(n_elements * sizeof(bool)));
        launch_elementwise_compare_ts<T, Op>(
            reinterpret_cast<T*>(a_ptr), value, d_out, n_elements);
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
        T* d_out = static_cast<T*>(g_pool.alloc(n_elements * sizeof(T)));
        launch_elementwise_math_1tensor<T, Op>(
            reinterpret_cast<T*>(x_ptr), d_out, n_elements);
        return reinterpret_cast<uintptr_t>(d_out);
    });
}

template<template<typename> class Op>
uintptr_t call_elemwise_2tensor(
    uintptr_t a_ptr, uintptr_t b_ptr,
    std::vector<int> a_shape,
    std::vector<int> b_shape,
    std::vector<int> out_shape,
    DType dtype
) {
    int ndim = out_shape.size();
    size_t n_elements = 1;
    for (int s : out_shape) n_elements *= s;

    BroadcastIndex a_idx;
    a_idx.ndim = ndim;
    int a_offset = ndim - a_shape.size();
    for (int i = 0; i < ndim; i++) {
        a_idx.shape[i] = (i < a_offset) ? 1 : a_shape[i - a_offset];
    }
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        a_idx.strides[i] = (a_idx.shape[i] == 1) ? 0 : stride;
        stride *= a_idx.shape[i];
    }

    BroadcastIndex b_idx;
    b_idx.ndim = ndim;
    int b_offset = ndim - b_shape.size();
    for (int i = 0; i < ndim; i++) {
        b_idx.shape[i] = (i < b_offset) ? 1 : b_shape[i - b_offset];
    }
    stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        b_idx.strides[i] = (b_idx.shape[i] == 1) ? 0 : stride;
        stride *= b_idx.shape[i];
    }

    ShapeArray out_shape_arr;
    out_shape_arr.ndim = ndim;
    for (int i = 0; i < ndim; i++) out_shape_arr.dims[i] = out_shape[i];

    DISPATCH_DTYPE(dtype, T, {
        T* d_out = static_cast<T*>(g_pool.alloc(n_elements * sizeof(T)));
        launch_elementwise_math_2tensor<T, Op>(
            reinterpret_cast<T*>(a_ptr),
            reinterpret_cast<T*>(b_ptr),
            d_out, a_idx, b_idx, out_shape_arr, ndim, n_elements);
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
        T* d_out = static_cast<T*>(g_pool.alloc(n_elements * sizeof(T)));
        launch_elementwise_math_tensorscalar<T, Op>(
            reinterpret_cast<T*>(base_ptr), d_out, value, n_elements);
        return reinterpret_cast<uintptr_t>(d_out);
    });
}

namespace nectar {

    /* COMPARISON */

    uintptr_t equal(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_compare<ElemWiseEqOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t equal_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_compare_ts<ElemWiseEqTSOp>(
            in_ptr, value, n_elements, dtype
        );
    }

    uintptr_t not_equal(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_compare<ElemWiseNeOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t not_equal_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_compare_ts<ElemWiseNeTSOp>(
            in_ptr, value, n_elements, dtype
        );
    }

    uintptr_t less_than(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_compare<ElemWiseLtOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t less_than_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_compare_ts<ElemWiseLtTSOp>(
            in_ptr, value, n_elements, dtype
        );
    }

    uintptr_t less_than_or_equal(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_compare<ElemWiseLeOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t less_than_or_equal_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_compare_ts<ElemWiseLeTSOp>(
            in_ptr, value, n_elements, dtype
        );
    }

    uintptr_t greater_than(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_compare<ElemWiseGtOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t greater_than_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_compare_ts<ElemWiseGtTSOp>(
            in_ptr, value, n_elements, dtype
        );
    }

    uintptr_t greater_than_or_equal(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_compare<ElemWiseGeOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t greater_than_or_equal_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_compare_ts<ElemWiseGeTSOp>(
            in_ptr, value, n_elements, dtype
        );
    }

    /* BASIC */

    uintptr_t add(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseAddOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t add_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    ) {
        return call_elemwise_tensorscalar<ElemWiseAddTSOp>(
            in_ptr, value, n_elements, dtype
        );
    }

    uintptr_t subtract(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseSubOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t subtract_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    ) {
        return call_elemwise_tensorscalar<ElemWiseSubTSOp>(
            in_ptr, value, n_elements, dtype
        );
    }

    uintptr_t multiply(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseMulOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t multiply_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    ) {
        return call_elemwise_tensorscalar<ElemWiseMulTSOp>(
            in_ptr, value, n_elements, dtype
        );
    }

    uintptr_t divide(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseDivOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t divide_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    ) {
        return call_elemwise_tensorscalar<ElemWiseDivTSOp>(
            in_ptr, value, n_elements, dtype
        );
    }

    uintptr_t negate(uintptr_t a_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseNegateOp>(a_ptr, n_elements, dtype);
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
        return call_elemwise_1tensor<ElemWiseTanhOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t atan(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseAtanOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t atanh(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseAtanhOp>(x_ptr, n_elements, dtype);
    }

    uintptr_t atan2(
        uintptr_t b_ptr, uintptr_t a_ptr,
        std::vector<int> b_shape,
        std::vector<int> a_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseAtan2Op>(
            b_ptr, a_ptr, b_shape, a_shape, out_shape, dtype
        );
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

    uintptr_t fmod(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseFModOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t fmod_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    ) {
        return call_elemwise_tensorscalar<ElemWiseFmodfTSOp>(
            in_ptr, value, n_elements, dtype
        );
    }

    /* MIN / MAX */

    uintptr_t min(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseMinOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t min_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    ) {
        return call_elemwise_tensorscalar<ElemWiseMinfTSOp>(
            in_ptr, value, n_elements, dtype
        );
    }

    uintptr_t max(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseMaxOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t max_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    ) {
        return call_elemwise_tensorscalar<ElemWiseMaxfTSOp>(
            in_ptr, value, n_elements, dtype
        );
    }

    uintptr_t clamp(
        uintptr_t base_ptr,
        float min_value, 
        float max_value,
        size_t n_elements, 
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out = static_cast<T*>(g_pool.alloc(n_elements * sizeof(T)));
            launch_elementwise_math_tensorscalar<T, ElemWiseScalarMaxOp>(
                reinterpret_cast<T*>(base_ptr), d_out, min_value, n_elements);
            launch_elementwise_math_tensorscalar<T, ElemWiseScalarMinOp>(
                d_out, d_out, max_value, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* SIGN */

    uintptr_t sign(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        return call_elemwise_1tensor<ElemWiseSignOp>(x_ptr, n_elements, dtype);
    }

    /* COPYSIGN */

    uintptr_t copysign(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseCopysignOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
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

    uintptr_t eq_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseScalarEqMaskOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t ne_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseScalarNeMaskOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t lt_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseScalarLtMaskOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t le_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseScalarLeMaskOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t gt_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseScalarGtMaskOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t ge_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        return call_elemwise_tensorscalar<ElemWiseScalarGeMaskOp>(base_ptr, value, n_elements, dtype);
    }

    uintptr_t eq_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseTensorEqMaskOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t ne_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseTensorNeMaskOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t lt_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseTensorLtMaskOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t le_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseTensorLeMaskOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t gt_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseTensorGtMaskOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }

    uintptr_t ge_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    ) {
        return call_elemwise_2tensor<ElemWiseTensorGeMaskOp>(
            a_ptr, b_ptr, a_shape, b_shape, out_shape, dtype
        );
    }
    
}



