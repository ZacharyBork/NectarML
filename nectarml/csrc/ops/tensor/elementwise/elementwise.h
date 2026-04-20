#pragma once

#include "common/dtype.h"
#include "common/data_structures.h"
#include "ops/policies/elementwise.h"
#include "allocator_pool/allocator_pool.h"

namespace nectar {

    /* COMPARISON */

    uintptr_t equal(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );

    uintptr_t equal_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype);

    uintptr_t not_equal(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );

    uintptr_t not_equal_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype);

    uintptr_t less_than(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );

    uintptr_t less_than_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype);

    uintptr_t less_than_or_equal(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );

    uintptr_t less_than_or_equal_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype);

    uintptr_t greater_than(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );

    uintptr_t greater_than_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype);

    uintptr_t greater_than_or_equal(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );

    uintptr_t greater_than_or_equal_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype);

    /* BASIC */

    uintptr_t add(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );

    uintptr_t add_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );

    uintptr_t subtract(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );

    uintptr_t subtract_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );

    uintptr_t multiply(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );

    uintptr_t multiply_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );

    uintptr_t divide(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );

    uintptr_t divide_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );

    /* MATH */

    uintptr_t negate(uintptr_t a_ptr, size_t n_elements, DType dtype);
    uintptr_t sqrt(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t rsqrt(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t exp(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t log(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t log2(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t log10(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t sin(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t asin(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t sinh(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t asinh(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t cos(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t acos(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t cosh(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t acosh(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t tan(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t tanh(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t atan(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t atanh(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t atan2(
        uintptr_t b_ptr, uintptr_t a_ptr,
        std::vector<int> b_shape,
        std::vector<int> a_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t pow(uintptr_t base_ptr, float exponent, size_t n_elements, DType dtype);
    uintptr_t abs(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t floor(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t ceil(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t round(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t fmod(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t fmod_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );
    uintptr_t min(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t min_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );
    uintptr_t max(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t max_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );
    uintptr_t clamp(
        uintptr_t base_ptr,
        float min_value, 
        float max_value,
        size_t n_elements, 
        DType dtype
    );
    uintptr_t sign(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t copysign(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t trunc(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t scalaradd(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t scalarsub(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t scalarmul(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t scalardiv(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t scalarmin(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t scalarmax(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t eq_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t ne_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t lt_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t le_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t gt_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t ge_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t eq_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t ne_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t lt_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t le_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t gt_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t ge_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    
}



