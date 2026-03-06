#include <pybind11/numpy.h>
#include "common.h"
#include "ops/policies/elementwise.h"

/* KERNELS */

template<typename T, template<typename> class Op>
void launch_elementwise_compare(T* x, T* y, bool* out, size_t n_elements);

template<typename T, template<typename> class Op>
void launch_elementwise_math_2tensor(T* x, T* y, T* out, size_t n_elements);

template<typename T, template<typename> class Op>
void launch_elementwise_math_1tensor(T* x, T* out, size_t n_elements);

template<typename T, template<typename> class Op>
void launch_elementwise_math_tensorscalar(T* x, T* out, float value, size_t n_elements);

namespace nectar {

    /* COMPARISON */

    uintptr_t equal(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            bool* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(bool));
            launch_elementwise_compare<T, ElemWiseEqOp>(
                reinterpret_cast<T*>(a_ptr),
                reinterpret_cast<T*>(b_ptr),
                d_out,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t less_than(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            bool* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(bool));
            launch_elementwise_compare<T, ElemWiseLtOp>(
                reinterpret_cast<T*>(a_ptr),
                reinterpret_cast<T*>(b_ptr),
                d_out,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t less_than_or_equal(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            bool* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(bool));
            launch_elementwise_compare<T, ElemWiseLeOp>(
                reinterpret_cast<T*>(a_ptr),
                reinterpret_cast<T*>(b_ptr),
                d_out,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t greater_than(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            bool* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(bool));
            launch_elementwise_compare<T, ElemWiseGtOp>(
                reinterpret_cast<T*>(a_ptr),
                reinterpret_cast<T*>(b_ptr),
                d_out,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t greater_than_or_equal(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            bool* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(bool));
            launch_elementwise_compare<T, ElemWiseGeOp>(
                reinterpret_cast<T*>(a_ptr),
                reinterpret_cast<T*>(b_ptr),
                d_out,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* ADDITION */

    uintptr_t add(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_2tensor<T, ElemWiseAddOp>(
                reinterpret_cast<T*>(a_ptr),
                reinterpret_cast<T*>(b_ptr),
                d_out,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* SUBTRACTION */

    uintptr_t subtract(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_2tensor<T, ElemWiseSubOp>(
                reinterpret_cast<T*>(a_ptr),
                reinterpret_cast<T*>(b_ptr),
                d_out,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* MULTIPLICATION */

    uintptr_t multiply(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_2tensor<T, ElemWiseMulOp>(
                reinterpret_cast<T*>(a_ptr),
                reinterpret_cast<T*>(b_ptr),
                d_out,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* DIVISION */

    uintptr_t divide(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_2tensor<T, ElemWiseDivOp>(
                reinterpret_cast<T*>(a_ptr),
                reinterpret_cast<T*>(b_ptr),
                d_out,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* SQRT */

    uintptr_t sqrt(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseSqrtOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t rsqrt(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseRSqrtOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* EXPONENT */

    uintptr_t exp(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseExpOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* LOG */

    uintptr_t log(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseLogOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t log2(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseLog2Op>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t log10(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseLog10Op>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* SIN / COS */

    uintptr_t sin(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseSinOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t asin(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseAsinOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t sinh(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseSinhOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t asinh(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseAsinhOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t cos(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseCosOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t acos(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseAcosOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t cosh(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseCoshOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t acosh(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseAcoshOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* TAN / ATAN */

    uintptr_t tan(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseTanOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t tanh(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseTahnOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t atan(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseAtanOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t atanh(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseAtanhOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t atan2(uintptr_t y_ptr, uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_2tensor<T, ElemWiseAtan2Op>(
                reinterpret_cast<T*>(y_ptr),
                reinterpret_cast<T*>(x_ptr),
                d_out, 
                n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* POW */

    uintptr_t pow(uintptr_t base_ptr, float exponent, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_tensorscalar<T, ElemWisePowOp>(
                reinterpret_cast<T*>(base_ptr),
                d_out,
                exponent,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* ABS */

    uintptr_t abs(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseAbsOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* ROUNDING */

    uintptr_t floor(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseFloorOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t ceil(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseCeilOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t round(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseRoundOp>(
                reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* MODULO */

    uintptr_t fmod(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_2tensor<T, ElemWiseFModOp>(
                reinterpret_cast<T*>(x_ptr),
                reinterpret_cast<T*>(y_ptr),
                d_out, 
                n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* MIN / MAX */

    uintptr_t min(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_2tensor<T, ElemWiseMinOp>(
                reinterpret_cast<T*>(x_ptr),
                reinterpret_cast<T*>(y_ptr),
                d_out, 
                n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t max(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_2tensor<T, ElemWiseMaxOp>(
                reinterpret_cast<T*>(x_ptr),
                reinterpret_cast<T*>(y_ptr),
                d_out, 
                n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* COPYSIGN */

    uintptr_t copysign(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_2tensor<T, ElemWiseCopysignOp>(
                reinterpret_cast<T*>(x_ptr),
                reinterpret_cast<T*>(y_ptr),
                d_out, 
                n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* TRUNCATE */

    uintptr_t trunc(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_1tensor<T, ElemWiseTruncOp>(
                reinterpret_cast<T*>(x_ptr),
                d_out, 
                n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* TENSOR/SCALAR MATH */
    
    uintptr_t scalaradd(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_tensorscalar<T, ElemWiseScalarAddOp>(
                reinterpret_cast<T*>(base_ptr),
                d_out,
                value,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t scalarsub(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_tensorscalar<T, ElemWiseScalarSubOp>(
                reinterpret_cast<T*>(base_ptr),
                d_out,
                value,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t scalarmul(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_tensorscalar<T, ElemWiseScalarMulOp>(
                reinterpret_cast<T*>(base_ptr),
                d_out,
                value,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t scalardiv(uintptr_t base_ptr, float value, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_elementwise_math_tensorscalar<T, ElemWiseScalarDivOp>(
                reinterpret_cast<T*>(base_ptr),
                d_out,
                value,
                n_elements
            );
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

}



