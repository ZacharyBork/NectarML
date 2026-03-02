#include <pybind11/numpy.h>
#include <common.h>

/* KERNELS */

template<typename T>
void launch_add(T* a, T* b, T* out, size_t n_elements);

template<typename T>
void launch_subtract(T* a, T* b, T* out, size_t n_elements);

template<typename T>
void launch_multiply(T* a, T* b, T* out, size_t n_elements);

template<typename T>
void launch_divide(T* a, T* b, T* out, size_t n_elements);

template<typename T>
void launch_sqrt(T* x, T* out, size_t n_elements);

template<typename T>
void launch_rsqrt(T* x, T* out, size_t n_elements);

template<typename T>
void launch_exp(T* x, T* out, size_t n_elements);

template<typename T>
void launch_log(T* x, T* out, size_t n_elements);

template<typename T>
void launch_log2(T* x, T* out, size_t n_elements);

template<typename T>
void launch_log10(T* x, T* out, size_t n_elements);

template<typename T>
void launch_sin(T* x, T* out, size_t n_elements);

template<typename T>
void launch_cos(T* x, T* out, size_t n_elements);

template<typename T>
void launch_tan(T* x, T* out, size_t n_elements);

template<typename T>
void launch_atan(T* x, T* out, size_t n_elements);

template<typename T>
void launch_atan2(T* y, T* x, T* out, size_t n_elements);

template<typename T>
void launch_pow(T* base, T* out, float exponent, size_t n_elements);

template<typename T>
void launch_abs(T* x, T* out, size_t n_elements);

template<typename T>
void launch_floor(T* x, T* out, size_t n_elements);

template<typename T>
void launch_ceil(T* x, T* out, size_t n_elements);

template<typename T>
void launch_round(T* x, T* out, size_t n_elements);

template<typename T>
void launch_mod(T* x, T* y, T* out, size_t n_elements);

template<typename T>
void launch_fmod(T* x, T* y, T* out, size_t n_elements);

template<typename T>
void launch_min(T* x, T* y, T* out, size_t n_elements);

template<typename T>
void launch_max(T* x, T* y, T* out, size_t n_elements);

template<typename T>
void launch_copysign(T* x, T* y, T* out, size_t n_elements);

template<typename T>
void launch_trunc(T* x, T* out, size_t n_elements);

namespace nectar {

    /* ADDITION */

    uintptr_t add(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_add<T>(
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
            launch_multiply<T>(
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
            launch_subtract<T>(
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
            launch_divide<T>(
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
            launch_sqrt<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t rsqrt(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_rsqrt<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* EXPONENT */

    uintptr_t exp(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_exp<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* LOG */

    uintptr_t log(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_log<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t log2(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_log2<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t log10(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_log10<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* SIN / COS */

    uintptr_t sin(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_sin<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t cos(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_cos<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* TAN / ATAN */

    uintptr_t tan(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_tan<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t atan(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_atan<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t atan2(uintptr_t y_ptr, uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_atan2<T>(
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
            launch_pow<T>(
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
            launch_abs<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* ROUNDING */

    uintptr_t floor(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_floor<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t ceil(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_ceil<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t round(uintptr_t x_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_round<T>(reinterpret_cast<T*>(x_ptr), d_out, n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    /* MODULO */

    uintptr_t mod(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_mod<T>(
                reinterpret_cast<T*>(x_ptr),
                reinterpret_cast<T*>(y_ptr),
                d_out, 
                n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t fmod(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, n_elements * sizeof(T));
            launch_fmod<T>(
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
            launch_min<T>(
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
            launch_max<T>(
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
            launch_copysign<T>(
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
            launch_trunc<T>(
                reinterpret_cast<T*>(x_ptr),
                d_out, 
                n_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }
    
}



