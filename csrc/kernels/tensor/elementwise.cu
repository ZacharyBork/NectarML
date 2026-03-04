#include "common.h"

/* COMPARISON */

template<typename T>
__global__ void equal_kernel(T* a, T* b, bool* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] == b[idx];
}

template<typename T>
void launch_equal(T* a, T* b, bool* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    equal_kernel<T><<<grid, block>>>(a, b, out, n_elements);
}

template void launch_equal<float>(float*, float*, bool*, size_t);
template void launch_equal<half>(half*, half*, bool*, size_t);
template void launch_equal<uint8_t>(uint8_t*, uint8_t*, bool*, size_t);
template void launch_equal<int32_t>(int32_t*, int32_t*, bool*, size_t);

template<typename T>
__global__ void less_than_kernel(T* a, T* b, bool* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] < b[idx];
}

template<typename T>
void launch_less_than(T* a, T* b, bool* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    less_than_kernel<T><<<grid, block>>>(a, b, out, n_elements);
}

template void launch_less_than<float>(float*, float*, bool*, size_t);
template void launch_less_than<half>(half*, half*, bool*, size_t);
template void launch_less_than<uint8_t>(uint8_t*, uint8_t*, bool*, size_t);
template void launch_less_than<int32_t>(int32_t*, int32_t*, bool*, size_t);

template<typename T>
__global__ void less_than_or_equal_kernel(T* a, T* b, bool* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] <= b[idx];
}

template<typename T>
void launch_less_than_or_equal(T* a, T* b, bool* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    less_than_or_equal_kernel<T><<<grid, block>>>(a, b, out, n_elements);
}

template void launch_less_than_or_equal<float>(float*, float*, bool*, size_t);
template void launch_less_than_or_equal<half>(half*, half*, bool*, size_t);
template void launch_less_than_or_equal<uint8_t>(uint8_t*, uint8_t*, bool*, size_t);
template void launch_less_than_or_equal<int32_t>(int32_t*, int32_t*, bool*, size_t);

template<typename T>
__global__ void greater_than_kernel(T* a, T* b, bool* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] > b[idx];
}

template<typename T>
void launch_greater_than(T* a, T* b, bool* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    greater_than_kernel<T><<<grid, block>>>(a, b, out, n_elements);
}

template void launch_greater_than<float>(float*, float*, bool*, size_t);
template void launch_greater_than<half>(half*, half*, bool*, size_t);
template void launch_greater_than<uint8_t>(uint8_t*, uint8_t*, bool*, size_t);
template void launch_greater_than<int32_t>(int32_t*, int32_t*, bool*, size_t);

template<typename T>
__global__ void greater_than_or_equal_kernel(T* a, T* b, bool* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] >= b[idx];
}

template<typename T>
void launch_greater_than_or_equal(T* a, T* b, bool* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    greater_than_or_equal_kernel<T><<<grid, block>>>(a, b, out, n_elements);
}

template void launch_greater_than_or_equal<float>(float*, float*, bool*, size_t);
template void launch_greater_than_or_equal<half>(half*, half*, bool*, size_t);
template void launch_greater_than_or_equal<uint8_t>(uint8_t*, uint8_t*, bool*, size_t);
template void launch_greater_than_or_equal<int32_t>(int32_t*, int32_t*, bool*, size_t);

/* ADDITION */

template<typename T>
__global__ void add_kernel(T* a, T* b, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}

template<typename T>
void launch_add(T* a, T* b, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    add_kernel<T><<<grid, block>>>(a, b, out, n_elements);
}

template void launch_add<float>(float*, float*, float*, size_t);
template void launch_add<half>(half*, half*, half*, size_t);
template void launch_add<uint8_t>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_add<int32_t>(int32_t*, int32_t*, int32_t*, size_t);

/* SUBTRACTION */

template<typename T>
__global__ void subtract_kernel(T* a, T* b, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] - b[idx];
}

template<typename T>
void launch_subtract(T* a, T* b, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    subtract_kernel<T><<<grid, block>>>(a, b, out, n_elements);
}

template void launch_subtract<float>(float*, float*, float*, size_t);
template void launch_subtract<half>(half*, half*, half*, size_t);
template void launch_subtract<uint8_t>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_subtract<int32_t>(int32_t*, int32_t*, int32_t*, size_t);

/* MULTIPLICATION */

template<typename T>
__global__ void multiply_kernel(T* a, T* b, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] * b[idx];
}

template<typename T>
void launch_multiply(T* a, T* b, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    multiply_kernel<T><<<grid, block>>>(a, b, out, n_elements);
}

template void launch_multiply<float>(float*, float*, float*, size_t);
template void launch_multiply<half>(half*, half*, half*, size_t);
template void launch_multiply<uint8_t>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_multiply<int32_t>(int32_t*, int32_t*, int32_t*, size_t);

/* DIVISION */

template<typename T>
__global__ void divide_kernel(T* a, T* b, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] / b[idx];
}

template<typename T>
void launch_divide(T* a, T* b, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    divide_kernel<T><<<grid, block>>>(a, b, out, n_elements);
}

template void launch_divide<float>(float*, float*, float*, size_t);
template void launch_divide<half>(half*, half*, half*, size_t);
template void launch_divide<uint8_t>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_divide<int32_t>(int32_t*, int32_t*, int32_t*, size_t);

/* SQRT */

template<typename T>
__global__ void sqrt_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(sqrt(static_cast<double>(x[idx])));
}

template<typename T>
void launch_sqrt(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    sqrt_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_sqrt<float>(float*, float*, size_t);
template void launch_sqrt<half>(half*, half*, size_t);
template void launch_sqrt<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_sqrt<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void rsqrt_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(rsqrtf(static_cast<double>(x[idx])));
}

template<typename T>
void launch_rsqrt(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    rsqrt_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_rsqrt<float>(float*, float*, size_t);
template void launch_rsqrt<half>(half*, half*, size_t);
template void launch_rsqrt<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_rsqrt<int32_t>(int32_t*, int32_t*, size_t);

/* EXPONENT */

template<typename T>
__global__ void exp_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(expf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_exp(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    exp_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_exp<float>(float*, float*, size_t);
template void launch_exp<half>(half*, half*, size_t);
template void launch_exp<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_exp<int32_t>(int32_t*, int32_t*, size_t);

/* LOG */

template<typename T>
__global__ void log_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(logf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_log(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    log_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_log<float>(float*, float*, size_t);
template void launch_log<half>(half*, half*, size_t);
template void launch_log<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_log<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void log2_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(log2f(static_cast<float>(x[idx])));
}

template<typename T>
void launch_log2(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    log2_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_log2<float>(float*, float*, size_t);
template void launch_log2<half>(half*, half*, size_t);
template void launch_log2<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_log2<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void log10_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(log10f(static_cast<float>(x[idx])));
}

template<typename T>
void launch_log10(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    log10_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_log10<float>(float*, float*, size_t);
template void launch_log10<half>(half*, half*, size_t);
template void launch_log10<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_log10<int32_t>(int32_t*, int32_t*, size_t);

/* SIN / COS */

template<typename T>
__global__ void sin_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(sinf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_sin(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    sin_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_sin<float>(float*, float*, size_t);
template void launch_sin<half>(half*, half*, size_t);
template void launch_sin<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_sin<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void asin_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(asinf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_asin(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    asin_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_asin<float>(float*, float*, size_t);
template void launch_asin<half>(half*, half*, size_t);
template void launch_asin<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_asin<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void sinh_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(sinhf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_sinh(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    sinh_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_sinh<float>(float*, float*, size_t);
template void launch_sinh<half>(half*, half*, size_t);
template void launch_sinh<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_sinh<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void asinh_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(asinhf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_asinh(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    asinh_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_asinh<float>(float*, float*, size_t);
template void launch_asinh<half>(half*, half*, size_t);
template void launch_asinh<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_asinh<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void cos_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(cosf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_cos(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    cos_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_cos<float>(float*, float*, size_t);
template void launch_cos<half>(half*, half*, size_t);
template void launch_cos<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_cos<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void acos_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(acosf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_acos(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    acos_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_acos<float>(float*, float*, size_t);
template void launch_acos<half>(half*, half*, size_t);
template void launch_acos<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_acos<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void cosh_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(coshf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_cosh(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    cosh_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_cosh<float>(float*, float*, size_t);
template void launch_cosh<half>(half*, half*, size_t);
template void launch_cosh<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_cosh<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void acosh_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(acoshf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_acosh(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    acosh_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_acosh<float>(float*, float*, size_t);
template void launch_acosh<half>(half*, half*, size_t);
template void launch_acosh<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_acosh<int32_t>(int32_t*, int32_t*, size_t);

/* TAN / ATAN */

template<typename T>
__global__ void tan_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(tanf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_tan(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    tan_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_tan<float>(float*, float*, size_t);
template void launch_tan<half>(half*, half*, size_t);
template void launch_tan<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_tan<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void tanh_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(tanhf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_tanh(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    tanh_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_tanh<float>(float*, float*, size_t);
template void launch_tanh<half>(half*, half*, size_t);
template void launch_tanh<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_tanh<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void atan_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(atanf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_atan(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    atan_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_atan<float>(float*, float*, size_t);
template void launch_atan<half>(half*, half*, size_t);
template void launch_atan<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_atan<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void atanh_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(atanhf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_atanh(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    atanh_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_atanh<float>(float*, float*, size_t);
template void launch_atanh<half>(half*, half*, size_t);
template void launch_atanh<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_atanh<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void atan2_kernel(T* y, T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(atan2f(
        static_cast<float>(y[idx]),
        static_cast<float>(x[idx])));
}

template<typename T>
void launch_atan2(T* y, T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    atan2_kernel<T><<<grid, block>>>(y, x, out, n_elements);
}

template void launch_atan2<float>(float*, float*, float*, size_t);
template void launch_atan2<half>(half*, half*, half*, size_t);
template void launch_atan2<uint8_t>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_atan2<int32_t>(int32_t*, int32_t*, int32_t*, size_t);

/* POW */

template<typename T>
__global__ void pow_kernel(T* base, T* out, float exponent, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(powf(static_cast<float>(base[idx]), exponent));
}

template<typename T>
void launch_pow(T* base, T* out, float exponent, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    pow_kernel<T><<<grid, block>>>(base, out, exponent, n_elements);
}

template void launch_pow<float>(float*, float*, float, size_t);
template void launch_pow<half>(half*, half*, float, size_t);
template void launch_pow<uint8_t>(uint8_t*, uint8_t*, float, size_t);
template void launch_pow<int32_t>(int32_t*, int32_t*, float, size_t);

/* ABS */

template<typename T>
__global__ void abs_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(fabsf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_abs(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    abs_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_abs<float>(float*, float*, size_t);
template void launch_abs<half>(half*, half*, size_t);
template void launch_abs<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_abs<int32_t>(int32_t*, int32_t*, size_t);

/* ROUNDING */

template<typename T>
__global__ void floor_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(floorf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_floor(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    floor_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_floor<float>(float*, float*, size_t);
template void launch_floor<half>(half*, half*, size_t);
template void launch_floor<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_floor<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void ceil_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(ceilf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_ceil(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    ceil_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_ceil<float>(float*, float*, size_t);
template void launch_ceil<half>(half*, half*, size_t);
template void launch_ceil<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_ceil<int32_t>(int32_t*, int32_t*, size_t);

template<typename T>
__global__ void round_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(roundf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_round(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    round_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_round<float>(float*, float*, size_t);
template void launch_round<half>(half*, half*, size_t);
template void launch_round<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_round<int32_t>(int32_t*, int32_t*, size_t);

/* MODULO */

// template<typename T>
// __global__ void mod_kernel(T* x, T* y, T* out, size_t n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= n) return;
//     out[idx] = x[idx] % y[idx];
// }

// template<typename T>
// void launch_mod(T* x, T* y, T* out, size_t n_elements) {
//     int block = BLOCK_SIZE_1D;
//     int grid = (n_elements + block - 1) / block;
//     mod_kernel<T><<<grid, block>>>(x, y, out, n_elements);
// }

// template void launch_mod<float>(float*, float*, float*, size_t);
// template void launch_mod<half>(half*, half*, half*, size_t);
// template void launch_mod<uint8_t>(uint8_t*, uint8_t*, uint8_t*, size_t);
// template void launch_mod<int32_t>(int32_t*, int32_t*, int32_t*, size_t);

template<typename T>
__global__ void fmod_kernel(T* x, T* y, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(fmodf(
        static_cast<float>(x[idx]),
        static_cast<float>(y[idx])));
}

template<typename T>
void launch_fmod(T* x, T* y, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    fmod_kernel<T><<<grid, block>>>(x, y, out, n_elements);
}

template void launch_fmod<float>(float*, float*, float*, size_t);
template void launch_fmod<half>(half*, half*, half*, size_t);
template void launch_fmod<uint8_t>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_fmod<int32_t>(int32_t*, int32_t*, int32_t*, size_t);

/* MIN / MAX */

template<typename T>
__global__ void min_kernel(T* x, T* y, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(fminf(
        static_cast<float>(x[idx]),
        static_cast<float>(y[idx])));
}

template<typename T>
void launch_min(T* x, T* y, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    min_kernel<T><<<grid, block>>>(x, y, out, n_elements);
}

template void launch_min<float>(float*, float*, float*, size_t);
template void launch_min<half>(half*, half*, half*, size_t);
template void launch_min<uint8_t>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_min<int32_t>(int32_t*, int32_t*, int32_t*, size_t);

template<typename T>
__global__ void max_kernel(T* x, T* y, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(fmaxf(
        static_cast<float>(x[idx]),
        static_cast<float>(y[idx])));
}

template<typename T>
void launch_max(T* x, T* y, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    max_kernel<T><<<grid, block>>>(x, y, out, n_elements);
}

template void launch_max<float>(float*, float*, float*, size_t);
template void launch_max<half>(half*, half*, half*, size_t);
template void launch_max<uint8_t>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_max<int32_t>(int32_t*, int32_t*, int32_t*, size_t);

/* COPYSIGN */

template<typename T>
__global__ void copysign_kernel(T* x, T* y, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(copysignf(
        static_cast<float>(x[idx]),
        static_cast<float>(y[idx])));
}

template<typename T>
void launch_copysign(T* x, T* y, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    copysign_kernel<T><<<grid, block>>>(x, y, out, n_elements);
}

template void launch_copysign<float>(float*, float*, float*, size_t);
template void launch_copysign<half>(half*, half*, half*, size_t);
template void launch_copysign<uint8_t>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_copysign<int32_t>(int32_t*, int32_t*, int32_t*, size_t);

/* TRUNCATE */

template<typename T>
__global__ void trunc_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<T>(truncf(static_cast<float>(x[idx])));
}

template<typename T>
void launch_trunc(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    trunc_kernel<T><<<grid, block>>>(x, out, n_elements);
}

template void launch_trunc<float>(float*, float*, size_t);
template void launch_trunc<half>(half*, half*, size_t);
template void launch_trunc<uint8_t>(uint8_t*, uint8_t*, size_t);
template void launch_trunc<int32_t>(int32_t*, int32_t*, size_t);

