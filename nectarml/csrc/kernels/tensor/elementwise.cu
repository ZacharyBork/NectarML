#include "kernels/common.h"
#include "common/data_structures.h"
#include "ops/policies/elementwise.h"

/* COMPARISON TENSOR-TENSOR */

template<typename T, template<typename> class Op>
__global__ void elementwise_compare_kernel(
    T* x, T* y, bool* out, 
    BroadcastIndex x_index, BroadcastIndex y_index,
    ShapeArray out_shape, int ndim,
    size_t n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int coords[MAX_DIMS];
    int remaining = idx;
    for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = remaining % out_shape.dims[d];
        remaining /= out_shape.dims[d];
    }

    int x_flat = x_index.get_flat(coords);
    int y_flat = y_index.get_flat(coords);

    out[idx] = Op<T>::operation(x[x_flat], y[y_flat]);
}

template<typename T, template<typename> class Op>
void launch_elementwise_compare(
    T* x, T* y, bool* out, 
    BroadcastIndex x_index, BroadcastIndex y_index,
    ShapeArray out_shape, int ndim,
    size_t n_elements
) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;

    elementwise_compare_kernel<T, Op><<<grid, block>>>(
        x, y, out, x_index, y_index, 
        out_shape, ndim, n_elements);
}

template void launch_elementwise_compare<  float, ElemWiseEqOp>(
    float*, float*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<   half, ElemWiseEqOp>(
    half*, half*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<uint8_t, ElemWiseEqOp>(
    uint8_t*, uint8_t*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<int32_t, ElemWiseEqOp>(
    int32_t*, int32_t*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

    template void launch_elementwise_compare<  float, ElemWiseNeOp>(
    float*, float*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<   half, ElemWiseNeOp>(
    half*, half*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<uint8_t, ElemWiseNeOp>(
    uint8_t*, uint8_t*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<int32_t, ElemWiseNeOp>(
    int32_t*, int32_t*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_compare<  float, ElemWiseLtOp>(
    float*, float*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<   half, ElemWiseLtOp>(
    half*, half*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<uint8_t, ElemWiseLtOp>(
    uint8_t*, uint8_t*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<int32_t, ElemWiseLtOp>(
    int32_t*, int32_t*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_compare<  float, ElemWiseLeOp>(
    float*, float*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<   half, ElemWiseLeOp>(
    half*, half*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<uint8_t, ElemWiseLeOp>(
    uint8_t*, uint8_t*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<int32_t, ElemWiseLeOp>(
    int32_t*, int32_t*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_compare<  float, ElemWiseGtOp>(
    float*, float*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<   half, ElemWiseGtOp>(
    half*, half*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<uint8_t, ElemWiseGtOp>(
    uint8_t*, uint8_t*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<int32_t, ElemWiseGtOp>(
    int32_t*, int32_t*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_compare<  float, ElemWiseGeOp>(
    float*, float*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<   half, ElemWiseGeOp>(
    half*, half*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<uint8_t, ElemWiseGeOp>(
    uint8_t*, uint8_t*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_compare<int32_t, ElemWiseGeOp>(
    int32_t*, int32_t*, bool*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

/* COMPARISON TENSOR-SCALAR */

template<typename T, template<typename> class Op>
__global__ void elementwise_compare_ts_kernel(T* x, float value, bool* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = Op<T>::operation(x[idx], value);
}

template<typename T, template<typename> class Op>
void launch_elementwise_compare_ts(T* x, float value, bool* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    elementwise_compare_ts_kernel<T, Op><<<grid, block>>>(x, value, out, n_elements);
}

template void launch_elementwise_compare_ts<  float, ElemWiseEqTSOp>(float*, float, bool*, size_t);
template void launch_elementwise_compare_ts<   half, ElemWiseEqTSOp>(half*, float, bool*, size_t);
template void launch_elementwise_compare_ts<uint8_t, ElemWiseEqTSOp>(uint8_t*, float, bool*, size_t);
template void launch_elementwise_compare_ts<int32_t, ElemWiseEqTSOp>(int32_t*, float, bool*, size_t);

template void launch_elementwise_compare_ts<  float, ElemWiseNeTSOp>(float*, float, bool*, size_t);
template void launch_elementwise_compare_ts<   half, ElemWiseNeTSOp>(half*, float, bool*, size_t);
template void launch_elementwise_compare_ts<uint8_t, ElemWiseNeTSOp>(uint8_t*, float, bool*, size_t);
template void launch_elementwise_compare_ts<int32_t, ElemWiseNeTSOp>(int32_t*, float, bool*, size_t);

template void launch_elementwise_compare_ts<  float, ElemWiseLtTSOp>(float*, float, bool*, size_t);
template void launch_elementwise_compare_ts<   half, ElemWiseLtTSOp>(half*, float, bool*, size_t);
template void launch_elementwise_compare_ts<uint8_t, ElemWiseLtTSOp>(uint8_t*, float, bool*, size_t);
template void launch_elementwise_compare_ts<int32_t, ElemWiseLtTSOp>(int32_t*, float, bool*, size_t);

template void launch_elementwise_compare_ts<  float, ElemWiseLeTSOp>(float*, float, bool*, size_t);
template void launch_elementwise_compare_ts<   half, ElemWiseLeTSOp>(half*, float, bool*, size_t);
template void launch_elementwise_compare_ts<uint8_t, ElemWiseLeTSOp>(uint8_t*, float, bool*, size_t);
template void launch_elementwise_compare_ts<int32_t, ElemWiseLeTSOp>(int32_t*, float, bool*, size_t);

template void launch_elementwise_compare_ts<  float, ElemWiseGtTSOp>(float*, float, bool*, size_t);
template void launch_elementwise_compare_ts<   half, ElemWiseGtTSOp>(half*, float, bool*, size_t);
template void launch_elementwise_compare_ts<uint8_t, ElemWiseGtTSOp>(uint8_t*, float, bool*, size_t);
template void launch_elementwise_compare_ts<int32_t, ElemWiseGtTSOp>(int32_t*, float, bool*, size_t);

template void launch_elementwise_compare_ts<  float, ElemWiseGeTSOp>(float*, float, bool*, size_t);
template void launch_elementwise_compare_ts<   half, ElemWiseGeTSOp>(half*, float, bool*, size_t);
template void launch_elementwise_compare_ts<uint8_t, ElemWiseGeTSOp>(uint8_t*, float, bool*, size_t);
template void launch_elementwise_compare_ts<int32_t, ElemWiseGeTSOp>(int32_t*, float, bool*, size_t);

/* MATH OPERATORS (2 TENSOR) */

template<typename T, template<typename> class Op>
__global__ void elementwise_math_2tensor_kernel(
    T* x, T* y, T* out, 
    BroadcastIndex x_index, BroadcastIndex y_index,
    ShapeArray out_shape, int ndim,
    size_t n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int coords[MAX_DIMS];
    int remaining = idx;
    for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = remaining % out_shape.dims[d];
        remaining /= out_shape.dims[d];
    }

    int x_flat = x_index.get_flat(coords);
    int y_flat = y_index.get_flat(coords);

    out[idx] = Op<T>::operation(x[x_flat], y[y_flat]);
}

template<typename T, template<typename> class Op>
void launch_elementwise_math_2tensor(
    T* x, T* y, T* out, 
    BroadcastIndex x_index, BroadcastIndex y_index,
    ShapeArray out_shape, int ndim,
    size_t n_elements
) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;

    elementwise_math_2tensor_kernel<T, Op><<<grid, block>>>(
        x, y, out, x_index, y_index, 
        out_shape, ndim, n_elements);
}

template void launch_elementwise_math_2tensor<  float, ElemWiseAddOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseAddOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseAddOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseAddOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseSubOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseSubOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseSubOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseSubOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseMulOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseMulOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseMulOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseMulOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseDivOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseDivOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseDivOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseDivOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseAtan2Op>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseAtan2Op>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseAtan2Op>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseAtan2Op>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseFModOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseFModOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseFModOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseFModOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseMinOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseMinOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseMinOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseMinOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseMaxOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseMaxOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseMaxOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseMaxOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseCopysignOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseCopysignOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseCopysignOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseCopysignOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseTensorEqMaskOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseTensorEqMaskOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseTensorEqMaskOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseTensorEqMaskOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseTensorNeMaskOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseTensorNeMaskOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseTensorNeMaskOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseTensorNeMaskOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseTensorLtMaskOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseTensorLtMaskOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseTensorLtMaskOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseTensorLtMaskOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseTensorLeMaskOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseTensorLeMaskOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseTensorLeMaskOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseTensorLeMaskOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseTensorGtMaskOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseTensorGtMaskOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseTensorGtMaskOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseTensorGtMaskOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseTensorGeMaskOp>(
    float*, float*, float*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseTensorGeMaskOp>(
    half*, half*, half*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseTensorGeMaskOp>(
    uint8_t*, uint8_t*, uint8_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseTensorGeMaskOp>(
    int32_t*, int32_t*, int32_t*, BroadcastIndex, BroadcastIndex, ShapeArray, int, size_t);

/* MATH OPERATORS (1 TENSOR) */

template<typename T, template<typename> class Op>
__global__ void elementwise_math_1tensor_kernel(T* x, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = Op<T>::operation(x[idx]);
}

template<typename T, template<typename> class Op>
void launch_elementwise_math_1tensor(T* x, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    elementwise_math_1tensor_kernel<T, Op><<<grid, block>>>(x, out, n_elements);
}

template void launch_elementwise_math_1tensor<  float, ElemWiseSignOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseSignOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseSignOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseSignOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseNegateOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseNegateOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseNegateOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseNegateOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseSqrtOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseSqrtOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseSqrtOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseSqrtOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseRSqrtOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseRSqrtOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseRSqrtOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseRSqrtOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseExpOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseExpOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseExpOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseExpOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseLogOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseLogOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseLogOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseLogOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseLog2Op>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseLog2Op>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseLog2Op>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseLog2Op>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseLog10Op>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseLog10Op>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseLog10Op>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseLog10Op>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseSinOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseSinOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseSinOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseSinOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseAsinOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseAsinOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseAsinOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseAsinOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseSinhOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseSinhOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseSinhOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseSinhOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseAsinhOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseAsinhOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseAsinhOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseAsinhOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseCosOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseCosOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseCosOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseCosOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseAcosOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseAcosOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseAcosOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseAcosOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseCoshOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseCoshOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseCoshOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseCoshOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseAcoshOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseAcoshOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseAcoshOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseAcoshOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseTanOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseTanOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseTanOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseTanOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseTanhOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseTanhOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseTanhOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseTanhOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseAtanOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseAtanOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseAtanOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseAtanOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseAtanhOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseAtanhOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseAtanhOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseAtanhOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseAbsOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseAbsOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseAbsOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseAbsOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseFloorOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseFloorOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseFloorOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseFloorOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseCeilOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseCeilOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseCeilOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseCeilOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseRoundOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseRoundOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseRoundOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseRoundOp>(int32_t*, int32_t*, size_t);

template void launch_elementwise_math_1tensor<  float, ElemWiseTruncOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseTruncOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseTruncOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseTruncOp>(int32_t*, int32_t*, size_t);

/* MATH OPERATIONS (TENSOR/SCALAR) */

template<typename T, template<typename> class Op>
__global__ void elementwise_math_tensorscalar_kernel(T* x, T* out, float value, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = Op<T>::operation(x[idx], value);
}

template<typename T, template<typename> class Op>
void launch_elementwise_math_tensorscalar(T* x, T* out, float value, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    elementwise_math_tensorscalar_kernel<T, Op><<<grid, block>>>(x, out, value, n_elements);
}

template void launch_elementwise_math_tensorscalar<  float, ElemWiseAddTSOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseAddTSOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseAddTSOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseAddTSOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseSubTSOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseSubTSOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseSubTSOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseSubTSOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseMulTSOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseMulTSOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseMulTSOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseMulTSOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseDivTSOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseDivTSOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseDivTSOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseDivTSOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseFmodfTSOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseFmodfTSOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseFmodfTSOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseFmodfTSOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseMinfTSOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseMinfTSOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseMinfTSOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseMinfTSOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseMaxfTSOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseMaxfTSOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseMaxfTSOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseMaxfTSOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWisePowOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWisePowOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWisePowOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWisePowOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseScalarAddOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseScalarAddOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseScalarAddOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseScalarAddOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseScalarSubOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseScalarSubOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseScalarSubOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseScalarSubOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseScalarMulOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseScalarMulOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseScalarMulOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseScalarMulOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseScalarDivOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseScalarDivOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseScalarDivOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseScalarDivOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseScalarMinOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseScalarMinOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseScalarMinOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseScalarMinOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseScalarMaxOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseScalarMaxOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseScalarMaxOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseScalarMaxOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseScalarEqMaskOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseScalarEqMaskOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseScalarEqMaskOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseScalarEqMaskOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseScalarNeMaskOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseScalarNeMaskOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseScalarNeMaskOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseScalarNeMaskOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseScalarLtMaskOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseScalarLtMaskOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseScalarLtMaskOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseScalarLtMaskOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseScalarLeMaskOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseScalarLeMaskOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseScalarLeMaskOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseScalarLeMaskOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseScalarGtMaskOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseScalarGtMaskOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseScalarGtMaskOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseScalarGtMaskOp>(int32_t*, int32_t*, float, size_t);

template void launch_elementwise_math_tensorscalar<  float, ElemWiseScalarGeMaskOp>(float*, float*, float, size_t);
template void launch_elementwise_math_tensorscalar<   half, ElemWiseScalarGeMaskOp>(half*, half*, float, size_t);
template void launch_elementwise_math_tensorscalar<uint8_t, ElemWiseScalarGeMaskOp>(uint8_t*, uint8_t*, float, size_t);
template void launch_elementwise_math_tensorscalar<int32_t, ElemWiseScalarGeMaskOp>(int32_t*, int32_t*, float, size_t);

