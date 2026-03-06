#include "common.h"
#include "ops/policies/elementwise.h"

/* COMPARISON OPERATORS */

template<typename T, template<typename> class Op>
__global__ void elementwise_compare_kernel(T* x, T* y, bool* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = Op<T>::operation(x[idx], y[idx]);
}

template<typename T, template<typename> class Op>
void launch_elementwise_compare(T* x, T* y, bool* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    elementwise_compare_kernel<T, Op><<<grid, block>>>(x, y, out, n_elements);
}

template void launch_elementwise_compare<  float, ElemWiseEqOp>(float*, float*, bool*, size_t);
template void launch_elementwise_compare<   half, ElemWiseEqOp>(half*, half*, bool*, size_t);
template void launch_elementwise_compare<uint8_t, ElemWiseEqOp>(uint8_t*, uint8_t*, bool*, size_t);
template void launch_elementwise_compare<int32_t, ElemWiseEqOp>(int32_t*, int32_t*, bool*, size_t);

template void launch_elementwise_compare<  float, ElemWiseLtOp>(float*, float*, bool*, size_t);
template void launch_elementwise_compare<   half, ElemWiseLtOp>(half*, half*, bool*, size_t);
template void launch_elementwise_compare<uint8_t, ElemWiseLtOp>(uint8_t*, uint8_t*, bool*, size_t);
template void launch_elementwise_compare<int32_t, ElemWiseLtOp>(int32_t*, int32_t*, bool*, size_t);

template void launch_elementwise_compare<  float, ElemWiseLeOp>(float*, float*, bool*, size_t);
template void launch_elementwise_compare<   half, ElemWiseLeOp>(half*, half*, bool*, size_t);
template void launch_elementwise_compare<uint8_t, ElemWiseLeOp>(uint8_t*, uint8_t*, bool*, size_t);
template void launch_elementwise_compare<int32_t, ElemWiseLeOp>(int32_t*, int32_t*, bool*, size_t);

template void launch_elementwise_compare<  float, ElemWiseGtOp>(float*, float*, bool*, size_t);
template void launch_elementwise_compare<   half, ElemWiseGtOp>(half*, half*, bool*, size_t);
template void launch_elementwise_compare<uint8_t, ElemWiseGtOp>(uint8_t*, uint8_t*, bool*, size_t);
template void launch_elementwise_compare<int32_t, ElemWiseGtOp>(int32_t*, int32_t*, bool*, size_t);

template void launch_elementwise_compare<  float, ElemWiseGeOp>(float*, float*, bool*, size_t);
template void launch_elementwise_compare<   half, ElemWiseGeOp>(half*, half*, bool*, size_t);
template void launch_elementwise_compare<uint8_t, ElemWiseGeOp>(uint8_t*, uint8_t*, bool*, size_t);
template void launch_elementwise_compare<int32_t, ElemWiseGeOp>(int32_t*, int32_t*, bool*, size_t);

/* MATH OPERATORS (2 TENSOR) */

template<typename T, template<typename> class Op>
__global__ void elementwise_math_2tensor_kernel(T* x, T* y, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = Op<T>::operation(x[idx], y[idx]);
}

template<typename T, template<typename> class Op>
void launch_elementwise_math_2tensor(T* x, T* y, T* out, size_t n_elements) {
    int block = BLOCK_SIZE_1D;
    int grid = (n_elements + block - 1) / block;
    elementwise_math_2tensor_kernel<T, Op><<<grid, block>>>(x, y, out, n_elements);
}

template void launch_elementwise_math_2tensor<  float, ElemWiseAddOp>(float*, float*, float*, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseAddOp>(half*, half*, half*, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseAddOp>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseAddOp>(int32_t*, int32_t*, int32_t*, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseSubOp>(float*, float*, float*, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseSubOp>(half*, half*, half*, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseSubOp>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseSubOp>(int32_t*, int32_t*, int32_t*, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseMulOp>(float*, float*, float*, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseMulOp>(half*, half*, half*, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseMulOp>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseMulOp>(int32_t*, int32_t*, int32_t*, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseDivOp>(float*, float*, float*, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseDivOp>(half*, half*, half*, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseDivOp>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseDivOp>(int32_t*, int32_t*, int32_t*, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseAtan2Op>(float*, float*, float*, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseAtan2Op>(half*, half*, half*, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseAtan2Op>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseAtan2Op>(int32_t*, int32_t*, int32_t*, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseFModOp>(float*, float*, float*, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseFModOp>(half*, half*, half*, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseFModOp>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseFModOp>(int32_t*, int32_t*, int32_t*, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseMinOp>(float*, float*, float*, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseMinOp>(half*, half*, half*, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseMinOp>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseMinOp>(int32_t*, int32_t*, int32_t*, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseMaxOp>(float*, float*, float*, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseMaxOp>(half*, half*, half*, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseMaxOp>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseMaxOp>(int32_t*, int32_t*, int32_t*, size_t);

template void launch_elementwise_math_2tensor<  float, ElemWiseCopysignOp>(float*, float*, float*, size_t);
template void launch_elementwise_math_2tensor<   half, ElemWiseCopysignOp>(half*, half*, half*, size_t);
template void launch_elementwise_math_2tensor<uint8_t, ElemWiseCopysignOp>(uint8_t*, uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_2tensor<int32_t, ElemWiseCopysignOp>(int32_t*, int32_t*, int32_t*, size_t);

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

template void launch_elementwise_math_1tensor<  float, ElemWiseTahnOp>(float*, float*, size_t);
template void launch_elementwise_math_1tensor<   half, ElemWiseTahnOp>(half*, half*, size_t);
template void launch_elementwise_math_1tensor<uint8_t, ElemWiseTahnOp>(uint8_t*, uint8_t*, size_t);
template void launch_elementwise_math_1tensor<int32_t, ElemWiseTahnOp>(int32_t*, int32_t*, size_t);

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

