#include "kernels/common.h"
#include "common/functions.h"

__device__ int nearest_1d_in_idx(
    int idx, int C, int L_in, int L_out
) {
    int l_out = idx % L_out;
    int c     = (idx / L_out) % C;
    int b     =  idx / (L_out * C);
    int l_in  = min((int)(l_out * L_in / L_out), L_in - 1);
    return b * (C * L_in) + c * L_in + l_in;
}

template<typename T>
__global__ void upsample_nearest_1d_kernel(
    T* input, T* output,
    int B, int C, int L_in, int L_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * L_out) return;
    output[idx] = input[nearest_1d_in_idx(idx, C, L_in, L_out)];
}

template<typename T>
void launch_upsample_nearest_1d(
    T* input, T* output,
    int B, int C, int L_in, int L_out
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * L_out + threads - 1) / threads;
    upsample_nearest_1d_kernel<T><<<blocks, threads>>>(
        input, output, B, C, L_in, L_out);
}

template void launch_upsample_nearest_1d<float>(float*, float*, int, int, int, int);
template void launch_upsample_nearest_1d<half>(half*, half*, int, int, int, int);
template void launch_upsample_nearest_1d<uint8_t>(uint8_t*, uint8_t*, int, int, int, int);
template void launch_upsample_nearest_1d<int32_t>(int32_t*, int32_t*, int, int, int, int);

template<typename T>
__global__ void upsample_nearest_1d_backward_kernel(
    T* grad_output, T* grad_input,
    int B, int C, int L_in, int L_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * L_out) return;
    int in_idx = nearest_1d_in_idx(idx, C, L_in, L_out);
    atomic_add<T>(&grad_input[in_idx], grad_output[idx]);
}

template<typename T>
void launch_upsample_nearest_1d_backward(
    T* grad_output, T* grad_input,
    int B, int C, int L_in, int L_out
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * L_out + threads - 1) / threads;
    upsample_nearest_1d_backward_kernel<T><<<blocks, threads>>>(
        grad_output, grad_input, B, C, L_in, L_out);
}

template void launch_upsample_nearest_1d_backward<float>(float*, float*, int, int, int, int);
template void launch_upsample_nearest_1d_backward<half>(half*, half*, int, int, int, int);
template void launch_upsample_nearest_1d_backward<uint8_t>(uint8_t*, uint8_t*, int, int, int, int);
template void launch_upsample_nearest_1d_backward<int32_t>(int32_t*, int32_t*, int, int, int, int);

__device__ int nearest_2d_in_idx(
    int idx, int C, int H_in, int W_in, int H_out, int W_out
) {
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c     = (idx / (W_out * H_out)) % C;
    int b     =  idx / (W_out * H_out * C);

    int h_in = min((int)(h_out * H_in / H_out), H_in - 1);
    int w_in = min((int)(w_out * W_in / W_out), W_in - 1);
    return b * (C * H_in * W_in) 
         + c * (H_in * W_in) 
         + h_in * W_in + w_in;
}

template<typename T>
__global__ void upsample_nearest_2d_kernel(
    T* input, T* output,
    int B, int C, int H_in, int W_in, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H_out * W_out) return;
    int in_idx = nearest_2d_in_idx(idx, C, H_in, W_in, H_out, W_out);
    output[idx] = input[in_idx];
}

template<typename T>
void launch_upsample_nearest_2d(
    T* input, T* output,
    int B, int C, int H_in, int W_in, int H_out, int W_out
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * H_out * W_out + threads - 1) / threads;
    upsample_nearest_2d_kernel<T><<<blocks, threads>>>(
        input, output, B, C, H_in, W_in, H_out, W_out);
}

template void launch_upsample_nearest_2d<float>(float*, float*, int, int, int, int, int, int);
template void launch_upsample_nearest_2d<half>(half*, half*, int, int, int, int, int, int);
template void launch_upsample_nearest_2d<uint8_t>(uint8_t*, uint8_t*, int, int, int, int, int, int);
template void launch_upsample_nearest_2d<int32_t>(int32_t*, int32_t*, int, int, int, int, int, int);

template<typename T>
__global__ void upsample_nearest_2d_backward_kernel(
    T* grad_output, T* grad_input,
    int B, int C, int H_in, int W_in, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H_out * W_out) return;
    int in_idx = nearest_2d_in_idx(idx, C, H_in, W_in, H_out, W_out);
    atomic_add<T>(&grad_input[in_idx], grad_output[idx]);
}

template<typename T>
void launch_upsample_nearest_2d_backward(
    T* grad_output, T* grad_input,
    int B, int C, int H_in, int W_in, int H_out, int W_out
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * H_out * W_out + threads - 1) / threads;
    upsample_nearest_2d_backward_kernel<T><<<blocks, threads>>>(
        grad_output, grad_input, B, C, H_in, W_in, H_out, W_out);
}

template void launch_upsample_nearest_2d_backward<float>(float*, float*, int, int, int, int, int, int);
template void launch_upsample_nearest_2d_backward<half>(half*, half*, int, int, int, int, int, int);
template void launch_upsample_nearest_2d_backward<uint8_t>(uint8_t*, uint8_t*, int, int, int, int, int, int);
template void launch_upsample_nearest_2d_backward<int32_t>(int32_t*, int32_t*, int, int, int, int, int, int);

__device__ int nearest_3d_in_idx(
    int idx, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c     = (idx / (W_out * H_out * D_out)) % C;
    int b     =  idx / (W_out * H_out * D_out * C);

    int d_in = min((int)(d_out * D_in / D_out), D_in - 1);
    int h_in = min((int)(h_out * H_in / H_out), H_in - 1);
    int w_in = min((int)(w_out * W_in / W_out), W_in - 1);
    return b * (C * D_in * H_in * W_in) 
         + c * (D_in * H_in * W_in) 
         + d_in * (H_in * W_in) 
         + h_in * W_in + w_in;
}

template<typename T>
__global__ void upsample_nearest_3d_kernel(
    T* input, T* output,
    int B, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * D_out * H_out * W_out) return;
    int in_idx = nearest_3d_in_idx(
        idx, C, D_in, H_in, W_in, D_out, H_out, W_out);
    output[idx] = input[in_idx];
}

template<typename T>
void launch_upsample_nearest_3d(
    T* input, T* output,
    int B, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * D_out * H_out * W_out + threads - 1) / threads;
    upsample_nearest_3d_kernel<T><<<blocks, threads>>>(
        input, output, B, C, D_in, H_in, W_in, D_out, H_out, W_out);
}

template void launch_upsample_nearest_3d<float>(float*, float*, int, int, int, int, int, int, int, int);
template void launch_upsample_nearest_3d<half>(half*, half*, int, int, int, int, int, int, int, int);
template void launch_upsample_nearest_3d<uint8_t>(uint8_t*, uint8_t*, int, int, int, int, int, int, int, int);
template void launch_upsample_nearest_3d<int32_t>(int32_t*, int32_t*, int, int, int, int, int, int, int, int);

template<typename T>
__global__ void upsample_nearest_3d_backward_kernel(
    T* grad_output, T* grad_input,
    int B, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * D_out * H_out * W_out) return;
    int in_idx = nearest_3d_in_idx(
        idx, C, D_in, H_in, W_in, D_out, H_out, W_out);
    atomic_add<T>(&grad_input[in_idx], grad_output[idx]);
}

template<typename T>
void launch_upsample_nearest_3d_backward(
    T* grad_output, T* grad_input,
    int B, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * D_out * H_out * W_out + threads - 1) / threads;
    upsample_nearest_3d_backward_kernel<T><<<blocks, threads>>>(
        grad_output, grad_input, B, C, D_in, H_in, W_in, D_out, H_out, W_out);
}

template void launch_upsample_nearest_3d_backward<float>(float*, float*, int, int, int, int, int, int, int, int);
template void launch_upsample_nearest_3d_backward<half>(half*, half*, int, int, int, int, int, int, int, int);
template void launch_upsample_nearest_3d_backward<uint8_t>(uint8_t*, uint8_t*, int, int, int, int, int, int, int, int);
template void launch_upsample_nearest_3d_backward<int32_t>(int32_t*, int32_t*, int, int, int, int, int, int, int, int);

