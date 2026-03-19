#include "common.h"

template<typename T>
__device__ int linear_1d_low(int l_out, int L_in, int L_out) {
    float l_in = (l_out + 0.5f) * (float)L_in / L_out - 0.5f;
    return max(0, min((int)floorf(l_in), L_in - 1));
}

template<typename T>
__global__ void upsample_linear_1d_kernel(
    T* input, T* output,
    int B, int C, int L_in, int L_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * L_out) return;

    int l_out = idx % L_out;
    int c     = (idx / L_out) % C;
    int b     =  idx / (L_out * C);

    float l_in_float = l_out * (float)L_in / L_out;
    int l_low  = max(0, min((int)floorf(l_in_float), L_in - 1));
    int l_high = max(0, min(l_low + 1, L_in - 1));
    float wt_high = l_in_float - floorf(l_in_float);
    float wt_low  = 1.0f - wt_high;

    int in_low  = b * (C * L_in) + c * L_in + l_low;
    int in_high = b * (C * L_in) + c * L_in + l_high;

    output[idx] = static_cast<T>(
        wt_low  * static_cast<float>(input[in_low]) +
        wt_high * static_cast<float>(input[in_high]));
}

template<typename T>
void launch_upsample_linear_1d(
    T* input, T* output,
    int B, int C, int L_in, int L_out
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * L_out + threads - 1) / threads;
    upsample_linear_1d_kernel<T><<<blocks, threads>>>(
        input, output, B, C, L_in, L_out);
}

template void launch_upsample_linear_1d<float>(float*, float*, int, int, int, int);
template void launch_upsample_linear_1d<half>(half*, half*, int, int, int, int);
template void launch_upsample_linear_1d<uint8_t>(uint8_t*, uint8_t*, int, int, int, int);
template void launch_upsample_linear_1d<int32_t>(int32_t*, int32_t*, int, int, int, int);

template<typename T>
__global__ void upsample_linear_1d_backward_kernel(
    T* grad_output, T* grad_input,
    int B, int C, int L_in, int L_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * L_out) return;

    int l_out = idx % L_out;
    int c     = (idx / L_out) % C;
    int b     =  idx / (L_out * C);

    float l_in_float = l_out * (float)L_in / L_out;
    int l_low  = max(0, min((int)floorf(l_in_float), L_in - 1));
    int l_high = max(0, min(l_low + 1, L_in - 1));
    float wt_high = l_in_float - floorf(l_in_float);
    float wt_low  = 1.0f - wt_high;

    int in_low  = b * (C * L_in) + c * L_in + l_low;
    int in_high = b * (C * L_in) + c * L_in + l_high;

    T grad = grad_output[idx];
    atomic_add<T>(&grad_input[in_low],  
                  static_cast<T>(wt_low  * static_cast<float>(grad)));
    atomic_add<T>(&grad_input[in_high], 
                  static_cast<T>(wt_high * static_cast<float>(grad)));
}

template<typename T>
void launch_upsample_linear_1d_backward(
    T* grad_output, T* grad_input,
    int B, int C, int L_in, int L_out
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * L_out + threads - 1) / threads;
    upsample_linear_1d_backward_kernel<T><<<blocks, threads>>>(
        grad_output, grad_input, B, C, L_in, L_out);
}

template void launch_upsample_linear_1d_backward<float>(float*, float*, int, int, int, int);
template void launch_upsample_linear_1d_backward<half>(half*, half*, int, int, int, int);
template void launch_upsample_linear_1d_backward<uint8_t>(uint8_t*, uint8_t*, int, int, int, int);
template void launch_upsample_linear_1d_backward<int32_t>(int32_t*, int32_t*, int, int, int, int);

template<typename T>
__global__ void upsample_linear_2d_kernel(
    T* input, T* output,
    int B, int C, int H_in, int W_in, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c     = (idx / (W_out * H_out)) % C;
    int b     =  idx / (W_out * H_out * C);

    float h_in_float = h_out * (float)H_in / H_out;
    float w_in_float = w_out * (float)W_in / W_out;

    int h_low  = max(0, min((int)floorf(h_in_float), H_in - 1));
    int h_high = max(0, min(h_low + 1, H_in - 1));
    int w_low  = max(0, min((int)floorf(w_in_float), W_in - 1));
    int w_high = max(0, min(w_low + 1, W_in - 1));

    float wt_h_high = h_in_float - floorf(h_in_float);
    float wt_h_low  = 1.0f - wt_h_high;
    float wt_w_high = w_in_float - floorf(w_in_float);
    float wt_w_low  = 1.0f - wt_w_high;

    int base = b * (C * H_in * W_in) + c * (H_in * W_in);

    float result =
        wt_h_low  * wt_w_low  * static_cast<float>(input[base + h_low  * W_in + w_low ]) +
        wt_h_low  * wt_w_high * static_cast<float>(input[base + h_low  * W_in + w_high]) +
        wt_h_high * wt_w_low  * static_cast<float>(input[base + h_high * W_in + w_low ]) +
        wt_h_high * wt_w_high * static_cast<float>(input[base + h_high * W_in + w_high]);

    output[idx] = static_cast<T>(result);
}

template<typename T>
void launch_upsample_linear_2d(
    T* input, T* output,
    int B, int C, int H_in, int W_in, int H_out, int W_out
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * H_out * W_out + threads - 1) / threads;
    upsample_linear_2d_kernel<T><<<blocks, threads>>>(
        input, output, B, C, H_in, W_in, H_out, W_out);
}

template void launch_upsample_linear_2d<float>(float*, float*, int, int, int, int, int, int);
template void launch_upsample_linear_2d<half>(half*, half*, int, int, int, int, int, int);
template void launch_upsample_linear_2d<uint8_t>(uint8_t*, uint8_t*, int, int, int, int, int, int);
template void launch_upsample_linear_2d<int32_t>(int32_t*, int32_t*, int, int, int, int, int, int);

template<typename T>
__global__ void upsample_linear_2d_backward_kernel(
    T* grad_output, T* grad_input,
    int B, int C, int H_in, int W_in, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c     = (idx / (W_out * H_out)) % C;
    int b     =  idx / (W_out * H_out * C);

    float h_in_float = h_out * (float)H_in / H_out;
    float w_in_float = w_out * (float)W_in / W_out;

    int h_low  = max(0, min((int)floorf(h_in_float), H_in - 1));
    int h_high = max(0, min(h_low + 1, H_in - 1));
    int w_low  = max(0, min((int)floorf(w_in_float), W_in - 1));
    int w_high = max(0, min(w_low + 1, W_in - 1));

    float wt_h_high = h_in_float - floorf(h_in_float);
    float wt_h_low  = 1.0f - wt_h_high;
    float wt_w_high = w_in_float - floorf(w_in_float);
    float wt_w_low  = 1.0f - wt_w_high;

    float grad = static_cast<float>(grad_output[idx]);
    int base = b * (C * H_in * W_in) + c * (H_in * W_in);

    atomic_add<T>(&grad_input[base + h_low  * W_in + w_low ],  
                  static_cast<T>(wt_h_low  * wt_w_low  * grad));
    atomic_add<T>(&grad_input[base + h_low  * W_in + w_high],  
                  static_cast<T>(wt_h_low  * wt_w_high * grad));
    atomic_add<T>(&grad_input[base + h_high * W_in + w_low ],  
                  static_cast<T>(wt_h_high * wt_w_low  * grad));
    atomic_add<T>(&grad_input[base + h_high * W_in + w_high], 
                  static_cast<T>(wt_h_high * wt_w_high * grad));
}

template<typename T>
void launch_upsample_linear_2d_backward(
    T* grad_output, T* grad_input,
    int B, int C, int H_in, int W_in, int H_out, int W_out
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * H_out * W_out + threads - 1) / threads;
    upsample_linear_2d_backward_kernel<T><<<blocks, threads>>>(
        grad_output, grad_input, B, C, H_in, W_in, H_out, W_out);
}

template void launch_upsample_linear_2d_backward<float>(float*, float*, int, int, int, int, int, int);
template void launch_upsample_linear_2d_backward<half>(half*, half*, int, int, int, int, int, int);
template void launch_upsample_linear_2d_backward<uint8_t>(uint8_t*, uint8_t*, int, int, int, int, int, int);
template void launch_upsample_linear_2d_backward<int32_t>(int32_t*, int32_t*, int, int, int, int, int, int);

template<typename T>
__global__ void upsample_linear_3d_kernel(
    T* input, T* output,
    int B, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * D_out * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c     = (idx / (W_out * H_out * D_out)) % C;
    int b     =  idx / (W_out * H_out * D_out * C);

    float d_in_float = d_out * (float)D_in / D_out;
    float h_in_float = h_out * (float)H_in / H_out;
    float w_in_float = w_out * (float)W_in / W_out;

    int d_low  = max(0, min((int)floorf(d_in_float), D_in - 1));
    int d_high = max(0, min(d_low + 1, D_in - 1));
    int h_low  = max(0, min((int)floorf(h_in_float), H_in - 1));
    int h_high = max(0, min(h_low + 1, H_in - 1));
    int w_low  = max(0, min((int)floorf(w_in_float), W_in - 1));
    int w_high = max(0, min(w_low + 1, W_in - 1));

    float wt_d_high = d_in_float - floorf(d_in_float);
    float wt_d_low  = 1.0f - wt_d_high;
    float wt_h_high = h_in_float - floorf(h_in_float);
    float wt_h_low  = 1.0f - wt_h_high;
    float wt_w_high = w_in_float - floorf(w_in_float);
    float wt_w_low  = 1.0f - wt_w_high;

    int base = b * (C * D_in * H_in * W_in) + c * (D_in * H_in * W_in);
    int hw   = H_in * W_in;

    float result =
        wt_d_low  * wt_h_low  * wt_w_low  * static_cast<float>(input[base + d_low  * hw + h_low  * W_in + w_low ]) +
        wt_d_low  * wt_h_low  * wt_w_high * static_cast<float>(input[base + d_low  * hw + h_low  * W_in + w_high]) +
        wt_d_low  * wt_h_high * wt_w_low  * static_cast<float>(input[base + d_low  * hw + h_high * W_in + w_low ]) +
        wt_d_low  * wt_h_high * wt_w_high * static_cast<float>(input[base + d_low  * hw + h_high * W_in + w_high]) +
        wt_d_high * wt_h_low  * wt_w_low  * static_cast<float>(input[base + d_high * hw + h_low  * W_in + w_low ]) +
        wt_d_high * wt_h_low  * wt_w_high * static_cast<float>(input[base + d_high * hw + h_low  * W_in + w_high]) +
        wt_d_high * wt_h_high * wt_w_low  * static_cast<float>(input[base + d_high * hw + h_high * W_in + w_low ]) +
        wt_d_high * wt_h_high * wt_w_high * static_cast<float>(input[base + d_high * hw + h_high * W_in + w_high]);

    output[idx] = static_cast<T>(result);
}

template<typename T>
void launch_upsample_linear_3d(
    T* input, T* output,
    int B, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * D_out * H_out * W_out + threads - 1) / threads;
    upsample_linear_3d_kernel<T><<<blocks, threads>>>(
        input, output, B, C, D_in, H_in, W_in, D_out, H_out, W_out);
}

template void launch_upsample_linear_3d<float>(float*, float*, int, int, int, int, int, int, int, int);
template void launch_upsample_linear_3d<half>(half*, half*, int, int, int, int, int, int, int, int);
template void launch_upsample_linear_3d<uint8_t>(uint8_t*, uint8_t*, int, int, int, int, int, int, int, int);
template void launch_upsample_linear_3d<int32_t>(int32_t*, int32_t*, int, int, int, int, int, int, int, int);

template<typename T>
__global__ void upsample_linear_3d_backward_kernel(
    T* grad_output, T* grad_input,
    int B, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * D_out * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c     = (idx / (W_out * H_out * D_out)) % C;
    int b     =  idx / (W_out * H_out * D_out * C);

    float d_in_float = d_out * (float)D_in / D_out;
    float h_in_float = h_out * (float)H_in / H_out;
    float w_in_float = w_out * (float)W_in / W_out;

    int d_low  = max(0, min((int)floorf(d_in_float), D_in - 1));
    int d_high = max(0, min(d_low + 1, D_in - 1));
    int h_low  = max(0, min((int)floorf(h_in_float), H_in - 1));
    int h_high = max(0, min(h_low + 1, H_in - 1));
    int w_low  = max(0, min((int)floorf(w_in_float), W_in - 1));
    int w_high = max(0, min(w_low + 1, W_in - 1));

    float wt_d_high = d_in_float - floorf(d_in_float);
    float wt_d_low  = 1.0f - wt_d_high;
    float wt_h_high = h_in_float - floorf(h_in_float);
    float wt_h_low  = 1.0f - wt_h_high;
    float wt_w_high = w_in_float - floorf(w_in_float);
    float wt_w_low  = 1.0f - wt_w_high;

    float grad = static_cast<float>(grad_output[idx]);
    int base = b * (C * D_in * H_in * W_in) + c * (D_in * H_in * W_in);
    int hw   = H_in * W_in;

    atomic_add<T>(&grad_input[base + d_low  * hw + h_low  * W_in + w_low ],  
                static_cast<T>(wt_d_low  * wt_h_low  * wt_w_low  * grad));
    atomic_add<T>(&grad_input[base + d_low  * hw + h_low  * W_in + w_high],  
                static_cast<T>(wt_d_low  * wt_h_low  * wt_w_high * grad));
    atomic_add<T>(&grad_input[base + d_low  * hw + h_high * W_in + w_low ],  
                static_cast<T>(wt_d_low  * wt_h_high * wt_w_low  * grad));
    atomic_add<T>(&grad_input[base + d_low  * hw + h_high * W_in + w_high],  
                static_cast<T>(wt_d_low  * wt_h_high * wt_w_high * grad));
    atomic_add<T>(&grad_input[base + d_high * hw + h_low  * W_in + w_low ],  
                static_cast<T>(wt_d_high * wt_h_low  * wt_w_low  * grad));
    atomic_add<T>(&grad_input[base + d_high * hw + h_low  * W_in + w_high],  
                static_cast<T>(wt_d_high * wt_h_low  * wt_w_high * grad));
    atomic_add<T>(&grad_input[base + d_high * hw + h_high * W_in + w_low ],  
                static_cast<T>(wt_d_high * wt_h_high * wt_w_low  * grad));
    atomic_add<T>(&grad_input[base + d_high * hw + h_high * W_in + w_high],  
                static_cast<T>(wt_d_high * wt_h_high * wt_w_high * grad));
}

template<typename T>
void launch_upsample_linear_3d_backward(
    T* grad_output, T* grad_input,
    int B, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * D_out * H_out * W_out + threads - 1) / threads;
    upsample_linear_3d_backward_kernel<T><<<blocks, threads>>>(
        grad_output, grad_input, B, C, D_in, H_in, W_in, D_out, H_out, W_out);
}

template void launch_upsample_linear_3d_backward<float>(float*, float*, int, int, int, int, int, int, int, int);
template void launch_upsample_linear_3d_backward<half>(half*, half*, int, int, int, int, int, int, int, int);
template void launch_upsample_linear_3d_backward<uint8_t>(uint8_t*, uint8_t*, int, int, int, int, int, int, int, int);
template void launch_upsample_linear_3d_backward<int32_t>(int32_t*, int32_t*, int, int, int, int, int, int, int, int);
