#include "common.h"

/* Conv1d */

template<typename T>
__global__ void add_bias_1d_kernel(
    T* output, T* bias,
    int B, int C_out, int L_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C_out * L_out) return;
    
    int c = idx / (B * L_out);  // changed from (idx / L_out) % C_out
    output[idx] += bias[c];
}

template<typename T>
void launch_add_bias_1d(
    T* output, T* bias,
    int B, int C_out, int L_out
) {
    int total = B * C_out * L_out;
    int threads = BLOCK_SIZE_1D;
    int blocks = (total + threads - 1) / threads;
    add_bias_1d_kernel<T><<<blocks, threads>>>(output, bias, B, C_out, L_out);
}

template void launch_add_bias_1d<float>(float*, float*, int, int, int);
template void launch_add_bias_1d<half>(half*, half*, int, int, int);
template void launch_add_bias_1d<uint8_t>(uint8_t*, uint8_t*, int, int, int);
template void launch_add_bias_1d<int32_t>(int32_t*, int32_t*, int, int, int);

template<typename T>
__global__ void transpose_output_1d_kernel(
    T* input, T* output,
    int B, int C_out, int L_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C_out * L_out) return;
    
    int l_out = idx % L_out;
    int c_out = (idx / L_out) % C_out;
    int b     = idx / (L_out * C_out);
    
    int in_row = b * L_out + l_out;
    int in_col = c_out;
    
    int in_flat = c_out * (B * L_out) + b * L_out + l_out;
    output[idx] = input[in_flat];
}

template<typename T>
void launch_transpose_output_1d(
    T* output, T* bias,
    int B, int C_out, int L_out
) {
    int total = B * C_out * L_out;
    int threads = BLOCK_SIZE_1D;
    int blocks = (total + threads - 1) / threads;
    transpose_output_1d_kernel<T><<<blocks, threads>>>(output, bias, B, C_out, L_out);
}

template void launch_transpose_output_1d<float>(float*, float*, int, int, int);
template void launch_transpose_output_1d<half>(half*, half*, int, int, int);
template void launch_transpose_output_1d<uint8_t>(uint8_t*, uint8_t*, int, int, int);
template void launch_transpose_output_1d<int32_t>(int32_t*, int32_t*, int, int, int);

/* Conv2d */

template<typename T>
__global__ void add_bias_2d_kernel(
    T* output, T* bias,
    int B, int C_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C_out * H_out * W_out) return;
    int c = idx / (B * H_out * W_out);
    output[idx] += bias[c];
}

template<typename T>
void launch_add_bias_2d(
    T* output, T* bias,
    int B, int C_out, int H_out, int W_out
) {
    int total = B * C_out * H_out * W_out;
    int threads = BLOCK_SIZE_1D;
    int blocks = (total + threads - 1) / threads;
    add_bias_2d_kernel<T><<<blocks, threads>>>(
        output, bias, B, C_out, H_out, W_out);
}

template void launch_add_bias_2d<float>(float*, float*, int, int, int, int);
template void launch_add_bias_2d<half>(half*, half*, int, int, int, int);
template void launch_add_bias_2d<uint8_t>(uint8_t*, uint8_t*, int, int, int, int);
template void launch_add_bias_2d<int32_t>(int32_t*, int32_t*, int, int, int, int);

template<typename T>
__global__ void transpose_output_2d_kernel(
    T* input, T* output,
    int B, int C_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C_out * H_out * W_out) return;

    int w_out =  idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int b     =  idx / (W_out * H_out * C_out);

    int in_flat = c_out * (B * H_out * W_out) 
                + b * H_out * W_out 
                + h_out * W_out + w_out;
    output[idx] = input[in_flat];
}

template<typename T>
void launch_transpose_output_2d(
    T* input, T* output,
    int B, int C_out, int H_out, int W_out
) {
    int total = B * C_out * H_out * W_out;
    int threads = BLOCK_SIZE_1D;
    int blocks = (total + threads - 1) / threads;
    transpose_output_2d_kernel<T><<<blocks, threads>>>(
        input, output, B, C_out, H_out, W_out);
}

template void launch_transpose_output_2d<float>(float*, float*, int, int, int, int);
template void launch_transpose_output_2d<half>(half*, half*, int, int, int, int);
template void launch_transpose_output_2d<uint8_t>(uint8_t*, uint8_t*, int, int, int, int);
template void launch_transpose_output_2d<int32_t>(int32_t*, int32_t*, int, int, int, int);

/* Conv3d */



