// RESOURCES:
// Nick, YouTube, "CUDA Crash Course: Naive 1-D Convolution" : https://www.youtube.com/watch?v=OlLquh9Lnbc
// Nick, Youtube, "CUDA Crash Course: 2-D Convolution" : https://www.youtube.com/watch?v=qxcfco89wvs&t=31s
//
// Zhou, Yangjie, et al., 
//      2021 IEEE International Symposium on Workload Characterization (IISWC). IEEE, (2021), 
//      Characterizing and demystifying the implicit convolution algorithm on commercial matrix-multiplication accelerators.
//      https://arxiv.org/abs/2110.03901

#include "common.h"

/* 1-Dimensional */

template<typename T>
__global__ void im2col_1d_kernel(
    T* input,
    T* output,
    int B, int C_in, int L,
    int K, int L_out,
    int stride, int padding, int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_in * K * B * L_out;
    if (idx >= total) return;

    int l_out = idx % L_out;
    int b    = (idx / L_out) % B;
    int k    = (idx / (L_out * B)) % K;
    int c_in = (idx / (L_out * B * K));

    int l_in = l_out * stride - padding + k * dilation;

    int row = c_in * K + k;
    int col = b * L_out + l_out;

    if (l_in >= 0 && l_in < L) {
        output[row * (B * L_out) + col] = input[b * (C_in * L) + c_in * L + l_in];
    } 
    else {
        output[row * (B * L_out) + col] = static_cast<T>(0);
    }
}

template<typename T>
void launch_im2col_1d(
    T* input,
    T* output,
    int B, int C_in, int L,
    int K, int L_out,
    int stride, int padding, int dilation
) {
    int total = C_in * K * B * L_out;
    int threads = BLOCK_SIZE_1D;
    int blocks = (total + threads - 1) / threads;
    im2col_1d_kernel<T><<<blocks, threads>>>(
        input, output,
        B, C_in, L, K, L_out,
        stride, padding, dilation);
}

template void launch_im2col_1d<float>(float*, float*, int, int, int, int, int, int, int, int);
template void launch_im2col_1d<half>(half*, half*, int, int, int, int, int, int, int, int);
template void launch_im2col_1d<uint8_t>(uint8_t*, uint8_t*, int, int, int, int, int, int, int, int);
template void launch_im2col_1d<int32_t>(int32_t*, int32_t*, int, int, int, int, int, int, int, int);

template<typename T>
__global__ void col2im_1d_kernel(
    T* col,
    T* input_grad,
    int B, int C_in, int L,
    int K, int L_out,
    int stride, int padding, int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_in * L;
    if (idx >= total) return;
    
    int l_in = idx % L;
    int c_in = (idx / L) % C_in;
    int b    = idx / (L * C_in);

    T grad = static_cast<T>(0);

    for (int k = 0; k < K; k++) {
        int l_out_origin = l_in + padding - k * dilation;
        
        if (l_out_origin % stride != 0) continue;
        int l_out = l_out_origin / stride;
        
        if (l_out < 0 || l_out >= L_out) continue;

        int row = c_in * K + k;
        int col_idx = b * L_out + l_out;

        grad += col[row * (B * L_out) + col_idx];
    }

    input_grad[idx] += grad;
}

template<typename T>
void launch_col2im_1d(
    T* col,
    T* input_grad,
    int B, int C_in, int L,
    int K, int L_out,
    int stride, int padding, int dilation
) {
    int total = B * C_in * L;
    int threads = BLOCK_SIZE_1D;
    int blocks = (total + threads - 1) / threads;
    col2im_1d_kernel<T><<<blocks, threads>>>(
        col, input_grad,
        B, C_in, L, K, L_out,
        stride, padding, dilation);
}

template void launch_col2im_1d<float>(float*, float*, int, int, int, int, int, int, int, int);
template void launch_col2im_1d<half>(half*, half*, int, int, int, int, int, int, int, int);
template void launch_col2im_1d<uint8_t>(uint8_t*, uint8_t*, int, int, int, int, int, int, int, int);
template void launch_col2im_1d<int32_t>(int32_t*, int32_t*, int, int, int, int, int, int, int, int);


/* 2-Dimensional */


