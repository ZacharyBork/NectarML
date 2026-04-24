#include "kernels/common.h"
#include "include/common/functions.h"

/* AVERAGE POOL */

// 1-Dimensional

template<typename T>
__global__ void avg_pool1d_forward_kernel(
    T* input, T* output,
    int B, int C, int L,
    int L_out, int K, int S, int P,
    bool count_include_pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * L_out) return;

    int l = idx % L_out;
    int c = (idx / L_out) % C;
    int b = idx / (L_out * C);

    float sum   = 0.0f;
    int count = 0;

    for (int k = 0; k < K; k++) {
        int l_in = l * S - P + k;
        if (l_in >= 0 && l_in < L) {
            sum += static_cast<float>(input[b * C * L + c * L + l_in]);
            count++;
        }
    }

    int denom = count_include_pad ? K : count;
    output[idx] = static_cast<T>(sum / denom);
}

template<typename T>
void launch_avg_pool1d_forward(
    T* input, T* output,
    int B, int C, int L, 
    int L_out, int K, int S, int P, 
    bool count_include_pad
) {
    int total   = B * C * L_out;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    avg_pool1d_forward_kernel<T><<<blocks, threads>>>(
        input, output,
        B, C, L, L_out, K, S, P,
        count_include_pad);
}

template void launch_avg_pool1d_forward<float>(
    float*, float*, int, int, int, int, int, int, int, bool);
template void launch_avg_pool1d_forward<half>(
    half*, half*, int, int, int, int, int, int, int, bool);
template void launch_avg_pool1d_forward<uint8_t>(
    uint8_t*, uint8_t*, int, int, int, int, int, int, int, bool);
template void launch_avg_pool1d_forward<int32_t>(
    int32_t*, int32_t*, int, int, int, int, int, int, int, bool);

template<typename T>
__global__ void avg_pool1d_backward_kernel(
    T* out_grad, T* grad_input,
    int B, int C, int L, int L_out,
    int K, int S, int P,
    bool count_include_pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * L_out) return;

    int l = idx % L_out;
    int c = (idx / L_out) % C;
    int b = idx / (L_out * C);

    int count = 0;
    for (int k = 0; k < K; k++) {
        int l_in = l * S - P + k;
        if (l_in >= 0 && l_in < L) count++;
    }
    int denom = count_include_pad ? K : count;

    for (int k = 0; k < K; k++) {
        int l_in = l * S - P + k;
        if (l_in >= 0 && l_in < L) {
            atomic_add<T>(
                &grad_input[b * C * L + c * L + l_in],
                static_cast<T>(out_grad[idx]) / static_cast<T>(denom));
        }
    }
}

template<typename T>
void launch_avg_pool1d_backward(
    T* input, T* output, 
    int B, int C, int L, int L_out,
    int K, int S, int P,
    bool count_include_pad
) {
    int total   = B * C * L_out;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    avg_pool1d_backward_kernel<T><<<blocks, threads>>>(
        input, output,
        B, C, L, L_out, K, S, P,
        count_include_pad);
}

template void launch_avg_pool1d_backward<float>(
    float*, float*, int, int, int, int, int, int, int, bool);
template void launch_avg_pool1d_backward<half>(
    half*, half*, int, int, int, int, int, int, int, bool);
template void launch_avg_pool1d_backward<uint8_t>(
    uint8_t*, uint8_t*, int, int, int, int, int, int, int, bool);
template void launch_avg_pool1d_backward<int32_t>(
    int32_t*, int32_t*, int, int, int, int, int, int, int, bool);

// 2-Dimensional

template<typename T>
__global__ void avg_pool2d_forward_kernel(
    T* input, T* output,
    int B, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW,
    int SH, int SW,
    int PH, int PW,
    bool count_include_pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H_out * W_out) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int b = idx / (W_out * H_out * C);

    float sum   = 0.0f;
    int   count = 0;

    for (int kh = 0; kh < KH; kh++) {
        for (int kw = 0; kw < KW; kw++) {
            int h_in = h * SH - PH + kh;
            int w_in = w * SW - PW + kw;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                sum += static_cast<float>(
                    input[b * C * H * W + c * H * W + h_in * W + w_in]);
                count++;
            }
        }
    }

    int denom = count_include_pad ? KH * KW : count;
    output[idx] = static_cast<T>(sum / denom);
}

template<typename T>
void launch_avg_pool2d_forward(
    T* input, T* output,
    int B, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW, 
    int SH, int SW,
    int PH, int PW, 
    bool count_include_pad
) {
    int total   = B * C * H_out * W_out;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    avg_pool2d_forward_kernel<T><<<blocks, threads>>>(
        input, output,
        B, C, H, W, H_out, W_out,
        KH, KW, SH, SW, PH, PW, 
        count_include_pad);
}

template void launch_avg_pool2d_forward<float>(
    float*, float*,
    int, int, int, int, int, int,
    int, int, int, int, int, int, bool);
template void launch_avg_pool2d_forward<half>(
    half*, half*,
    int, int, int, int, int, int,
    int, int, int, int, int, int, bool);
template void launch_avg_pool2d_forward<uint8_t>(
    uint8_t*, uint8_t*,
    int, int, int, int, int, int,
    int, int, int, int, int, int, bool);
template void launch_avg_pool2d_forward<int32_t>(
    int32_t*, int32_t*,
    int, int, int, int, int, int,
    int, int, int, int, int, int, bool);

template<typename T>
__global__ void avg_pool2d_backward_kernel(
    T* out_grad, T* grad_input,
    int B, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW,
    int SH, int SW,
    int PH, int PW,
    bool count_include_pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H_out * W_out) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int b = idx / (W_out * H_out * C);

    int count = 0;
    for (int kh = 0; kh < KH; kh++) {
        for (int kw = 0; kw < KW; kw++) {
            int h_in = h * SH - PH + kh;
            int w_in = w * SW - PW + kw;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) count++;
        }
    }
    int denom = count_include_pad ? KH * KW : count;

    for (int kh = 0; kh < KH; kh++) {
        for (int kw = 0; kw < KW; kw++) {
            int h_in = h * SH - PH + kh;
            int w_in = w * SW - PW + kw;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                atomic_add<T>(
                    &grad_input[b * C * H * W + c * H * W + h_in * W + w_in],
                    static_cast<T>(out_grad[idx]) / static_cast<T>(denom));
            }
        }
    }
}

template<typename T>
void launch_avg_pool2d_backward(
    T* input, T* output,
    int B, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW, 
    int SH, int SW,
    int PH, int PW, 
    bool count_include_pad
) {
    int total   = B * C * H_out * W_out;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    avg_pool2d_backward_kernel<T><<<blocks, threads>>>(
        input, output,
        B, C, H, W, H_out, W_out,
        KH, KW, SH, SW, PH, PW, 
        count_include_pad);
}

template void launch_avg_pool2d_backward<float>(
    float*, float*,
    int, int, int, int, int, int,
    int, int, int, int, int, int, bool);
template void launch_avg_pool2d_backward<half>(
    half*, half*,
    int, int, int, int, int, int,
    int, int, int, int, int, int, bool);
template void launch_avg_pool2d_backward<uint8_t>(
    uint8_t*, uint8_t*,
    int, int, int, int, int, int,
    int, int, int, int, int, int, bool);
template void launch_avg_pool2d_backward<int32_t>(
    int32_t*, int32_t*,
    int, int, int, int, int, int,
    int, int, int, int, int, int, bool);

// 3-Dimensional

template<typename T>
__global__ void avg_pool3d_forward_kernel(
    T* input, T* output,
    int B, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int KD, int KH, int KW,
    int SD, int SH, int SW,
    int PD, int PH, int PW,
    bool count_include_pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * D_out * H_out * W_out) return;

    int w  = idx % W_out;
    int h  = (idx / W_out) % H_out;
    int d  = (idx / (W_out * H_out)) % D_out;
    int c  = (idx / (W_out * H_out * D_out)) % C;
    int b  = idx / (W_out * H_out * D_out * C);

    float sum   = 0.0f;
    int   count = 0;

    for (int kd = 0; kd < KD; kd++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int d_in = d * SD - PD + kd;
                int h_in = h * SH - PH + kh;
                int w_in = w * SW - PW + kw;
                if (d_in >= 0 && d_in < D &&
                    h_in >= 0 && h_in < H &&
                    w_in >= 0 && w_in < W) {
                    sum += static_cast<float>(
                        input[b * C * D * H * W
                            + c * D * H * W
                            + d_in * H * W
                            + h_in * W
                            + w_in]);
                    count++;
                }
            }
        }
    }

    int denom = count_include_pad ? KD * KH * KW : count;
    output[idx] = static_cast<T>(sum / denom);
}

template<typename T>
void launch_avg_pool3d_forward(
    T* input, T* output,
    int B, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int KD, int KH, int KW, 
    int SD, int SH, int SW,
    int PD, int PH, int PW, 
    bool count_include_pad
) {
    int total   = B * C * D_out * H_out * W_out;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    avg_pool3d_forward_kernel<T><<<blocks, threads>>>(
        input, output,
        B, C, D, H, W, 
        D_out, H_out, W_out,
        KD, KH, KW, 
        SD, SH, SW, 
        PD, PH, PW, 
        count_include_pad);
}

template void launch_avg_pool3d_forward<float>(
    float*, float*,
    int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, bool);
template void launch_avg_pool3d_forward<half>(
    half*, half*,
    int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, bool);
template void launch_avg_pool3d_forward<uint8_t>(
    uint8_t*, uint8_t*,
    int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, bool);
template void launch_avg_pool3d_forward<int32_t>(
    int32_t*, int32_t*,
    int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, bool);

template<typename T>
__global__ void avg_pool3d_backward_kernel(
    T* out_grad, T* grad_input,
    int B, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int KD, int KH, int KW,
    int SD, int SH, int SW,
    int PD, int PH, int PW,
    bool count_include_pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * D_out * H_out * W_out) return;

    int w  = idx % W_out;
    int h  = (idx / W_out) % H_out;
    int d  = (idx / (W_out * H_out)) % D_out;
    int c  = (idx / (W_out * H_out * D_out)) % C;
    int b  = idx / (W_out * H_out * D_out * C);

    int count = 0;
    for (int kd = 0; kd < KD; kd++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int d_in = d * SD - PD + kd;
                int h_in = h * SH - PH + kh;
                int w_in = w * SW - PW + kw;
                if (d_in >= 0 && d_in < D &&
                    h_in >= 0 && h_in < H &&
                    w_in >= 0 && w_in < W) count++;
            }
        }
    }
    int denom = count_include_pad ? KD * KH * KW : count;

    for (int kd = 0; kd < KD; kd++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int d_in = d * SD - PD + kd;
                int h_in = h * SH - PH + kh;
                int w_in = w * SW - PW + kw;
                if (d_in >= 0 && d_in < D &&
                    h_in >= 0 && h_in < H &&
                    w_in >= 0 && w_in < W) {
                    atomic_add<T>(
                        &grad_input[b * C * D * H * W
                                  + c * D * H * W
                                  + d_in * H * W
                                  + h_in * W
                                  + w_in],
                        static_cast<T>(out_grad[idx]) / static_cast<T>(denom));
                }
            }
        }
    }
}

template<typename T>
void launch_avg_pool3d_backward(
    T* input, T* output,
    int B, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int KD, int KH, int KW, 
    int SD, int SH, int SW,
    int PD, int PH, int PW, 
    bool count_include_pad
) {
    int total   = B * C * D_out * H_out * W_out;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    avg_pool3d_backward_kernel<T><<<blocks, threads>>>(
        input, output,
        B, C, D, H, W, 
        D_out, H_out, W_out,
        KD, KH, KW, 
        SD, SH, SW, 
        PD, PH, PW, 
        count_include_pad);
}

template void launch_avg_pool3d_backward<float>(
    float*, float*,
    int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, bool);
template void launch_avg_pool3d_backward<half>(
    half*, half*,
    int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, bool);
template void launch_avg_pool3d_backward<uint8_t>(
    uint8_t*, uint8_t*,
    int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, bool);
template void launch_avg_pool3d_backward<int32_t>(
    int32_t*, int32_t*,
    int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, bool);

/* MAX POOL */

// 1-Dimensional

template<typename T>
__global__ void max_pool1d_forward_kernel(
    T* input, T* output, int32_t* indices,
    int B, int C, int L,
    int L_out, int K, int S, int P, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * L_out) return;

    int l = idx % L_out;
    int c = (idx / L_out) % C;
    int b = idx / (L_out * C);

    float max_val = -1e38f;
    int   max_idx = -1;

    for (int k = 0; k < K; k++) {
        int l_in = l * S - P + k * D;
        if (l_in >= 0 && l_in < L) {
            float val = static_cast<float>(
                input[b * C * L + c * L + l_in]);
            if (val > max_val) {
                max_val = val;
                max_idx = l_in;
            }
        }
    }

    output[idx]  = static_cast<T>(max_val);
    indices[idx] = static_cast<int32_t>(max_idx);
}

template<typename T>
void launch_max_pool1d_forward(
    T* input, T* output, int32_t* indices,
    int B, int C, int L, int L_out, 
    int K, int S, int P, int D
) {
    int total   = B * C * L_out;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    max_pool1d_forward_kernel<T><<<blocks, threads>>>(
        input, output, indices,
        B, C, L, L_out, K, S, P, D);
}

template void launch_max_pool1d_forward<float>(
    float*, float*, int32_t*, int, int, int, int, int, int, int, int);
template void launch_max_pool1d_forward<half>(
    half*, half*, int32_t*, int, int, int, int, int, int, int, int);
template void launch_max_pool1d_forward<uint8_t>(
    uint8_t*, uint8_t*, int32_t*, int, int, int, int, int, int, int, int);
template void launch_max_pool1d_forward<int32_t>(
    int32_t*, int32_t*, int32_t*, int, int, int, int, int, int, int, int);

template<typename T>
__global__ void max_pool1d_backward_kernel(
    T* out_grad, int32_t* indices, T* grad_input,
    int B, int C, int L, int L_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * L_out) return;

    int l = idx % L_out;
    int c = (idx / L_out) % C;
    int b = idx / (L_out * C);

    int l_in = indices[idx];
    if (l_in >= 0) {
        atomic_add<T>(
            &grad_input[b * C * L + c * L + l_in],
            static_cast<T>(out_grad[idx]));
    }
}

template<typename T>
void launch_max_pool1d_backward(
    T* out_grad, int32_t* indices, T* grad_input,
    int B, int C, int L, int L_out
) {
    int total   = B * C * L_out;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    max_pool1d_backward_kernel<T><<<blocks, threads>>>(
        out_grad, indices, grad_input, B, C, L, L_out);
}

template void launch_max_pool1d_backward<float>(
    float*, int32_t*, float*, int, int, int, int);
template void launch_max_pool1d_backward<half>(
    half*, int32_t*, half*, int, int, int, int);
template void launch_max_pool1d_backward<uint8_t>(
    uint8_t*, int32_t*, uint8_t*, int, int, int, int);
template void launch_max_pool1d_backward<int32_t>(
    int32_t*, int32_t*, int32_t*, int, int, int, int);

// 2-Dimensional

template<typename T>
__global__ void max_pool2d_forward_kernel(
    T* input, T* output, int32_t* indices,
    int B, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW,
    int SH, int SW,
    int PH, int PW, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H_out * W_out) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int b = idx / (W_out * H_out * C);

    float   max_val = -1e38f;
    int32_t max_idx = -1;

    for (int kh = 0; kh < KH; kh++) {
        for (int kw = 0; kw < KW; kw++) {
            int h_in = h * SH - PH + kh * D;
            int w_in = w * SW - PW + kw * D;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                float val = static_cast<float>(
                    input[b * C * H * W + c * H * W + h_in * W + w_in]);
                if (val > max_val) {
                    max_val = val;
                    max_idx = static_cast<int32_t>(h_in * W + w_in);
                }
            }
        }
    }

    output[idx]  = static_cast<T>(max_val);
    indices[idx] = max_idx;
}

template<typename T>
void launch_max_pool2d_forward(
    T* input, T* output, int32_t* indices,
    int B, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW, int SH, int SW,
    int PH, int PW, int D
) {
    int total   = B * C * H_out * W_out;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    max_pool2d_forward_kernel<T><<<blocks, threads>>>(
        input, output, indices,
        B, C, H, W, H_out, W_out,
        KH, KW, SH, SW, PH, PW, D);
}

template void launch_max_pool2d_forward<float>(
    float*, float*, int32_t*, 
    int, int, int, int, int, int,
    int, int, int, int, int, int, int);
template void launch_max_pool2d_forward<half>(
    half*, half*, int32_t*, 
    int, int, int, int, int, int,
    int, int, int, int, int, int, int);
template void launch_max_pool2d_forward<uint8_t>(
    uint8_t*, uint8_t*, int32_t*, 
    int, int, int, int, int, int,
    int, int, int, int, int, int, int);
template void launch_max_pool2d_forward<int32_t>(
    int32_t*, int32_t*, int32_t*, 
    int, int, int, int, int, int,
    int, int, int, int, int, int, int);

template<typename T>
__global__ void max_pool2d_backward_kernel(
    T* out_grad, int32_t* indices, T* grad_input,
    int B, int C, int H, int W, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H_out * W_out) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int b = idx / (W_out * H_out * C);

    int32_t flat = indices[idx];
    if (flat >= 0) {
        int h_in = flat / W;
        int w_in = flat % W;
        atomic_add<T>(
            &grad_input[b * C * H * W + c * H * W + h_in * W + w_in],
            static_cast<T>(out_grad[idx]));
    }
}

template<typename T>
void launch_max_pool2d_backward(
    T* out_grad, int32_t* indices, T* grad_input,
    int B, int C, int H, int W, int H_out, int W_out
) {
    int total   = B * C * H_out * W_out;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    max_pool2d_backward_kernel<T><<<blocks, threads>>>(
        out_grad, indices, grad_input,
        B, C, H, W, H_out, W_out);
}

template void launch_max_pool2d_backward<float>(
    float*, int32_t*, float*, 
    int, int, int, int, int, int);
template void launch_max_pool2d_backward<half>(
    half*, int32_t*, half*, 
    int, int, int, int, int, int);
template void launch_max_pool2d_backward<uint8_t>(
    uint8_t*, int32_t*, uint8_t*,
    int, int, int, int, int, int);
template void launch_max_pool2d_backward<int32_t>(
    int32_t*, int32_t*, int32_t*,
    int, int, int, int, int, int);

// 3-Dimensional

template<typename T>
__global__ void max_pool3d_forward_kernel(
    T* input, T* output, int32_t* indices,
    int B, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int KD, int KH, int KW,
    int SD, int SH, int SW,
    int PD, int PH, int PW, 
    int Dil
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * D_out * H_out * W_out) return;

    int w  = idx % W_out;
    int h  = (idx / W_out) % H_out;
    int d  = (idx / (W_out * H_out)) % D_out;
    int c  = (idx / (W_out * H_out * D_out)) % C;
    int b  = idx / (W_out * H_out * D_out * C);

    float   max_val = -1e38f;
    int32_t max_idx = -1;

    for (int kd = 0; kd < KD; kd++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int d_in = d * SD - PD + kd * Dil;
                int h_in = h * SH - PH + kh * Dil;
                int w_in = w * SW - PW + kw * Dil;
                if (d_in >= 0 && d_in < D &&
                    h_in >= 0 && h_in < H &&
                    w_in >= 0 && w_in < W) {
                    float val = static_cast<float>(
                        input[b * C * D * H * W
                            + c * D * H * W
                            + d_in * H * W
                            + h_in * W
                            + w_in]);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = static_cast<int32_t>(
                            d_in * H * W + h_in * W + w_in);
                    }
                }
            }
        }
    }

    output[idx]  = static_cast<T>(max_val);
    indices[idx] = max_idx;
}

template<typename T>
void launch_max_pool3d_forward(
    T* input, T* output, int32_t* indices,
    int B, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int KD, int KH, int KW, 
    int SD, int SH, int SW,
    int PD, int PH, int PW, 
    int Dil
) {
    int total   = B * C * D_out * H_out * W_out;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    max_pool3d_forward_kernel<T><<<blocks, threads>>>(
        input, output, indices,
        B, C, D, H, W, 
        D_out, H_out, W_out,
        KD, KH, KW, 
        SD, SH, SW, 
        PD, PH, PW, 
        Dil);
}

template void launch_max_pool3d_forward<float>(
    float*, float*, int32_t*, 
    int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int);
template void launch_max_pool3d_forward<half>(
    half*, half*, int32_t*, 
    int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int);
template void launch_max_pool3d_forward<uint8_t>(
    uint8_t*, uint8_t*, int32_t*, 
    int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int);
template void launch_max_pool3d_forward<int32_t>(
    int32_t*, int32_t*, int32_t*, 
    int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int);

template<typename T>
__global__ void max_pool3d_backward_kernel(
    T* out_grad, int32_t* indices, T* grad_input,
    int B, int C, int D, int H, int W,
    int D_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * D_out * H_out * W_out) return;

    int w  = idx % W_out;
    int h  = (idx / W_out) % H_out;
    int d  = (idx / (W_out * H_out)) % D_out;
    int c  = (idx / (W_out * H_out * D_out)) % C;
    int b  = idx / (W_out * H_out * D_out * C);

    int32_t flat = indices[idx];
    if (flat >= 0) {
        int d_in = flat / (H * W);
        int h_in = (flat % (H * W)) / W;
        int w_in = flat % W;
        atomic_add<T>(
            &grad_input[b * C * D * H * W
                      + c * D * H * W
                      + d_in * H * W
                      + h_in * W
                      + w_in],
            static_cast<T>(out_grad[idx]));
    }
}

template<typename T>
void launch_max_pool3d_backward(
    T* out_grad, int32_t* indices, T* grad_input,
    int B, int C, int D, int H, int W,
    int D_out, int H_out, int W_out
) {
    int total   = B * C * D_out * H_out * W_out;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    max_pool3d_backward_kernel<T><<<blocks, threads>>>(
        out_grad, indices, grad_input,
        B, C, D, H, W, D_out, H_out, W_out);
}

template void launch_max_pool3d_backward<float>(
    float*, int32_t*, float*, 
    int, int, int, int, int, int, int, int);
template void launch_max_pool3d_backward<half>(
    half*, int32_t*, half*, 
    int, int, int, int, int, int, int, int);
template void launch_max_pool3d_backward<uint8_t>(
    uint8_t*, int32_t*, uint8_t*, 
    int, int, int, int, int, int, int, int);
template void launch_max_pool3d_backward<int32_t>(
    int32_t*, int32_t*, int32_t*, 
    int, int, int, int, int, int, int, int);
