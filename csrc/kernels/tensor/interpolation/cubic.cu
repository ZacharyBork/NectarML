#include "common.h"

/* BICUBIC */

__device__ float cubic_weight(float t, float a = -0.75f) {
    t = fabsf(t);
    if (t <= 1.0f)
        return (a + 2.0f) * t*t*t - (a + 3.0f) * t*t + 1.0f;
    else if (t < 2.0f)
        return a * t*t*t - 5.0f*a * t*t + 8.0f*a * t - 4.0f*a;
    return 0.0f;
}

template<typename T>
__global__ void upsample_bicubic_kernel(
    T* input, T* output,
    int B, int C, int H_in, int W_in, int H_out, int W_out,
    float a = -0.75f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c     = (idx / (W_out * H_out)) % C;
    int b     =  idx / (W_out * H_out * C);

    float h_in_float = h_out * (float)H_in / H_out;
    float w_in_float = w_out * (float)W_in / W_out;

    int h_base = (int)floorf(h_in_float);
    int w_base = (int)floorf(w_in_float);

    int base = b * (C * H_in * W_in) + c * (H_in * W_in);

    float result = 0.0f;
    for (int i = 0; i < 4; i++) {
        int h_idx = max(0, min(h_base + i - 1, H_in - 1));
        float wh = cubic_weight(h_in_float - (float)(h_base + i - 1));
        for (int j = 0; j < 4; j++) {
            int w_idx = max(0, min(w_base + j - 1, W_in - 1));
            float ww = cubic_weight(w_in_float - (float)(w_base + j - 1));
            result += wh * ww * static_cast<float>(
                input[base + h_idx * W_in + w_idx]);
        }
    }

    output[idx] = static_cast<T>(result);
}

template<typename T>
void launch_upsample_bicubic(
    T* input, T* output,
    int B, int C, int H_in, int W_in, int H_out, int W_out,
    float a = -0.75f
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * H_out * W_out + threads - 1) / threads;
    upsample_bicubic_kernel<T><<<blocks, threads>>>(
        input, output, B, C, H_in, W_in, H_out, W_out, a);
}

template void launch_upsample_bicubic<float>(float*, float*, int, int, int, int, int, int, float);
template void launch_upsample_bicubic<half>(half*, half*, int, int, int, int, int, int, float);
template void launch_upsample_bicubic<uint8_t>(uint8_t*, uint8_t*, int, int, int, int, int, int, float);
template void launch_upsample_bicubic<int32_t>(int32_t*, int32_t*, int, int, int, int, int, int, float);

template<typename T>
__global__ void upsample_bicubic_backward_kernel(
    T* grad_output, T* grad_input,
    int B, int C, 
    int H_in, int W_in,
    int H_out, int W_out,
    float a = -0.75f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c     = (idx / (W_out * H_out)) % C;
    int b     =  idx / (W_out * H_out * C);

    float h_in_float = h_out * (float)H_in / H_out;
    float w_in_float = w_out * (float)W_in / W_out;

    int h_base = (int)floorf(h_in_float);
    int w_base = (int)floorf(w_in_float);

    float grad = static_cast<float>(grad_output[idx]);
    int base = b * (C * H_in * W_in) + c * (H_in * W_in);

    for (int i = 0; i < 4; i++) {
        int h_idx = max(0, min(h_base + i - 1, H_in - 1));
        float wh = cubic_weight(h_in_float - (float)(h_base + i - 1), a);
        for (int j = 0; j < 4; j++) {
            int w_idx = max(0, min(w_base + j - 1, W_in - 1));
            float ww = cubic_weight(w_in_float - (float)(w_base + j - 1), a);
            atomic_add<T>(&grad_input[base + h_idx * W_in + w_idx],
                          static_cast<T>(wh * ww * grad));
        }
    }
}

template<typename T>
void launch_upsample_bicubic_backward(
    T* grad_output, T* grad_input,
    int B, int C, 
    int H_in, int W_in, 
    int H_out, int W_out,
    float a = -0.75f
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (B * C * H_out * W_out + threads - 1) / threads;
    upsample_bicubic_backward_kernel<T><<<blocks, threads>>>(
        grad_output, grad_input, B, C, H_in, W_in, H_out, W_out);
}

template void launch_upsample_bicubic_backward<float>(float*, float*, int, int, int, int, int, int, float);
template void launch_upsample_bicubic_backward<half>(half*, half*, int, int, int, int, int, int, float);
template void launch_upsample_bicubic_backward<uint8_t>(uint8_t*, uint8_t*, int, int, int, int, int, int, float);
template void launch_upsample_bicubic_backward<int32_t>(int32_t*, int32_t*, int, int, int, int, int, int, float);


