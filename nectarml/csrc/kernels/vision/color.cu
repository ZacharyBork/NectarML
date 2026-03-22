#include "common.h"

__device__ void rgb_to_hsv(
    float r, float g, float b, 
    float& h, float& s, float& v
) {
    float cmax = fmaxf(fmaxf(r, g), b);
    float cmin = fminf(fminf(r, g), b);
    float delta = cmax - cmin;

    v = cmax;
    s = (cmax == 0.0f) ? 0.0f : delta / cmax;
    if (delta == 0.0f) { h = 0.0f; } 
    else if (cmax == r) { h = fmodf((g - b) / delta, 6.0f); } 
    else if (cmax == g) { h = (b - r) / delta + 2.0f; } 
    else { h = (r - g) / delta + 4.0f; }

    h = h / 6.0f;
    if (h < 0.0f) h += 1.0f;
}

__device__ void hsv_to_rgb(
    float h, float s, float v, 
    float& r, float& g, float& b
) {
    if (s == 0.0f) {
        r = g = b = v;
        return;
    }

    float i = floorf(h * 6.0f);
    float f = h * 6.0f - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);

    switch ((int)i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }
}

#include <iostream>

template<typename T>
__global__ void hsv_adjust_kernel(
    T* d_in,
    int B, int C, int H, int W,
    float hue_shift, float saturation, float value
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (x >= W || y >= H || b >= B) return;

    int r_idx = b * (C * H * W) + 0 * (H * W) + y * W + x;
    int g_idx = b * (C * H * W) + 1 * (H * W) + y * W + x;
    int b_idx = b * (C * H * W) + 2 * (H * W) + y * W + x;

    float scale;
    if constexpr (std::is_same_v<T, uint8_t>) scale = 255.0f;
    else scale = 1.0f;

    float rc = static_cast<float>(d_in[r_idx]) / scale;
    float gc = static_cast<float>(d_in[g_idx]) / scale;
    float bc = static_cast<float>(d_in[b_idx]) / scale;

    float h, s, v;
    rgb_to_hsv(rc, gc, bc, h, s, v);

    h = fmodf(h + hue_shift, 1.0f);
    if (h < 0.0f) h += 1.0f;
    s = fmaxf(0.0f, fminf(1.0f, s * saturation));
    v = fmaxf(0.0f, fminf(1.0f, v * value));

    hsv_to_rgb(h, s, v, rc, gc, bc);

    d_in[r_idx] = static_cast<T>(rc * scale);
    d_in[g_idx] = static_cast<T>(gc * scale);
    d_in[b_idx] = static_cast<T>(bc * scale);
}

template<typename T>
void launch_hsv_adjust(
    T* d_in,
    int B, int C, int H, int W,
    float hue_shift, float saturation, float value
) {
    int BS2D = BLOCK_SIZE_2D;
    dim3 block(BS2D, BS2D, 1);
    dim3 grid((W + BS2D - 1) / BS2D, (H + BS2D - 1) / BS2D, B);
    hsv_adjust_kernel<T><<<grid, block>>>(
        d_in, B, C, H, W,
        hue_shift, saturation, value);
}

template void launch_hsv_adjust<float>(float*, int, int, int, int, float, float, float);
template void launch_hsv_adjust<half>(half*, int, int, int, int, float, float, float);
template void launch_hsv_adjust<uint8_t>(uint8_t*, int, int, int, int, float, float, float);
template void launch_hsv_adjust<int32_t>(int32_t*, int, int, int, int, float, float, float);

