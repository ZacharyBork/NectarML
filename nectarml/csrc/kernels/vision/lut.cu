#include "kernels/common.h"

template<typename T>
__global__ void apply_lut_kernel(
    T* input, T* output,
    float* lut,
    int B, int H, int W,
    int lut_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int b = idx / (W * H);

    int r_offset = b * 3 * H * W + 0 * H * W + h * W + w;
    int g_offset = b * 3 * H * W + 1 * H * W + h * W + w;
    int b_offset = b * 3 * H * W + 2 * H * W + h * W + w;

    float r = static_cast<float>(input[r_offset]);
    float g = static_cast<float>(input[g_offset]);
    float b_val = static_cast<float>(input[b_offset]);

    float scale = static_cast<float>(lut_size - 1);
    float r_idx = r * scale;
    float g_idx = g * scale;
    float b_idx = b_val * scale;

    int r0 = min(static_cast<int>(r_idx), lut_size - 1);
    int g0 = min(static_cast<int>(g_idx), lut_size - 1);
    int b0 = min(static_cast<int>(b_idx), lut_size - 1);
    int r1 = min(r0 + 1, lut_size - 1);
    int g1 = min(g0 + 1, lut_size - 1);
    int b1 = min(b0 + 1, lut_size - 1);

    float rf = r_idx - static_cast<float>(r0);
    float gf = g_idx - static_cast<float>(g0);
    float bf = b_idx - static_cast<float>(b0);

    auto lut_idx = [&](int ri, int gi, int bi, int ci) -> int {
        return ri * lut_size * lut_size * 3
             + gi * lut_size * 3
             + bi * 3
             + ci;
    };

    for (int c = 0; c < 3; c++) {
        float c000 = lut[lut_idx(r0, g0, b0, c)];
        float c001 = lut[lut_idx(r0, g0, b1, c)];
        float c010 = lut[lut_idx(r0, g1, b0, c)];
        float c011 = lut[lut_idx(r0, g1, b1, c)];
        float c100 = lut[lut_idx(r1, g0, b0, c)];
        float c101 = lut[lut_idx(r1, g0, b1, c)];
        float c110 = lut[lut_idx(r1, g1, b0, c)];
        float c111 = lut[lut_idx(r1, g1, b1, c)];

        float c00 = c000 + rf * (c100 - c000);
        float c01 = c001 + rf * (c101 - c001);
        float c10 = c010 + rf * (c110 - c010);
        float c11 = c011 + rf * (c111 - c011);

        float c0 = c00 + gf * (c10 - c00);
        float c1 = c01 + gf * (c11 - c01);

        float result = c0 + bf * (c1 - c0);

        int out_offset = b * 3 * H * W + c * H * W + h * W + w;
        output[out_offset] = static_cast<T>(result);
    }
}

template<typename T>
void launch_apply_lut(
    T* input, T* output,
    float* lut,
    int B, int H, int W,
    int lut_size
) {
    int total   = B * H * W;
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total + threads - 1) / threads;
    apply_lut_kernel<T><<<blocks, threads>>>(
        input, output, lut,
        B, H, W, lut_size);
}

template void launch_apply_lut<float>(float*, float*, float*, int, int, int, int);
template void launch_apply_lut<half>(half*, half*, float*, int, int, int, int);
template void launch_apply_lut<uint8_t>(uint8_t*, uint8_t*, float*, int, int, int, int);
template void launch_apply_lut<int32_t>(int32_t*, int32_t*, float*, int, int, int, int);

