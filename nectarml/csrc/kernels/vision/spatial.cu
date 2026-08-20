#include "kernels/common.h"

template<typename T>
__global__ void rotate_kernel(
    T* input, T* output,
    int B, int C, int H, int W,
    float cos_angle, float sin_angle,
    float fill_value
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;
    if (x >= W || y >= H || bc >= B * C) return;

    float cx = (W - 1) / 2.0f;
    float cy = (H - 1) / 2.0f;

    float x_c = x - cx;
    float y_c = y - cy;
    float x_src = x_c * cos_angle + y_c * sin_angle + cx;
    float y_src = -x_c * sin_angle + y_c * cos_angle + cy;

    int base = bc * (H * W);

    if (x_src < 0 || x_src >= W - 1 || y_src < 0 || y_src >= H - 1) {
        output[base + y * W + x] = static_cast<T>(fill_value);
        return;
    }

    int x0 = (int)floorf(x_src);
    int y0 = (int)floorf(y_src);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wx = x_src - x0;
    float wy = y_src - y0;

    float v00 = static_cast<float>(input[base + y0 * W + x0]);
    float v01 = static_cast<float>(input[base + y0 * W + x1]);
    float v10 = static_cast<float>(input[base + y1 * W + x0]);
    float v11 = static_cast<float>(input[base + y1 * W + x1]);

    float result = (1-wy) * ((1-wx) * v00 + wx * v01)
                 + wy * ((1-wx) * v10 + wx * v11);

    output[base + y * W + x] = static_cast<T>(result);
}

template<typename T>
void launch_rotate(
    T* input, T* output,
    int B, int C, int H, int W,
    float angle_degrees, float fill_value
) {
    float angle_rad = angle_degrees * M_PI / 180.0f;
    float cos_a = cosf(angle_rad);
    float sin_a = sinf(angle_rad);

    dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
    dim3 grid(
        (W + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D,
        (H + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D,
        B * C);
    rotate_kernel<T><<<grid, block>>>(
        input, output, B, C, H, W, cos_a, sin_a, fill_value);
}

template void launch_rotate<float>(float*, float*, int, int, int, int, float, float);
template void launch_rotate<half>(half*, half*, int, int, int, int, float, float);
template void launch_rotate<uint8_t>(uint8_t*, uint8_t*, int, int, int, int, float, float);
template void launch_rotate<int32_t>(int32_t*, int32_t*, int, int, int, int, float, float);

