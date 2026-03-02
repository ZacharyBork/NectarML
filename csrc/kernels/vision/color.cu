#include <common.h>

__device__ void rgb_to_hsv(float r, float g, float b, float& h, float& s, float& v) {
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

__device__ void hsv_to_rgb(float h, float s, float v, float& r, float& g, float& b) {
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

__global__ void hue_shift_kernel(
    uint8_t* image,
    int width,
    int height,
    float shift
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    float r = image[idx]     / 255.0f;
    float g = image[idx + 1] / 255.0f;
    float b = image[idx + 2] / 255.0f;

    // To HSV
    float h, s, v;
    rgb_to_hsv(r, g, b, h, s, v);

    // Shift hue
    h = fmodf(h + shift, 1.0f);
    if (h < 0.0f) h += 1.0f;

    // To RGB
    hsv_to_rgb(h, s, v, r, g, b);

    // Write back result
    image[idx]     = (uint8_t)(r * 255.0f);
    image[idx + 1] = (uint8_t)(g * 255.0f);
    image[idx + 2] = (uint8_t)(b * 255.0f);
}

void launch_hue_shift(uint8_t* d_image, int width, int height, float shift) {
    int BS2D = BLOCK_SIZE_2D;
    dim3 block(BS2D, BS2D);
    dim3 grid((width + (BS2D-1)) / BS2D, (height + (BS2D-1)) / BS2D);
    hue_shift_kernel<<<grid, block>>>(d_image, width, height, shift);
}
