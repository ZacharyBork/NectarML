#include "kernels/common.h"

__global__ void unscale_and_check_grad_kernel(
    float* grad,
    float  inv_scale,
    int*   found_bad,
    int    n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    float val = grad[idx] * inv_scale;
    grad[idx] = val;

    if (!isfinite(val)) atomicOr(found_bad, 1);
}

void launch_unscale_and_check_grad(
    float* grad,
    float  inv_scale,
    int*   found_bad,
    int    n_elements
) {
    int threads = 256;
    int blocks  = (n_elements + threads - 1) / threads;
    unscale_and_check_grad_kernel<<<blocks, threads>>>(
        grad, inv_scale, found_bad, n_elements
    );
}

