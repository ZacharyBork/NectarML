#include "kernels/common.h"

__global__ void adam_update_kernel(
    float* param,
    float* grad,
    float* exp_avg,
    float* exp_avg_sq,
    float  lr,
    float  beta1,
    float  beta2,
    float  eps,
    float  bias_correction1,
    float  bias_correction2,
    float  weight_decay,
    int    n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    float g = grad[idx];
    if (weight_decay != 0.0f) g += weight_decay * param[idx];

    float ea    = beta1 * exp_avg[idx]    + (1.0f - beta1) * g;
    float ea_sq = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g * g;
    
    exp_avg[idx]    = ea;
    exp_avg_sq[idx] = ea_sq;

    float ea_corr    = ea    / bias_correction1;
    float ea_sq_corr = ea_sq / bias_correction2;

    param[idx] -= lr * ea_corr / (sqrtf(ea_sq_corr) + eps);
}

__global__ void adamw_update_kernel(
    float* param,
    float* grad,
    float* exp_avg,
    float* exp_avg_sq,
    float  lr,
    float  beta1,
    float  beta2,
    float  eps,
    float  bias_correction1,
    float  bias_correction2,
    float  weight_decay,
    int    n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    float g = grad[idx];
    if (weight_decay != 0.0f) param[idx] *= (1.0f - lr * weight_decay);

    float ea    = beta1 * exp_avg[idx]    + (1.0f - beta1) * g;
    float ea_sq = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g * g;
    
    exp_avg[idx]    = ea;
    exp_avg_sq[idx] = ea_sq;

    float ea_corr    = ea    / bias_correction1;
    float ea_sq_corr = ea_sq / bias_correction2;

    param[idx] -= lr * ea_corr / (sqrtf(ea_sq_corr) + eps);
}

void launch_adam_update(
    float* param,
    float* grad,
    float* exp_avg,
    float* exp_avg_sq,
    float  lr,
    float  beta1,
    float  beta2,
    float  eps,
    float  bias_correction1,
    float  bias_correction2,
    float  weight_decay,
    bool   decoupled_weight_decay,
    int    n_elements
) {
    int threads = 256;
    int blocks  = (n_elements + threads - 1) / threads;
    if (decoupled_weight_decay) {
        adamw_update_kernel<<<blocks, threads>>>(
            param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps,
            bias_correction1, bias_correction2, weight_decay, n_elements
        );
    }
    else {
        adam_update_kernel<<<blocks, threads>>>(
            param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps,
            bias_correction1, bias_correction2, weight_decay, n_elements
        );
    }
}

