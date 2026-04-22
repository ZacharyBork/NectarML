#include "kernels/common.h"

struct AdamVanilla {
    __device__ static void apply(
        float& g,
        float* param, 
        float  lr,
        float  weight_decay, 
        int    idx
    ) { 
        g += weight_decay * param[idx];
    }
};

struct AdamW {
    __device__ static void apply(
        float& g,
        float* param, 
        float  lr,
        float  weight_decay, 
        int    idx
    ) { 
        param[idx] *= 1.0f - lr * weight_decay;
    }
};

struct AMSGradOff {
    __device__ static float denom(
        float  ea_sq_corr, 
        float* max_ea_sq, 
        float  eps, 
        int    idx
    ) {
        return sqrtf(ea_sq_corr) + eps;
    }
};

struct AMSGradOn {
    __device__ static float denom(
        float  ea_sq_corr, 
        float* max_ea_sq, 
        float  eps, 
        int    idx
    ) {
        float m = fmaxf(max_ea_sq[idx], ea_sq_corr);
        max_ea_sq[idx] = m;
        return sqrtf(m) + eps;
    }
};
template<class Decay, class AMSGrad>
__global__ void adam_kernel(
    float* param,
    float* grad,
    float* exp_avg,
    float* exp_avg_sq,
    float* max_ea_sq,
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
    if (weight_decay != 0.0f) Decay::apply(g, param, lr, weight_decay, idx);

    float ea    = beta1 * exp_avg[idx]    + (1.0f - beta1) * g;
    float ea_sq = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g * g;
    
    exp_avg[idx]    = ea;
    exp_avg_sq[idx] = ea_sq;

    float ea_corr    = ea    / bias_correction1;
    float ea_sq_corr = ea_sq / bias_correction2;

    float denom = AMSGrad::denom(ea_sq_corr, max_ea_sq, eps, idx);

    param[idx] -= lr * ea_corr / denom;
}

template __global__ void adam_kernel<AdamVanilla, AMSGradOn>(
    float*, float*, float*, float*, float*, 
    float, float, float, float, float, float, float, int
);
template __global__ void adam_kernel<AdamVanilla, AMSGradOff>(
    float*, float*, float*, float*, float*, 
    float, float, float, float, float, float, float, int
);
template __global__ void adam_kernel<AdamW, AMSGradOn>(
    float*, float*, float*, float*, float*, 
    float, float, float, float, float, float, float, int
);
template __global__ void adam_kernel<AdamW, AMSGradOff>(
    float*, float*, float*, float*, float*, 
    float, float, float, float, float, float, float, int
);

void launch_adam_update(
    float* param,
    float* grad,
    float* exp_avg,
    float* exp_avg_sq,
    float* max_ea_sq,
    float  lr,
    float  beta1,
    float  beta2,
    float  eps,
    float  bias_correction1,
    float  bias_correction2,
    float  weight_decay,
    bool   amsgrad,
    bool   decoupled_weight_decay,
    int    n_elements
) {
    int threads = 256;
    int blocks  = (n_elements + threads - 1) / threads;

    auto launch = [&]<typename DecayPolicy, typename AMSGradPolicy>() {
        adam_kernel<DecayPolicy, AMSGradPolicy><<<blocks, threads>>>(
            param, grad, exp_avg, exp_avg_sq, max_ea_sq, lr, beta1, beta2, 
            eps, bias_correction1, bias_correction2, weight_decay, n_elements);
    };

    if (decoupled_weight_decay) {
        if (amsgrad) launch.operator()<AdamW, AMSGradOn>();
        else         launch.operator()<AdamW, AMSGradOff>();
    }
    else {
        if (amsgrad) launch.operator()<AdamVanilla, AMSGradOn>();
        else         launch.operator()<AdamVanilla, AMSGradOff>();
    }
}

