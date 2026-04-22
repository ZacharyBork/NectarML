#include "kernels/common.h"

struct SGDVanilla {
    __device__ static void apply(
        float& g,
        float* v, 
        float  momentum, 
        float  dampening,
        int    idx
    ) {  }
};

struct SGDMomentum {
    __device__ static void apply(
        float& g,
        float* v, 
        float  momentum, 
        float  dampening,
        int    idx
    ) {
        if (momentum > 0.0) {
            v[idx] = momentum * v[idx] + (1.0f - dampening) * g;
            g = v[idx];
        }
    }
};

struct SGDNesterov {
    __device__ static void apply(
        float& g,
        float* v, 
        float  momentum, 
        float  dampening,
        int    idx
    ) {
        if (momentum > 0.0) {
            v[idx] = momentum * v[idx] + (1.0f - dampening) * g;
            g = g + momentum * v[idx];
        }
    }
};

template<class MomentumPolicy>
__global__ void sgd_kernel(
    float* param,
    float* grad,
    float* velocity,
    float  lr,
    float  momentum,
    float  dampening,
    float  weight_decay,
    int    n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    float g = grad[idx];
    if (weight_decay != 0.0f) g += weight_decay * param[idx];

    MomentumPolicy::apply(g, velocity, momentum, dampening, idx);
    param[idx] -= lr * g;
}

template __global__ void sgd_kernel<SGDVanilla>(
    float*, float*, float*, float, float, float, float, int);
template __global__ void sgd_kernel<SGDMomentum>(
    float*, float*, float*, float, float, float, float, int);
template __global__ void sgd_kernel<SGDNesterov>(
    float*, float*, float*, float, float, float, float, int);

void launch_sgd(
    float* param,
    float* grad,
    float* velocity,
    float  lr,
    float  momentum,
    float  dampening,
    float  weight_decay,
    bool   nesterov,
    int    n_elements
) {
    int threads = 256;
    int blocks  = (n_elements + threads - 1) / threads;
    
    auto launch = [&]<typename Policy>() {
        sgd_kernel<Policy><<<blocks, threads>>>(
            param, grad, velocity,
            lr, momentum, dampening, weight_decay, n_elements);
    };

    if (momentum <= 0.0f) launch.operator()<SGDVanilla>();
    else if (nesterov)    launch.operator()<SGDNesterov>();
    else                  launch.operator()<SGDMomentum>();

}

