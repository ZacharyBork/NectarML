#include "ops/common.h"
#include "ops/tensor/elementwise/elementwise.h"

/* KERNELS */

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
);

/* NAMESPACE OPS */

namespace nectar {

    uintptr_t adam_update(
        uintptr_t param_ptr,
        uintptr_t grad_ptr,
        uintptr_t exp_avg_ptr,
        uintptr_t exp_avg_sq_ptr,
        uintptr_t max_ea_sq_ptr,
        float     lr,
        float     beta1,
        float     beta2,
        float     eps,
        float     bias_correction1,
        float     bias_correction2,
        float     weight_decay,
        bool      decoupled_weight_decay,
        bool      amsgrad,
        bool      maximize,
        int       n_elements
    ) {
        if (maximize) grad_ptr = negate(grad_ptr, n_elements, DType::Float32);

        float* param          = reinterpret_cast<float*>(param_ptr);
        float* grad           = reinterpret_cast<float*>(grad_ptr);
        float* exp_avg        = reinterpret_cast<float*>(exp_avg_ptr);
        float* exp_avg_sq     = reinterpret_cast<float*>(exp_avg_sq_ptr);
        float* max_exp_avg_sq = max_ea_sq_ptr ? 
            reinterpret_cast<float*>(max_ea_sq_ptr) : nullptr;

        launch_adam_update(
            param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, 
            eps, bias_correction1, bias_correction2, weight_decay, 
            amsgrad, decoupled_weight_decay, n_elements);

        return reinterpret_cast<uintptr_t>(param);
    }

}


