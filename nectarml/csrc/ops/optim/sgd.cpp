#include "ops/common.h"
#include "ops/tensor/elementwise/elementwise.h"

/* KERNELS */

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
);

/* NAMESPACE OPS */

namespace nectar {

    uintptr_t sgd_update(
        uintptr_t param_ptr,
        uintptr_t grad_ptr,
        uintptr_t velocity_ptr,
        float     lr,
        float     momentum,
        float     dampening,
        float     weight_decay,
        bool      nesterov,
        bool      maximize,
        int       n_elements
    ) {
        uintptr_t negated_ptr = 0;
        if (maximize) {
            negated_ptr = negate(grad_ptr, n_elements, DType::Float32);
            grad_ptr    = negated_ptr;
        }

        float* param    = reinterpret_cast<float*>(param_ptr);
        float* grad     = reinterpret_cast<float*>(grad_ptr);
        float* velocity = reinterpret_cast<float*>(velocity_ptr);

        launch_sgd(
            param, grad, velocity, lr, momentum, dampening, 
            weight_decay, nesterov, n_elements);

        return reinterpret_cast<uintptr_t>(param);
    }

}


