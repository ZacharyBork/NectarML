#include "ops/common.h"

#include <iostream>

/* KERNELS */

void launch_unscale_and_check_grad(
    float* grad,
    float  inv_scale,
    int*   found_bad,
    int    n_elements
);

/* NAMESPACE OPS */

namespace nectar {

    bool unscale_and_check_grad(
        uintptr_t grad_ptr,
        float     inv_scale,
        int       n_elements
    ) {
        int* found_bad;
        cudaMalloc(&found_bad, sizeof(int));
        cudaMemset(found_bad, 0, sizeof(int));
        
        launch_unscale_and_check_grad(
            reinterpret_cast<float*>(grad_ptr), 
            inv_scale, found_bad, n_elements);
        
        int result = 0;
        cudaMemcpy(&result, found_bad, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(found_bad);
        return result != 0;
    }

}


