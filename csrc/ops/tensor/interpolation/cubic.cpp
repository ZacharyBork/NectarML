#include "common.h"

/* KERNELS */

template<typename T>
void launch_upsample_bicubic(
    T* input, T* output,
    int B, int C, 
    int H_in, int W_in, 
    int H_out, int W_out,
    float a
);

template<typename T>
void launch_upsample_bicubic_backward(
    T* grad_output, T* grad_input,
    int B, int C, 
    int H_in, int W_in, 
    int H_out, int W_out,
    float a
);

namespace nectar {

    uintptr_t upsample_bicubic(
        uintptr_t input_ptr,
        int B, int C, 
        int H_in, int W_in, 
        int H_out, int W_out,
        float a,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, B * C * H_out * W_out * sizeof(T));
            launch_upsample_bicubic<T>(
                reinterpret_cast<T*>(input_ptr), d_out,
                B, C, H_in, W_in, H_out, W_out, a);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t upsample_bicubic_backward(
        uintptr_t grad_ptr,
        int B, int C, 
        int H_in, int W_in, 
        int H_out, int W_out,
        float a,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input;
            cudaMalloc(&d_grad_input, B * C * H_in * W_in * sizeof(T));
            cudaMemset(d_grad_input, 0, B * C * H_in * W_in * sizeof(T));
            launch_upsample_bicubic_backward<T>(
                reinterpret_cast<T*>(grad_ptr), d_grad_input,
                B, C, H_in, W_in, H_out, W_out, a);
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }

}

