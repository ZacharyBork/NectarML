#include "common.h"
#include "allocator_pool/allocator_pool.h"

/* KERNELS */

template<typename T, bool align_corners>
void launch_upsample_bicubic(
    T* input, T* output,
    int B, int C, 
    int H_in, int W_in, 
    int H_out, int W_out,
    float a
);

template<typename T, bool align_corners>
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
        float a, bool align_corners,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out = static_cast<T*>(g_pool.alloc(B * C * H_out * W_out * sizeof(T)));
            if (align_corners) {
                launch_upsample_bicubic<T, true>(
                    reinterpret_cast<T*>(input_ptr), d_out,
                    B, C, H_in, W_in, H_out, W_out, a);
            }
            else {
                launch_upsample_bicubic<T, false>(
                    reinterpret_cast<T*>(input_ptr), d_out,
                    B, C, H_in, W_in, H_out, W_out, a);
            }
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t upsample_bicubic_backward(
        uintptr_t grad_ptr,
        int B, int C, 
        int H_in, int W_in, 
        int H_out, int W_out,
        float a, bool align_corners,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input = static_cast<T*>(g_pool.alloc( B * C * H_in * W_in * sizeof(T)));
            cudaMemset(d_grad_input, 0, B * C * H_in * W_in * sizeof(T));
            if (align_corners) {
                launch_upsample_bicubic_backward<T, true>(
                    reinterpret_cast<T*>(grad_ptr), d_grad_input,
                    B, C, H_in, W_in, H_out, W_out, a);
            }
            else {
                launch_upsample_bicubic_backward<T, false>(
                    reinterpret_cast<T*>(grad_ptr), d_grad_input,
                    B, C, H_in, W_in, H_out, W_out, a);
            }
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }

}

