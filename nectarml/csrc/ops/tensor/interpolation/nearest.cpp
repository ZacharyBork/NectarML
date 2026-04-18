#include "common.h"
#include "allocator_pool/allocator_pool.h"

/* Kernels */

template<typename T>
void launch_upsample_nearest_1d(
    T* input, T* output,
    int B, int C, int L_in, int L_out
);

template<typename T>
void launch_upsample_nearest_1d_backward(
    T* grad_output, T* grad_input,
    int B, int C, int L_in, int L_out
);

template<typename T>
void launch_upsample_nearest_2d(
    T* input, T* output,
    int B, int C, int H_in, int W_in, int H_out, int W_out
);

template<typename T>
void launch_upsample_nearest_2d_backward(
    T* grad_output, T* grad_input,
    int B, int C, int H_in, int W_in, int H_out, int W_out
);

template<typename T>
void launch_upsample_nearest_3d(
    T* input, T* output,
    int B, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
);

template<typename T>
void launch_upsample_nearest_3d_backward(
    T* grad_output, T* grad_input,
    int B, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
);

namespace nectar {

    uintptr_t upsample_nearest_1d(
        uintptr_t input_ptr,
        int B, int C, int L_in, int L_out,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out = static_cast<T*>(g_pool.alloc(B * C * L_out * sizeof(T)));
            launch_upsample_nearest_1d<T>(
                reinterpret_cast<T*>(input_ptr), d_out,
                B, C, L_in, L_out);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t upsample_nearest_1d_backward(
        uintptr_t grad_ptr,
        int B, int C, int L_in, int L_out,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input = static_cast<T*>(g_pool.alloc(B * C * L_in * sizeof(T)));
            cudaMemset(d_grad_input, 0, B * C * L_in * sizeof(T));
            launch_upsample_nearest_1d_backward<T>(
                reinterpret_cast<T*>(grad_ptr), d_grad_input,
                B, C, L_in, L_out);
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }

    uintptr_t upsample_nearest_2d(
        uintptr_t input_ptr,
        int B, int C, int H_in, int W_in, int H_out, int W_out,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out = static_cast<T*>(g_pool.alloc(B * C * H_out * W_out * sizeof(T)));
            launch_upsample_nearest_2d<T>(
                reinterpret_cast<T*>(input_ptr), d_out,
                B, C, H_in, W_in, H_out, W_out);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t upsample_nearest_2d_backward(
        uintptr_t grad_ptr,
        int B, int C, int H_in, int W_in, int H_out, int W_out,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input = static_cast<T*>(g_pool.alloc(B * C * H_in * W_in * sizeof(T)));
            cudaMemset(d_grad_input, 0, B * C * H_in * W_in * sizeof(T));
            launch_upsample_nearest_2d_backward<T>(
                reinterpret_cast<T*>(grad_ptr), d_grad_input,
                B, C, H_in, W_in, H_out, W_out);
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }

    uintptr_t upsample_nearest_3d(
        uintptr_t input_ptr,
        int B, int C,
        int D_in, int H_in, int W_in,
        int D_out, int H_out, int W_out,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out = static_cast<T*>(g_pool.alloc(B * C * D_out * H_out * W_out * sizeof(T)));
            launch_upsample_nearest_3d<T>(
                reinterpret_cast<T*>(input_ptr), d_out,
                B, C, D_in, H_in, W_in, D_out, H_out, W_out);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t upsample_nearest_3d_backward(
        uintptr_t grad_ptr,
        int B, int C,
        int D_in, int H_in, int W_in,
        int D_out, int H_out, int W_out,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input = static_cast<T*>(g_pool.alloc(B * C * D_in * H_in * W_in * sizeof(T)));
            cudaMemset(d_grad_input, 0, B * C * D_in * H_in * W_in * sizeof(T));
            launch_upsample_nearest_3d_backward<T>(
                reinterpret_cast<T*>(grad_ptr), d_grad_input,
                B, C, D_in, H_in, W_in, D_out, H_out, W_out);
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }
}
