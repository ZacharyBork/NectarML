#include "common/dtype.h"
#include "allocator_pool/allocator_pool.h"

/* KERNELS */

template<typename T, bool align_corners>
void launch_upsample_linear_1d(
    T* input, T* output,
    int B, int C, int L_in, int L_out
);

template<typename T, bool align_corners>
void launch_upsample_linear_1d_backward(
    T* grad_output, T* grad_input,
    int B, int C, int L_in, int L_out
);

template<typename T, bool align_corners>
void launch_upsample_linear_2d(
    T* input, T* output,
    int B, int C, int H_in, int W_in, int H_out, int W_out
);

template<typename T, bool align_corners>
void launch_upsample_linear_2d_backward(
    T* grad_output, T* grad_input,
    int B, int C, int H_in, int W_in, int H_out, int W_out
);

template<typename T, bool align_corners>
void launch_upsample_linear_3d(
    T* input, T* output,
    int B, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
);

template<typename T, bool align_corners>
void launch_upsample_linear_3d_backward(
    T* grad_output, T* grad_input,
    int B, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
);

namespace nectar {

    uintptr_t upsample_linear(
        uintptr_t input_ptr,
        int B, int C, int L_in, int L_out,
        bool align_corners,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out = static_cast<T*>(g_pool.alloc(B * C * L_out * sizeof(T)));
            if (align_corners) {
                launch_upsample_linear_1d<T, true>(
                    reinterpret_cast<T*>(input_ptr), d_out,
                    B, C, L_in, L_out);
            }
            else {
                launch_upsample_linear_1d<T, false>(
                    reinterpret_cast<T*>(input_ptr), d_out,
                    B, C, L_in, L_out);
            }
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t upsample_linear_backward(
        uintptr_t grad_ptr,
        int B, int C, int L_in, int L_out,
        bool align_corners,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input = static_cast<T*>(g_pool.alloc(B * C * L_in * sizeof(T)));
            cudaMemset(d_grad_input, 0, B * C * L_in * sizeof(T));
            if (align_corners) {
                launch_upsample_linear_1d_backward<T, true>(
                    reinterpret_cast<T*>(grad_ptr), d_grad_input,
                    B, C, L_in, L_out);
            }
            else {
                launch_upsample_linear_1d_backward<T, false>(
                    reinterpret_cast<T*>(grad_ptr), d_grad_input,
                    B, C, L_in, L_out);
            }
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }

    uintptr_t upsample_bilinear(
        uintptr_t input_ptr,
        int B, int C, int H_in, int W_in, int H_out, int W_out,
        bool align_corners,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out = static_cast<T*>(g_pool.alloc(B * C * H_out * W_out * sizeof(T)));
            if (align_corners) {
                launch_upsample_linear_2d<T, true>(
                    reinterpret_cast<T*>(input_ptr), d_out,
                    B, C, H_in, W_in, H_out, W_out);
            }
            else {
                launch_upsample_linear_2d<T, false>(
                    reinterpret_cast<T*>(input_ptr), d_out,
                    B, C, H_in, W_in, H_out, W_out);
            }
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t upsample_bilinear_backward(
        uintptr_t grad_ptr,
        int B, int C, int H_in, int W_in, int H_out, int W_out,
        bool align_corners,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input = static_cast<T*>(g_pool.alloc(B * C * H_in * W_in * sizeof(T)));
            cudaMemset(d_grad_input, 0, B * C * H_in * W_in * sizeof(T));
            if (align_corners) {
                launch_upsample_linear_2d_backward<T, true>(
                    reinterpret_cast<T*>(grad_ptr), d_grad_input,
                    B, C, H_in, W_in, H_out, W_out);
            }
            else {
                launch_upsample_linear_2d_backward<T, false>(
                    reinterpret_cast<T*>(grad_ptr), d_grad_input,
                    B, C, H_in, W_in, H_out, W_out);
            }
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }

    uintptr_t upsample_trilinear(
        uintptr_t input_ptr,
        int B, int C,
        int D_in, int H_in, int W_in,
        int D_out, int H_out, int W_out,
        bool align_corners,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_out = static_cast<T*>(g_pool.alloc(B * C * D_out * H_out * W_out * sizeof(T)));
            if (align_corners) {
                launch_upsample_linear_3d<T, true>(
                    reinterpret_cast<T*>(input_ptr), d_out,
                    B, C, D_in, H_in, W_in, D_out, H_out, W_out);
            }
            else {
                launch_upsample_linear_3d<T, false>(
                    reinterpret_cast<T*>(input_ptr), d_out,
                    B, C, D_in, H_in, W_in, D_out, H_out, W_out);
            }
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    uintptr_t upsample_trilinear_backward(
        uintptr_t grad_ptr,
        int B, int C,
        int D_in, int H_in, int W_in,
        int D_out, int H_out, int W_out,
        bool align_corners,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input = static_cast<T*>(g_pool.alloc(B * C * D_in * H_in * W_in * sizeof(T)));
            cudaMemset(d_grad_input, 0, B * C * D_in * H_in * W_in * sizeof(T));
            if (align_corners) {
                launch_upsample_linear_3d_backward<T, true>(
                    reinterpret_cast<T*>(grad_ptr), d_grad_input,
                    B, C, D_in, H_in, W_in, D_out, H_out, W_out);
            }
            else {
                launch_upsample_linear_3d_backward<T, false>(
                    reinterpret_cast<T*>(grad_ptr), d_grad_input,
                    B, C, D_in, H_in, W_in, D_out, H_out, W_out);
            }
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }

}
