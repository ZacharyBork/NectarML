#include "common.h"
#include "allocator_pool/allocator_pool.h"

/* KERNELS */

template<typename T>
void launch_im2col_1d(
    T* input,
    T* output,
    int B, int C_in, int L,
    int K, int L_out,
    int stride, int padding, int dilation
);

template<typename T>
void launch_col2im_1d(
    T* col,
    T* input_grad,
    int B, int C_in, int L,
    int K, int L_out,
    int stride, int padding, int dilation
);

template<typename T>
void launch_im2col_2d(
    T* input, T* output,
    int B, int C_in, int H, int W,
    int KH, int KW, int H_out, int W_out,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);

template<typename T>
void launch_col2im_2d(
    T* col, T* input_grad,
    int B, int C_in, int H, int W,
    int KH, int KW, int H_out, int W_out,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);

/* NAMESPACE OPS */

namespace nectar {

    uintptr_t im2col_1d(
        uintptr_t input_ptr,
        int B, int C_in, int L,
        int C_out, int K,
        int stride, int padding, int dilation, 
        int groups,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            int L_out = (L + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

            T* d_col = static_cast<T*>(g_pool.alloc(C_in * K * B * L_out * sizeof(T)));
            
            launch_im2col_1d<T>(
                reinterpret_cast<T*>(input_ptr), d_col,
                B, C_in, L, K, L_out, stride, padding, dilation);
        
            return reinterpret_cast<uintptr_t>(d_col);
        });

    }

    uintptr_t col2im_1d(
        uintptr_t input_ptr,
        int B, int C_in, int L,
        int C_out, int K,
        int stride, int padding, int dilation, 
        int groups,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            int L_out = (L) * stride - 2*padding + dilation*(K-1) + 1;
            int spatial_in  = B * L;
            int kernel_size = C_out * K;

            T* d_out = static_cast<T*>(g_pool.alloc(B * C_out * L_out * sizeof(T)));
            cudaMemset(d_out, 0, B * C_out * L_out * sizeof(T));

            launch_col2im_1d<T>(
                reinterpret_cast<T*>(input_ptr), d_out,
                B, C_out, L_out, K, L,
                stride, padding, dilation);
        
            return reinterpret_cast<uintptr_t>(d_out);
        });

    }

    uintptr_t im2col_2d(
        uintptr_t input_ptr,
        int B, int C_in, int H, int W,
        int C_out, int KH, int KW,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            int H_out = (H + 2*padding_h - dilation_h*(KH-1) - 1) / stride_h + 1;
            int W_out = (W + 2*padding_w - dilation_w*(KW-1) - 1) / stride_w + 1;
            int spatial_out = B * H_out * W_out;
            int kernel_size = C_in * KH * KW;

            T* d_col = static_cast<T*>(g_pool.alloc(kernel_size * spatial_out * sizeof(T)));

            launch_im2col_2d<T>(
                reinterpret_cast<T*>(input_ptr), d_col,
                B, C_in, H, W, KH, KW, H_out, W_out,
                stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w);
        
            return reinterpret_cast<uintptr_t>(d_col);
        });

    }

    uintptr_t col2im_2d(
        uintptr_t input_ptr,
        int B, int C_in, int H, int W,
        int C_out, int KH, int KW,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            int H_out = (H - 1) * stride_h - 2*padding_h + dilation_h*(KH-1) + 1;
            int W_out = (W - 1) * stride_w - 2*padding_w + dilation_w*(KW-1) + 1;
            int spatial_in  = B * H * W;
            int kernel_size = C_out * KH * KW;

            T* d_out = static_cast<T*>(g_pool.alloc(B * C_out * H_out * W_out * sizeof(T)));
            cudaMemset(d_out, 0, B * C_out * H_out * W_out * sizeof(T));

            launch_col2im_2d<T>(
                reinterpret_cast<T*>(input_ptr), d_out,
                B, C_out, H_out, W_out, KH, KW, H, W,
                stride_h, stride_w,
                padding_h, padding_w,
                dilation_h, dilation_w);
        
            return reinterpret_cast<uintptr_t>(d_out);
        });

    }

}
