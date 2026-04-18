// RESOURCES:
// Nick, YouTube, "CUDA Crash Course: Naive 1-D Convolution" : https://www.youtube.com/watch?v=OlLquh9Lnbc
// Nick, Youtube, "CUDA Crash Course: 2-D Convolution" : https://www.youtube.com/watch?v=qxcfco89wvs&t=31s
//
// Zhou, Yangjie, et al., 
//      2021 IEEE International Symposium on Workload Characterization (IISWC). IEEE, (2021), 
//      Characterizing and demystifying the implicit convolution algorithm on commercial matrix-multiplication accelerators.
//      https://arxiv.org/abs/2110.03901

#include "common/dtype.h"
#include "ops/system/device.h"
#include "allocator_pool/allocator_pool.h"
#include <cublas_v2.h>

/* KERNELS */

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

template<typename T>
void launch_add_bias_2d(
    T* output, T* bias,
    int B, int C_out, int H_out, int W_out
);

template<typename T>
void launch_add_bias_2d_nchw(
    T* output, T* bias,
    int B, int C_out, int H_out, int W_out
);

template<typename T>
void launch_transpose_output_2d(
    T* input, T* output,
    int B, int C_out, int H_out, int W_out
);

template<typename T>
void launch_transpose_input_2d(
    T* input, T* output,
    int B, int C, int H, int W
);

/* FUNCTIONS */

template<typename T>
uintptr_t run_conv2d(
    uintptr_t input_ptr,
    uintptr_t weight_ptr,
    uintptr_t bias_ptr,
    int B, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    int H_out = (H + 2*padding_h - dilation_h*(KH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*padding_w - dilation_w*(KW-1) - 1) / stride_w + 1;
    int spatial_out = B * H_out * W_out;
    int kernel_size = C_in * KH * KW;

    T* d_col = static_cast<T*>(g_pool.alloc(kernel_size * spatial_out * sizeof(T)));
    T* d_out = static_cast<T*>(g_pool.alloc(C_out * spatial_out * sizeof(T)));

    launch_im2col_2d<T>(
        reinterpret_cast<T*>(input_ptr), d_col,
        B, C_in, H, W, KH, KW, H_out, W_out,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w);

    cublasHandle_t handle = get_cublas_handle();
    if constexpr (std::is_same_v<T, float>) {
        T alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            spatial_out, C_out, kernel_size,
            &alpha,
            d_col, spatial_out,
            reinterpret_cast<T*>(weight_ptr), kernel_size,
            &beta,
            d_out, spatial_out);
    }
    else if constexpr (std::is_same_v<T, half>) {
        T alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            spatial_out, C_out, kernel_size,
            &alpha,
            d_col, spatial_out,
            reinterpret_cast<T*>(weight_ptr), kernel_size,
            &beta,
            d_out, spatial_out);
    }
    else throw std::runtime_error("conv2d only supports float32 and float16");

    g_pool.free(d_col, kernel_size * spatial_out * sizeof(T));

    if (bias_ptr != 0) {
        launch_add_bias_2d<T>(
            d_out, reinterpret_cast<T*>(bias_ptr),
            B, C_out, H_out, W_out);
    }

    T* d_final = static_cast<T*>(g_pool.alloc(B * C_out * H_out * W_out * sizeof(T)));
    launch_transpose_output_2d<T>(d_out, d_final, B, C_out, H_out, W_out);
    g_pool.free(d_out, C_out * spatial_out * sizeof(T));
    return reinterpret_cast<uintptr_t>(d_final);
}

template<typename T>
uintptr_t run_conv_transpose2d(
    uintptr_t input_ptr,
    uintptr_t weight_ptr,
    uintptr_t bias_ptr,
    int B, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int output_padding_h, int output_padding_w,
    int groups
) {
    int H_out = (H - 1) * stride_h - 2*padding_h
              + dilation_h*(KH-1) + 1 + output_padding_h;
    int W_out = (W - 1) * stride_w - 2*padding_w
              + dilation_w*(KW-1) + 1 + output_padding_w;

    int spatial_in  = B * H * W;
    int kernel_size = C_out * KH * KW;

    T* d_input = static_cast<T*>(g_pool.alloc(B * C_in * H * W * sizeof(T)));
    launch_transpose_input_2d<T>(
        reinterpret_cast<T*>(input_ptr), d_input,
        B, C_in, H, W);

    T* d_col = static_cast<T*>(g_pool.alloc(kernel_size * spatial_in * sizeof(T)));

    cublasHandle_t handle = get_cublas_handle();
    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            spatial_in, kernel_size, C_in,
            &alpha,
            d_input, C_in,
            reinterpret_cast<T*>(weight_ptr), kernel_size,
            &beta,
            d_col, spatial_in);
    }
    else if constexpr (std::is_same_v<T, half>) {
        half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            spatial_in, kernel_size, C_in,
            &alpha,
            d_input, C_in,
            reinterpret_cast<T*>(weight_ptr), kernel_size,
            &beta,
            d_col, spatial_in);
    }
    else throw std::runtime_error(
        "conv_transpose2d only supports float32 and float16");

    g_pool.free(d_input, B * C_in * H * W * sizeof(T));

    T* d_out = static_cast<T*>(g_pool.alloc(B * C_out * H_out * W_out * sizeof(T)));
    cudaMemset(d_out, 0, B * C_out * H_out * W_out * sizeof(T));

    launch_col2im_2d<T>(
        d_col, d_out,
        B, C_out, H_out, W_out, KH, KW, H, W,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w);

    g_pool.free(d_col, kernel_size * spatial_in * sizeof(T));

    if (bias_ptr != 0) {
        launch_add_bias_2d_nchw<T>(
            d_out, reinterpret_cast<T*>(bias_ptr),
            B, C_out, H_out, W_out);
    }

    return reinterpret_cast<uintptr_t>(d_out);
}

template<typename T>
uintptr_t run_conv2d_backward_input(
    uintptr_t out_grad_ptr,
    uintptr_t weight_ptr,
    int B, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    int spatial_out = B * H_out * W_out;
    int kernel_size = C_in * KH * KW;

    T* d_out_grad = static_cast<T*>(g_pool.alloc(B * C_out * H_out * W_out * sizeof(T)));
    launch_transpose_input_2d<T>(
        reinterpret_cast<T*>(out_grad_ptr), d_out_grad,
        B, C_out, H_out, W_out);

    T* d_col_grad = static_cast<T*>(g_pool.alloc(kernel_size * spatial_out * sizeof(T)));

    cublasHandle_t handle = get_cublas_handle();
    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            spatial_out, kernel_size, C_out,
            &alpha,
            d_out_grad, spatial_out,
            reinterpret_cast<T*>(weight_ptr), kernel_size,
            &beta,
            d_col_grad, spatial_out);
    }
    else if constexpr (std::is_same_v<T, half>) {
        half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            spatial_out, kernel_size, C_out,
            &alpha,
            d_out_grad, spatial_out,
            reinterpret_cast<T*>(weight_ptr), kernel_size,
            &beta,
            d_col_grad, spatial_out);
    }
    else throw std::runtime_error(
        "conv2d_backward_input only supports float32 and float16");

    g_pool.free(d_out_grad, B * C_out * H_out * W_out * sizeof(T));

    T* d_grad_input = static_cast<T*>(g_pool.alloc(B * C_in * H * W * sizeof(T)));
    cudaMemset(d_grad_input, 0, B * C_in * H * W * sizeof(T));

    launch_col2im_2d<T>(
        d_col_grad, d_grad_input,
        B, C_in, H, W, KH, KW, H_out, W_out,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w);

    g_pool.free(d_col_grad, kernel_size * spatial_out * sizeof(T));
    return reinterpret_cast<uintptr_t>(d_grad_input);
}

template<typename T>
uintptr_t run_conv2d_backward_weight(
    uintptr_t out_grad_ptr,
    uintptr_t input_ptr,
    int B, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    int spatial_out = B * H_out * W_out;
    int kernel_size = C_in * KH * KW;

    T* d_col = static_cast<T*>(g_pool.alloc(kernel_size * spatial_out * sizeof(T)));

    launch_im2col_2d<T>(
        reinterpret_cast<T*>(input_ptr), d_col,
        B, C_in, H, W, KH, KW, H_out, W_out,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w);

    T* d_grad_weight = static_cast<T*>(g_pool.alloc( C_out * C_in * KH * KW * sizeof(T)));

    cublasHandle_t handle = get_cublas_handle();
    if constexpr (std::is_same_v<T, float>) {
        T alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            kernel_size, C_out, spatial_out,
            &alpha,
            d_col, spatial_out,
            reinterpret_cast<T*>(out_grad_ptr), spatial_out,
            &beta,
            d_grad_weight, kernel_size);
    }
    else if constexpr (std::is_same_v<T, half>) {
        T alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            kernel_size, C_out, spatial_out,
            &alpha,
            d_col, spatial_out,
            reinterpret_cast<T*>(out_grad_ptr), spatial_out,
            &beta,
            d_grad_weight, kernel_size);
    }
    else throw std::runtime_error(
        "conv2d_backward_weight only supports float32 and float16");

    g_pool.free(d_col, kernel_size * spatial_out * sizeof(T));
    return reinterpret_cast<uintptr_t>(d_grad_weight);
}

template<typename T>
uintptr_t run_conv_transpose2d_backward_weight(
    uintptr_t out_grad_ptr,
    uintptr_t input_ptr,
    int B, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    int spatial_in  = B * H * W;
    int kernel_size = C_out * KH * KW;

    T* d_col = static_cast<T*>(g_pool.alloc(kernel_size * spatial_in * sizeof(T)));
    launch_im2col_2d<T>(
        reinterpret_cast<T*>(out_grad_ptr), d_col,
        B, C_out, H_out, W_out, KH, KW, H, W,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w);

    T* d_input_t = static_cast<T*>(g_pool.alloc(B * C_in * H * W * sizeof(T)));
    launch_transpose_input_2d<T>(
        reinterpret_cast<T*>(input_ptr), d_input_t,
        B, C_in, H, W);

    T* d_grad_weight = static_cast<T*>(g_pool.alloc(C_in * C_out * KH * KW * sizeof(T)));

    cublasHandle_t handle = get_cublas_handle();
    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            kernel_size, C_in, spatial_in,
            &alpha,
            d_col, spatial_in,
            d_input_t, C_in,
            &beta,
            d_grad_weight, kernel_size);
    }
    else if constexpr (std::is_same_v<T, half>) {
        half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            kernel_size, C_in, spatial_in,
            &alpha,
            d_col, spatial_in,
            d_input_t, C_in,
            &beta,
            d_grad_weight, kernel_size);
    }
    else throw std::runtime_error(
        "conv_transpose2d_backward_weight only supports float32 and float16");

    g_pool.free(d_col, kernel_size * spatial_in * sizeof(T));
    g_pool.free(d_input_t, B * C_in * H * W * sizeof(T));
    return reinterpret_cast<uintptr_t>(d_grad_weight);
}

/* NAMESPACE OPS */

namespace nectar {

    uintptr_t conv2d(
        uintptr_t input_ptr,
        uintptr_t weight_ptr,
        uintptr_t bias_ptr,
        int B, int C_in, int H, int W,
        int C_out, int KH, int KW,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w,
        int groups,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            return run_conv2d<T>(
                input_ptr, weight_ptr, bias_ptr,
                B, C_in, H, W, C_out, KH, KW,
                stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w, groups
            );
        });
    }

    uintptr_t conv_transpose2d(
        uintptr_t input_ptr,
        uintptr_t weight_ptr,
        uintptr_t bias_ptr,
        int B, int C_in, int H, int W,
        int C_out, int KH, int KW,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w,
        int output_padding_h, int output_padding_w,
        int groups,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            return run_conv_transpose2d<T>(
                input_ptr, weight_ptr, bias_ptr,
                B, C_in, H, W, C_out, KH, KW,
                stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w, output_padding_h, output_padding_w,
                groups
            );
        });
    }

    uintptr_t conv2d_backward_input(
        uintptr_t out_grad_ptr,
        uintptr_t weight_ptr,
        int B, int C_in, int H, int W,
        int C_out, int KH, int KW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w,
        int groups,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            return run_conv2d_backward_input<T>(
                out_grad_ptr, weight_ptr,
                B, C_in, H, W, C_out, KH, KW, H_out, W_out,
                stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w, groups);
        });
    }

    uintptr_t conv_transpose2d_backward_input(
        uintptr_t out_grad_ptr,
        uintptr_t weight_ptr,
        int B, int C_in, int H, int W,
        int C_out, int KH, int KW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w,
        int groups,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            return run_conv2d<T>(
                out_grad_ptr, weight_ptr, 0,
                B, C_out, H_out, W_out,
                C_in, KH, KW,
                stride_h, stride_w,
                padding_h, padding_w,
                dilation_h, dilation_w,
                groups);
        });
    }

    uintptr_t conv2d_backward_weight(
        uintptr_t out_grad_ptr,
        uintptr_t input_ptr,
        int B, int C_in, int H, int W,
        int C_out, int KH, int KW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            return run_conv2d_backward_weight<T>(
                out_grad_ptr, input_ptr,
                B, C_in, H, W, C_out, KH, KW, H_out, W_out,
                stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w
            );
        });
    }

    uintptr_t conv_transpose2d_backward_weight(
    uintptr_t out_grad_ptr,
    uintptr_t input_ptr,
    int B, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    DType dtype
) {
    DISPATCH_DTYPE(dtype, T, {
        return run_conv_transpose2d_backward_weight<T>(
            out_grad_ptr, input_ptr,
            B, C_in, H, W, C_out, KH, KW, H_out, W_out,
            stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w);
    });
}

}


