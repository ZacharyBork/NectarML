// RESOURCES:
// Nick, YouTube, "CUDA Crash Course: Naive 1-D Convolution" : https://www.youtube.com/watch?v=OlLquh9Lnbc
// Nick, Youtube, "CUDA Crash Course: 2-D Convolution" : https://www.youtube.com/watch?v=qxcfco89wvs&t=31s
//
// Zhou, Yangjie, et al., 
//      2021 IEEE International Symposium on Workload Characterization (IISWC). IEEE, (2021), 
//      Characterizing and demystifying the implicit convolution algorithm on commercial matrix-multiplication accelerators.
//      https://arxiv.org/abs/2110.03901

#include "common.h"
#include "ops/device.h"
#include <cublas_v2.h>

/* 1-Dimensional */

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
void launch_add_bias_1d(
    T* output,
    T* bias,
    int B, int C_out, int L_out
);

template<typename T>
void launch_transpose_output_1d(
    T* output, T* bias,
    int B, int C_out, int L_out
);

template<typename T>
uintptr_t run_conv1d(
    uintptr_t input_ptr,
    uintptr_t weight_ptr,
    uintptr_t bias_ptr,
    int B, int C_in, int L,
    int C_out, int K,
    int stride, int padding, int dilation, int groups,
    DType dtype
) {
    // Compute output length
    int L_out = (L + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    T* d_col;
    T* d_out;
    cudaMalloc(&d_col, C_in * K * B * L_out * sizeof(T));
    cudaMalloc(&d_out, C_out * B * L_out * sizeof(T));
    
    // Reshape to columns
    launch_im2col_1d<T>(
        reinterpret_cast<T*>(input_ptr), d_col,
        B, C_in, L, K, L_out, stride, padding, dilation);

    // Matrix multiplication
    // Weight ->   [C_out, C_in * K]
    // Col Matrix: [C_in * K, B * L_out]
    // Result:     [C_out, B * L_out]
    
    cublasHandle_t handle = get_cublas_handle();
    if constexpr (std::is_same_v<T, float>) {
        T alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            B * L_out, C_out, C_in * K,
            &alpha, d_col, B * L_out,
            reinterpret_cast<T*>(weight_ptr), C_in * K,
            &beta, d_out, B * L_out); 
    }
    else if constexpr (std::is_same_v<T, half>) {
        T alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            B * L_out, C_out, C_in * K,
            &alpha, d_col, B * L_out,
            reinterpret_cast<T*>(weight_ptr), C_in * K,
            &beta, d_out, B * L_out); 
    }
    else throw std::runtime_error("conv1d only supports float32 and float16");

    cudaFree(d_col);
    
    // Add bias is applicable
    if (bias_ptr != 0) {
        launch_add_bias_1d<T>(
            d_out, reinterpret_cast<T*>(bias_ptr),
            B, C_out, L_out);
    }

    T* d_final;
    cudaMalloc(&d_final, B * C_out * L_out * sizeof(T));
    launch_transpose_output_1d<T>(d_out, d_final, B, C_out, L_out);
    cudaFree(d_out);
    return reinterpret_cast<uintptr_t>(d_final);
}

template<typename T>
uintptr_t run_conv1d_backward_input(
    uintptr_t out_grad_ptr,
    uintptr_t weight_ptr,
    int B, int C_in, int L,
    int C_out, int K, int L_out,
    int stride, int padding, int dilation,
    DType dtype
) {
    // out_grad -> [C_out, B * L_out]
    // weight   -> [C_out, C_in * K]
    // col_grad  = weight^T @ out_grad = [C_in * K, B * L_out]
    T* d_col_grad;
    cudaMalloc(&d_col_grad, C_in * K * B * L_out * sizeof(T));
    
    cublasHandle_t handle = get_cublas_handle();
    if constexpr (std::is_same_v<T, float>) {
        T alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            B * L_out, C_in * K, C_out,
            &alpha,
            reinterpret_cast<T*>(out_grad_ptr), B * L_out,
            reinterpret_cast<T*>(weight_ptr), C_in * K,
            &beta,
            d_col_grad, B * L_out);
    }
    else if constexpr (std::is_same_v<T, half>) {
        T alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            B * L_out, C_in * K, C_out,
            &alpha,
            reinterpret_cast<T*>(out_grad_ptr), B * L_out,
            reinterpret_cast<T*>(weight_ptr), C_in * K,
            &beta,
            d_col_grad, B * L_out);
    }
    else throw std::runtime_error(
        "conv1d_backward_input only supports float32 and float16");

    T* d_grad_input;
    cudaMalloc(&d_grad_input, B * C_in * L * sizeof(T));
    cudaMemset(d_grad_input, 0, B * C_in * L * sizeof(T));
    
    launch_col2im_1d<T>(
        d_col_grad, d_grad_input,
        B, C_in, L, K, L_out,
        stride, padding, dilation);
    
    cudaFree(d_col_grad);
    return reinterpret_cast<uintptr_t>(d_grad_input);
}

template<typename T>
uintptr_t run_conv1d_backward_weight(
    uintptr_t out_grad_ptr,
    uintptr_t input_ptr,
    int B, int C_in, int L,
    int C_out, int K, int L_out,
    int stride, int padding, int dilation,
    DType dtype
) {
    T* d_col;
    cudaMalloc(&d_col, C_in * K * B * L_out * sizeof(T));
    launch_im2col_1d<T>(
        reinterpret_cast<T*>(input_ptr), d_col,
        B, C_in, L, K, L_out,
        stride, padding, dilation);
    
    T* d_grad_weight;
    cudaMalloc(&d_grad_weight, C_out * C_in * K * sizeof(T));
    
    cublasHandle_t handle = get_cublas_handle();
    if constexpr (std::is_same_v<T, float>) {
        T alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            C_in * K, C_out, B * L_out,
            &alpha,
            d_col, B * L_out,
            reinterpret_cast<T*>(out_grad_ptr), B * L_out, 
            &beta,
            d_grad_weight, C_in * K);
    }
    else if constexpr (std::is_same_v<T, half>) {
        T alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            C_in * K, C_out, B * L_out,
            &alpha,
            d_col, B * L_out,
            reinterpret_cast<T*>(out_grad_ptr), B * L_out, 
            &beta,
            d_grad_weight, C_in * K);
    }
    else throw std::runtime_error(
        "conv1d_backward_weight only supports float32 and float16");
    
    cudaFree(d_col);
    return reinterpret_cast<uintptr_t>(d_grad_weight);
}

/* 2-Dimensional */

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
void launch_transpose_output_2d(
    T* input, T* output,
    int B, int C_out, int H_out, int W_out
);

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
    int groups,
    DType dtype
) {
    int H_out = (H + 2*padding_h - dilation_h*(KH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*padding_w - dilation_w*(KW-1) - 1) / stride_w + 1;
    int spatial_out = B * H_out * W_out;
    int kernel_size = C_in * KH * KW;

    T* d_col;
    T* d_out;
    cudaMalloc(&d_col,   kernel_size * spatial_out * sizeof(T));
    cudaMalloc(&d_out,   C_out * spatial_out * sizeof(T));

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

    cudaFree(d_col);

    if (bias_ptr != 0) {
        launch_add_bias_2d<T>(
            d_out, reinterpret_cast<T*>(bias_ptr),
            B, C_out, H_out, W_out);
    }

    T* d_final;
    cudaMalloc(&d_final, B * C_out * H_out * W_out * sizeof(T));
    launch_transpose_output_2d<T>(d_out, d_final, B, C_out, H_out, W_out);
    cudaFree(d_out);
    return reinterpret_cast<uintptr_t>(d_final);
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
    DType dtype
) {
    int spatial_out = B * H_out * W_out;
    int kernel_size = C_in * KH * KW;

    T* d_col_grad;
    cudaMalloc(&d_col_grad, kernel_size * spatial_out * sizeof(T));

    cublasHandle_t handle = get_cublas_handle();
    if constexpr (std::is_same_v<T, float>) {
        T alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            spatial_out, kernel_size, C_out,
            &alpha,
            reinterpret_cast<T*>(out_grad_ptr), spatial_out,
            reinterpret_cast<T*>(weight_ptr), kernel_size,
            &beta,
            d_col_grad, spatial_out);
    }
    else if constexpr (std::is_same_v<T, half>) {
        T alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            spatial_out, kernel_size, C_out,
            &alpha,
            reinterpret_cast<T*>(out_grad_ptr), spatial_out,
            reinterpret_cast<T*>(weight_ptr), kernel_size,
            &beta,
            d_col_grad, spatial_out);
    }
    else throw std::runtime_error(
        "conv2d_backward_input only supports float32 and float16");
    T* d_grad_input;
    cudaMalloc(&d_grad_input, B * C_in * H * W * sizeof(T));
    cudaMemset(d_grad_input, 0, B * C_in * H * W * sizeof(T));

    launch_col2im_2d<T>(
        d_col_grad, d_grad_input,
        B, C_in, H, W, KH, KW, H_out, W_out,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w);

    cudaFree(d_col_grad);
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
    int dilation_h, int dilation_w,
    DType dtype
) {
    int spatial_out = B * H_out * W_out;
    int kernel_size = C_in * KH * KW;

    T* d_col;
    cudaMalloc(&d_col, kernel_size * spatial_out * sizeof(T));

    launch_im2col_2d<T>(
        reinterpret_cast<T*>(input_ptr), d_col,
        B, C_in, H, W, KH, KW, H_out, W_out,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w);

    T* d_grad_weight;
    cudaMalloc(&d_grad_weight, C_out * C_in * KH * KW * sizeof(T));

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

    cudaFree(d_col);
    return reinterpret_cast<uintptr_t>(d_grad_weight);
}

/* 3-Dimensional */




namespace nectar {

    /* 1-Dimensional */

    uintptr_t conv1d(
        uintptr_t input_ptr,
        uintptr_t weight_ptr,
        uintptr_t bias_ptr,
        int B, int C_in, int L,
        int C_out, int K,
        int stride, int padding, int dilation, int groups,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            return run_conv1d<T>(
                input_ptr, weight_ptr, bias_ptr,
                B, C_in, L, C_out, K,
                stride, padding, dilation, groups, dtype
            );
        });
    }

    uintptr_t conv1d_backward_input(
        uintptr_t out_grad_ptr,
        uintptr_t weight_ptr,
        int B, int C_in, int L,
        int C_out, int K, int L_out,
        int stride, int padding, int dilation,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            return run_conv1d_backward_input<T>(
                out_grad_ptr, weight_ptr, 
                B, C_in, L, C_out, K, L_out,
                stride, padding, dilation, dtype
            );
        });
    }

    uintptr_t conv1d_backward_weight(
        uintptr_t out_grad_ptr,
        uintptr_t input_ptr,
        int B, int C_in, int L,
        int C_out, int K, int L_out,
        int stride, int padding, int dilation,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            return run_conv1d_backward_weight<T>(
                out_grad_ptr, input_ptr, 
                B, C_in, L, C_out, K, L_out,
                stride, padding, dilation, dtype 
            );
        });
    }

    /* 2-Dimensional */

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
                dilation_h, dilation_w, groups, dtype
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
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            return run_conv2d_backward_input<T>(
                out_grad_ptr, weight_ptr,
                B, C_in, H, W, C_out, KH, KW, H_out, W_out,
                stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w, dtype
            );
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
                dilation_h, dilation_w, dtype
            );
        });
    }

    /* 3-Dimensional */



}


