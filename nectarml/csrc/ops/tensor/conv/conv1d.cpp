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
void launch_add_bias_1d(
    T* output,
    T* bias,
    int B, int C_out, int L_out
);

template<typename T>
void launch_add_bias_1d_nchw(
    T* output, T* bias,
    int B, int C_out, int L_out
);

template<typename T>
void launch_transpose_output_1d(
    T* output, T* bias,
    int B, int C_out, int L_out
);

template<typename T>
void launch_transpose_input_1d(
    T* input, T* output,
    int B, int C_out, int L_out
);

/* FUNCTIONS */

template<typename T>
uintptr_t run_conv1d(
    uintptr_t input_ptr,
    uintptr_t weight_ptr,
    uintptr_t bias_ptr,
    int B, int C_in, int L,
    int C_out, int K,
    int stride, int padding, int dilation, 
    int groups
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
uintptr_t run_conv_transpose1d(
    uintptr_t input_ptr,
    uintptr_t weight_ptr,
    uintptr_t bias_ptr,
    int B, int C_in, int L_in,
    int C_out, int K,
    int stride, int padding, int dilation, int groups,
    int output_padding
) {
    int L_out = (L_in - 1) * stride - 2*padding + dilation*(K-1) + 1 + output_padding;
    int spatial_in  = B * L_in;
    int kernel_size = C_out * K;

    T* d_input;
    cudaMalloc(&d_input, B * C_in * L_in * sizeof(T));
    launch_transpose_input_1d<T>(
        reinterpret_cast<T*>(input_ptr), d_input,
        B, C_in, L_in);

    T* d_col;
    cudaMalloc(&d_col, kernel_size * spatial_in * sizeof(T));

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
        "conv_transpose1d only supports float32 and float16");

    cudaFree(d_input);

    T* d_out;
    cudaMalloc(&d_out, B * C_out * L_out * sizeof(T));
    cudaMemset(d_out, 0, B * C_out * L_out * sizeof(T));

    launch_col2im_1d<T>(
        d_col, d_out,
        B, C_out, L_out, K, L_in,
        stride, padding, dilation);

    cudaFree(d_col);

    if (bias_ptr != 0) {
        launch_add_bias_1d_nchw<T>(
            d_out, reinterpret_cast<T*>(bias_ptr),
            B, C_out, L_out);
    }

    return reinterpret_cast<uintptr_t>(d_out);
}

template<typename T>
uintptr_t run_conv1d_backward_input(
    uintptr_t out_grad_ptr,
    uintptr_t weight_ptr,
    int B, int C_in, int L,
    int C_out, int K, int L_out,
    int stride, int padding, int dilation,
    int groups
) {
    T* d_out_grad;
    cudaMalloc(&d_out_grad, B * C_out * L_out * sizeof(T));
    launch_transpose_input_1d<T>(
        reinterpret_cast<T*>(out_grad_ptr), d_out_grad,
        B, C_out, L_out);

    T* d_col_grad;
    cudaMalloc(&d_col_grad, C_in * K * B * L_out * sizeof(T));

    cublasHandle_t handle = get_cublas_handle();
    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            B * L_out, C_in * K, C_out,
            &alpha,
            d_out_grad, B * L_out,
            reinterpret_cast<T*>(weight_ptr), C_in * K,
            &beta,
            d_col_grad, B * L_out);
    }
    else if constexpr (std::is_same_v<T, half>) {
        half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            B * L_out, C_in * K, C_out,
            &alpha,
            d_out_grad, B * L_out,
            reinterpret_cast<T*>(weight_ptr), C_in * K,
            &beta,
            d_col_grad, B * L_out);
    }
    else throw std::runtime_error(
        "conv1d_backward_input only supports float32 and float16");

    cudaFree(d_out_grad);

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
    int stride, int padding, int dilation
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

template<typename T>
uintptr_t run_conv_transpose1d_backward_weight(
    uintptr_t out_grad_ptr,
    uintptr_t input_ptr,
    int B, int C_in, int L_in,
    int C_out, int K, int L_out,
    int stride, int padding, int dilation
) {
    T* d_col;
    cudaMalloc(&d_col, C_out * K * B * L_in * sizeof(T));
    launch_im2col_1d<T>(
        reinterpret_cast<T*>(out_grad_ptr), d_col,
        B, C_out, L_out, K, L_in,
        stride, padding, dilation);

    T* d_input_t;
    cudaMalloc(&d_input_t, B * C_in * L_in * sizeof(T));
    launch_transpose_input_1d<T>(
        reinterpret_cast<T*>(input_ptr), d_input_t,
        B, C_in, L_in);

    T* d_grad_weight;
    cudaMalloc(&d_grad_weight, C_in * C_out * K * sizeof(T));

    cublasHandle_t handle = get_cublas_handle();
    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            C_out * K, C_in, B * L_in,
            &alpha,
            d_col, B * L_in,
            d_input_t, C_in,
            &beta,
            d_grad_weight, C_out * K);
    }
    else if constexpr (std::is_same_v<T, half>) {
        half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            C_out * K, C_in, B * L_in,
            &alpha,
            d_col, B * L_in,
            d_input_t, C_in,
            &beta,
            d_grad_weight, C_out * K);
    }
    else throw std::runtime_error(
        "conv_transpose1d_backward_weight only supports float32 and float16");

    cudaFree(d_col);
    cudaFree(d_input_t);
    return reinterpret_cast<uintptr_t>(d_grad_weight);
}

/* NAMESPACE OPS */

namespace nectar {

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
                stride, padding, dilation, groups
            );
        });
    }

    uintptr_t conv_transpose1d(
        uintptr_t input_ptr,
        uintptr_t weight_ptr,
        uintptr_t bias_ptr,
        int B, int C_in, int L,
        int C_out, int K,
        int stride, int padding, int dilation, 
        int output_padding, 
        int groups,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            return run_conv_transpose1d<T>(
                input_ptr, weight_ptr, bias_ptr,
                B, C_in, L, C_out, K,
                stride, padding, dilation, 
                groups, output_padding
            );
        });
    }

    uintptr_t conv1d_backward_input(
        uintptr_t out_grad_ptr,
        uintptr_t weight_ptr,
        int B, int C_in, int L,
        int C_out, int K, int L_out,
        int stride, int padding, int dilation, int groups,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            return run_conv1d_backward_input<T>(
                out_grad_ptr, weight_ptr,
                B, C_in, L, C_out, K, L_out,
                stride, padding, dilation, groups);
        });
    }

    uintptr_t conv_transpose1d_backward_input(
        uintptr_t out_grad_ptr,
        uintptr_t weight_ptr,
        int B, int C_in, int L,
        int C_out, int K, int L_out,
        int stride, int padding, int dilation, int groups,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            return run_conv1d<T>(
                out_grad_ptr, weight_ptr, 0,
                B, C_out, L_out, C_in, K,
                stride, padding, dilation, groups);
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
                stride, padding, dilation 
            );
        });
    }

    uintptr_t conv_transpose1d_backward_weight(
        uintptr_t out_grad_ptr,
        uintptr_t input_ptr,
        int B, int C_in, int L,
        int C_out, int K, int L_out,
        int stride, int padding, int dilation,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            return run_conv_transpose1d_backward_weight<T>(
                out_grad_ptr, input_ptr,
                B, C_in, L, C_out, K, L_out,
                stride, padding, dilation);
        });
    }

}
