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
void launch_transpose_output_1d(
    T* output, T* bias,
    int B, int C_out, int L_out
);

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
        // Compute output length
        int L_out = (L + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

        if (dtype == DType::Float32) {
            float* d_col;
            float* d_final;
            cudaMalloc(&d_col, C_in * K * B * L_out * sizeof(float));
            cudaMalloc(&d_final, B * C_out * L_out * sizeof(float));
            
            // Reshape to columns
            launch_im2col_1d<float>(
                reinterpret_cast<float*>(input_ptr), d_col,
                B, C_in, L, K, L_out,
                stride, padding, dilation);

            // Matrix multiplication
            // Weight ->   [C_out, C_in * K]
            // Col Matrix: [C_in * K, B * L_out]
            // Result:     [C_out, B * L_out]
            float* d_out;
            cudaMalloc(&d_out, C_out * B * L_out * sizeof(float));
            
            cublasHandle_t handle = get_cublas_handle();
            float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                B * L_out, C_out, C_in * K,
                &alpha,
                d_col, B * L_out,
                reinterpret_cast<float*>(weight_ptr), C_in * K,
                &beta,
                d_out, B * L_out); 
            
            cudaFree(d_col);
            
            // Add bias is applicable
            if (bias_ptr != 0) {
                launch_add_bias_1d<float>(
                    d_out, reinterpret_cast<float*>(bias_ptr),
                    B, C_out, L_out);
            }

            launch_transpose_output_1d<float>(d_out, d_final, B, C_out, L_out);
            cudaFree(d_out);
            return reinterpret_cast<uintptr_t>(d_final);
        } 
        else if (dtype == DType::Float16) {
            // Same deal as float32, just for half-precision floats
            half* d_col;
            half* d_final;
            cudaMalloc(&d_col, C_in * K * B * L_out * sizeof(half));
            cudaMalloc(&d_final, B * C_out * L_out * sizeof(half));
            
            launch_im2col_1d<half>(
                reinterpret_cast<half*>(input_ptr), d_col,
                B, C_in, L, K, L_out,
                stride, padding, dilation);
            
            half* d_out;
            cudaMalloc(&d_out, C_out * B * L_out * sizeof(half));
            
            cublasHandle_t handle = get_cublas_handle();
            half alpha = __float2half(1.0f), beta = __float2half(0.0f);
            cublasHgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                B * L_out, C_out, C_in * K,
                &alpha,
                d_col, B * L_out,
                reinterpret_cast<half*>(weight_ptr), C_in * K,
                &beta,
                d_out, B * L_out); 
            
            cudaFree(d_col);
            
            if (bias_ptr != 0) {
                launch_add_bias_1d<half>(
                    d_out, reinterpret_cast<half*>(bias_ptr),
                    B, C_out, L_out);
            }
            
            launch_transpose_output_1d<half>(d_out, d_final, B, C_out, L_out);
            cudaFree(d_out);
            return reinterpret_cast<uintptr_t>(d_final);
        } 
        else {
            throw std::runtime_error(
                "conv1d only supports float32 and float16");
        }
    }

    uintptr_t conv1d_backward_input(
        uintptr_t out_grad_ptr,
        uintptr_t weight_ptr,
        int B, int C_in, int L,
        int C_out, int K, int L_out,
        int stride, int padding, int dilation,
        DType dtype
    ) {
        if (dtype == DType::Float32) {
            // out_grad -> [C_out, B * L_out]
            // weight   -> [C_out, C_in * K]
            // col_grad  = weight^T @ out_grad = [C_in * K, B * L_out]
            float* d_col_grad;
            cudaMalloc(&d_col_grad, C_in * K * B * L_out * sizeof(float));
            
            cublasHandle_t handle = get_cublas_handle();
            float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                B * L_out, C_in * K, C_out,
                &alpha,
                reinterpret_cast<float*>(out_grad_ptr), B * L_out,
                reinterpret_cast<float*>(weight_ptr), C_in * K,
                &beta,
                d_col_grad, B * L_out);
            
            float* d_grad_input;
            cudaMalloc(&d_grad_input, B * C_in * L * sizeof(float));
            cudaMemset(d_grad_input, 0, B * C_in * L * sizeof(float));
            
            launch_col2im_1d<float>(
                d_col_grad, d_grad_input,
                B, C_in, L, K, L_out,
                stride, padding, dilation);
            
            cudaFree(d_col_grad);
            return reinterpret_cast<uintptr_t>(d_grad_input);
        }
        else if (dtype == DType::Float16) {
            half* d_col_grad;
            cudaMalloc(&d_col_grad, C_in * K * B * L_out * sizeof(half));
            
            cublasHandle_t handle = get_cublas_handle();
            half alpha = __float2half(1.0f), beta = __float2half(0.0f);
            cublasHgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                B * L_out, C_in * K, C_out,
                &alpha,
                reinterpret_cast<half*>(out_grad_ptr), B * L_out,
                reinterpret_cast<half*>(weight_ptr), C_in * K,
                &beta,
                d_col_grad, B * L_out);
            
            half* d_grad_input;
            cudaMalloc(&d_grad_input, B * C_in * L * sizeof(half));
            cudaMemset(d_grad_input, 0, B * C_in * L * sizeof(half));
            
            launch_col2im_1d<half>(
                reinterpret_cast<half*>(d_col_grad), 
                reinterpret_cast<half*>(d_grad_input),
                B, C_in, L, K, L_out,
                stride, padding, dilation);
            
            cudaFree(d_col_grad);
            return reinterpret_cast<uintptr_t>(d_grad_input);
        }
        else {
            throw std::runtime_error(
                "conv1d_backward_input only supports float32 and float16");
        }
    }

    uintptr_t conv1d_backward_weight(
        uintptr_t out_grad_ptr,
        uintptr_t input_ptr,
        int B, int C_in, int L,
        int C_out, int K, int L_out,
        int stride, int padding, int dilation,
        DType dtype
    ) {
        if (dtype == DType::Float32) {
            float* d_col;
            cudaMalloc(&d_col, C_in * K * B * L_out * sizeof(float));
            launch_im2col_1d<float>(
                reinterpret_cast<float*>(input_ptr), d_col,
                B, C_in, L, K, L_out,
                stride, padding, dilation);
            
            float* d_grad_weight;
            cudaMalloc(&d_grad_weight, C_out * C_in * K * sizeof(float));
            
            cublasHandle_t handle = get_cublas_handle();
            float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                C_in * K, C_out, B * L_out,
                &alpha,
                d_col, B * L_out,
                reinterpret_cast<float*>(out_grad_ptr), B * L_out, 
                &beta,
                d_grad_weight, C_in * K);
            
            cudaFree(d_col);
            return reinterpret_cast<uintptr_t>(d_grad_weight);
        }
        else if (dtype == DType::Float16) {
            half* d_col;
            cudaMalloc(&d_col, C_in * K * B * L_out * sizeof(float));
            launch_im2col_1d<half>(
                reinterpret_cast<half*>(input_ptr), d_col,
                B, C_in, L, K, L_out,
                stride, padding, dilation);
            
            half* d_grad_weight;
            cudaMalloc(&d_grad_weight, C_out * C_in * K * sizeof(half));
            
            cublasHandle_t handle = get_cublas_handle();
            half alpha = __float2half(1.0f), beta = __float2half(0.0f);
            cublasHgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                C_in * K, C_out, B * L_out,
                &alpha,
                d_col, B * L_out,
                reinterpret_cast<half*>(out_grad_ptr), B * L_out, 
                &beta,
                d_grad_weight, C_in * K);
            
            cudaFree(d_col);
            return reinterpret_cast<uintptr_t>(d_grad_weight);
        }
        else {
            throw std::runtime_error(
                "conv1d_backward_input only supports float32 and float16");
        }
    }


}


