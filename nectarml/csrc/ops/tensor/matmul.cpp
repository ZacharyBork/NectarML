#include "common.h"
#include "ops/device.h"
#include <cublas_v2.h>

namespace nectar {

    uintptr_t matmul(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape, std::vector<int> b_shape,
        DType dtype
    ) {
        int M = a_shape[a_shape.size() - 2];
        int K = a_shape[a_shape.size() - 1];
        int N = b_shape[b_shape.size() - 1];
        int batch = 1;
        for (int i = 0; i < (int)a_shape.size() - 2; i++)
            batch *= a_shape[i];

        cublasHandle_t handle = get_cublas_handle();

        if (dtype == DType::Float32) {
            float* d_out;
            cudaMalloc(&d_out, batch * M * N * sizeof(float));
            float alpha = 1.0f, beta = 0.0f;
            for (int b = 0; b < batch; b++) {
                float* A = reinterpret_cast<float*>(a_ptr) + b * M * K;
                float* B = reinterpret_cast<float*>(b_ptr) + b * K * N;
                float* C = d_out + b * M * N;
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, B, N, A, K, &beta, C, N);
            }
            return reinterpret_cast<uintptr_t>(d_out);
        } 
        else if (dtype == DType::Float16) {
            half* d_out;
            cudaMalloc(&d_out, batch * M * N * sizeof(half));
            half alpha = __float2half(1.0f), beta = __float2half(0.0f);
            for (int b = 0; b < batch; b++) {
                half* A = reinterpret_cast<half*>(a_ptr) + b * M * K;
                half* B = reinterpret_cast<half*>(b_ptr) + b * K * N;
                half* C = d_out + b * M * N;
                cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, B, N, A, K, &beta, C, N);
            }
            return reinterpret_cast<uintptr_t>(d_out);

        } 
        else { throw std::runtime_error("matmul only supports float32 and float16"); }
    }

}

