#include "include/common/dtype.h"
#include "ops/common.h"
#include "ops/system/device/device.h"
#include "pool/allocator_pool.h"

template<typename T>
void launch_compute_mean_var_welford(
    const T* x, float* mean_out, float* var_out,
    int N, int C, int H, int W,
    int reduce_N, int reduce_H, int reduce_W
);

template<typename T>
void launch_normalize(
    const T* x, const float* mean, const float* var,
    const float* gamma, const float* beta, T* out,
    int N, int C, int H, int W, float eps, int reduce_N
);

template<typename T>
void launch_batch_norm_backward(
    const T* grad_out,
    const T* x,
    const float* mean,
    const float* var,
    const float* gamma,
    float* dx,
    float* dgamma,
    float* dbeta,
    int N, int C, int H, int W,
    int reduce_N, int reduce_H, int reduce_W,
    float eps
);

namespace nectar {

    uintptr_t batch_norm_forward(
        uintptr_t x_ptr,
        uintptr_t gamma_ptr,
        uintptr_t beta_ptr,
        uintptr_t mean_ptr,
        uintptr_t var_ptr,
        int N, int C, int H, int W,
        bool reduce_N, bool reduce_H, bool reduce_W,
        float eps
    ) {
        float* x     = reinterpret_cast<float*>(x_ptr);
        float* gamma = gamma_ptr ? reinterpret_cast<float*>(gamma_ptr) : nullptr;
        float* beta  = beta_ptr  ? reinterpret_cast<float*>(beta_ptr)  : nullptr;

        float* mean  = reinterpret_cast<float*>(mean_ptr);
        float* var   = reinterpret_cast<float*>(var_ptr);

        launch_compute_mean_var_welford<float>(
            x, mean, var, N, C, H, W, reduce_N, reduce_H, reduce_W);

        size_t total = (size_t)N * C * H * W;
        float* out   = reinterpret_cast<float*>(g_pool.alloc(total * sizeof(float)));
        launch_normalize<float>(x, mean, var, gamma, beta, out, N, C, H, W, eps, reduce_N);

        return reinterpret_cast<uintptr_t>(out);
    }

    void batch_norm_backward(
        uintptr_t grad_out_ptr,
        uintptr_t x_ptr,
        uintptr_t mean_ptr,
        uintptr_t var_ptr,
        uintptr_t gamma_ptr,
        uintptr_t dx_ptr,
        uintptr_t dgamma_ptr,
        uintptr_t dbeta_ptr,
        int N, int C, int H, int W,
        bool reduce_N, bool reduce_H, bool reduce_W,
        float eps
    ) {
        if (dx_ptr) {
            cudaMemsetAsync(
                reinterpret_cast<void*>(dx_ptr), 0,
                static_cast<size_t>(N) * C * H * W * sizeof(float)
            );
        }

        if (dgamma_ptr) {
            cudaMemsetAsync(
                reinterpret_cast<void*>(dgamma_ptr), 0,
                static_cast<size_t>(C) * sizeof(float)
            );
        }

        if (dbeta_ptr) {
            cudaMemsetAsync(
                reinterpret_cast<void*>(dbeta_ptr), 0,
                static_cast<size_t>(C) * sizeof(float)
            );
        }

        launch_batch_norm_backward<float>(
            reinterpret_cast<float*>(grad_out_ptr),
            reinterpret_cast<float*>(x_ptr),
            reinterpret_cast<float*>(mean_ptr),
            reinterpret_cast<float*>(var_ptr),
            gamma_ptr ? reinterpret_cast<float*>(gamma_ptr) : nullptr,
            reinterpret_cast<float*>(dx_ptr),
            reinterpret_cast<float*>(dgamma_ptr),
            reinterpret_cast<float*>(dbeta_ptr),
            N, C, H, W, reduce_N, reduce_H, reduce_W, eps
        );
    }
}
