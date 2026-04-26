#include "kernels/common.h"
#include "include/common/data_structures.h"
#include "include/common/functions.h"
#include "ops/policies/elementwise.h"

struct WelfordResult {
    float mean;
    float M2;
    int   count;
};

__device__ WelfordResult welford_combine(
    WelfordResult a, WelfordResult b
) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;
    
    int   count  = a.count + b.count;
    float delta  = b.mean - a.mean;
    float mean   = a.mean + delta * b.count / count;
    float M2     = a.M2 + b.M2 + delta * delta * a.count * b.count / count;
    return {mean, M2, count};
}

template<typename T>
__global__ void compute_mean_var_welford_kernel(
    const T* __restrict__ x,
    float*   __restrict__ mean_out,
    float*   __restrict__ var_out,
    int N, int C, int H, int W,
    int reduce_N, int reduce_H, int reduce_W
) {
    int nc    = blockIdx.x;
    int n     = reduce_N ? 0  : nc / C;
    int c     = reduce_N ? nc : nc % C; 
    if (c >= C) return;

    int reduce_size = (reduce_N ? N : 1) * 
                      (reduce_H ? H : 1) * 
                      (reduce_W ? W : 1);
    
    WelfordResult local = {0.0f, 0.0f, 0};
    
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        int tmp   = i;
        int w_idx = reduce_W ? tmp % W : 0; tmp /= (reduce_W ? W : 1);
        int h_idx = reduce_H ? tmp % H : 0; tmp /= (reduce_H ? H : 1);
        int n_idx = reduce_N ? tmp % N : n;
        
        int flat  = n_idx * C*H*W + c * H*W + h_idx * W + w_idx;
        float val = static_cast<float>(x[flat]);
        
        local.count++;
        float delta  = val - local.mean;
        local.mean  += delta / local.count;
        float delta2 = val - local.mean;
        local.M2    += delta * delta2;
    }
    
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        WelfordResult other;
        other.mean  = __shfl_down_sync(0xffffffff, local.mean,  offset);
        other.M2    = __shfl_down_sync(0xffffffff, local.M2,    offset);
        other.count = __shfl_down_sync(0xffffffff, local.count, offset);
        local = welford_combine(local, other);
    }
    
    __shared__ float sh_mean[32];
    __shared__ float sh_M2[32];
    __shared__ int   sh_count[32];
    
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    
    if (lane == 0) {
        sh_mean[warp]  = local.mean;
        sh_M2[warp]    = local.M2;
        sh_count[warp] = local.count;
    }
    __syncthreads();
    
    if (warp == 0) {
        int n_warps = (blockDim.x + warpSize - 1) / warpSize;
        WelfordResult block = {0.0f, 0.0f, 0};
        if (lane < n_warps) {
            block = {sh_mean[lane], sh_M2[lane], sh_count[lane]};
        }
        
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            WelfordResult other;
            other.mean  = __shfl_down_sync(0xffffffff, block.mean,  offset);
            other.M2    = __shfl_down_sync(0xffffffff, block.M2,    offset);
            other.count = __shfl_down_sync(0xffffffff, block.count, offset);
            block = welford_combine(block, other);
        }
        
        if (threadIdx.x == 0) {
            mean_out[nc] = block.mean;
            var_out[nc]  = block.count > 1 ? block.M2 / block.count : 0.0f;
        }
    }
}

template<typename T>
void launch_compute_mean_var_welford(
    const T* x, float* mean, float* var,
    int N, int C, int H, int W,
    int reduce_N, int reduce_H, int reduce_W
) {
    int n_blocks = reduce_N ? C : N * C;
    compute_mean_var_welford_kernel<T><<<n_blocks, 256>>>(
        x, mean, var, N, C, H, W,
        reduce_N, reduce_H, reduce_W);
}

template void launch_compute_mean_var_welford<float>(
    const float*, float*, float*, int, int, int, int, int, int, int);
template void launch_compute_mean_var_welford<half>(
    const half*, float*, float*, int, int, int, int, int, int, int);
template void launch_compute_mean_var_welford<uint8_t>(
    const uint8_t*, float*, float*, int, int, int, int, int, int, int);
template void launch_compute_mean_var_welford<int32_t>(
    const int32_t*, float*, float*, int, int, int, int, int, int, int);

template<typename T>
__global__ void normalize_kernel(
    const T*     __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    T*           __restrict__ out,
    int N, int C, int H, int W,
    float eps,
    int reduce_N
) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    
    int n  = idx / (C * H * W);
    int c  = (idx / (H * W)) % C;
    
    int stat_idx = reduce_N ? c : n * C + c;
    
    float x_val  = static_cast<float>(x[idx]);
    float x_norm = (x_val - mean[stat_idx]) / sqrtf(var[stat_idx] + eps);
    
    if (gamma) x_norm *= gamma[c];
    if (beta)  x_norm += beta[c];
    
    out[idx] = static_cast<T>(x_norm);
}

template<typename T>
void launch_normalize(
    const T* x, const float* mean, const float* var,
    const float* gamma, const float* beta,
    T* out,
    int N, int C, int H, int W,
    float eps, int reduce_N
) {
    size_t total = (size_t)N * C * H * W;
    int threads  = 256;
    int blocks   = (total + threads - 1) / threads;
    normalize_kernel<T><<<blocks, threads>>>(
        x, mean, var, gamma, beta, out,
        N, C, H, W, eps, reduce_N);
}

template void launch_normalize<float>(
    const float*, const float*, const float*, const float*, 
    const float*, float*, int, int, int, int, float, int);
template void launch_normalize<half>(
    const half*, const float*, const float*, const float*, 
    const float*, half*, int, int, int, int, float, int);
template void launch_normalize<uint8_t>(
    const uint8_t*, const float*, const float*, const float*,
     const float*, uint8_t*, int, int, int, int, float, int);
template void launch_normalize<int32_t>(
    const int32_t*, const float*, const float*, const float*, 
    const float*, int32_t*, int, int, int, int, float, int);

template<typename T>
__global__ void batch_norm_backward_kernel(
    const T*     __restrict__ grad_out,
    const T*     __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ gamma,
    float*       __restrict__ dx,
    float*       __restrict__ dgamma,
    float*       __restrict__ dbeta,
    int N, int C, int H, int W,
    int reduce_N, int reduce_H, int reduce_W,
    float eps
) {
    __shared__ float sh_dgamma[32], sh_dbeta[32],
                     sh_dxhat[32],  sh_xhat_dxhat[32];
    __shared__ float sh_sum_dxhat, sh_sum_xhat_dxhat;

    int nc = blockIdx.x;
    int n  = reduce_N ? 0   : nc / C;
    int c  = reduce_N ? nc  : nc % C;
    int stat_idx = reduce_N ? c : n * C + c;
    if (c >= C) return;
    
    int reduce_size = (reduce_N ? N : 1) * 
                      (reduce_H ? H : 1) * 
                      (reduce_W ? W : 1);
    
    float mu      = mean[stat_idx];
    float sig_inv = 1.0f / sqrtf(var[stat_idx] + eps);
    float g       = gamma ? gamma[c] : 1.0f;
    
    float sum_dgamma     = 0.0f;
    float sum_dbeta      = 0.0f;
    float sum_dxhat      = 0.0f;
    float sum_xhat_dxhat = 0.0f;
    
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        int tmp   = i;
        int w_idx = reduce_W ? tmp % W : 0; tmp /= (reduce_W ? W : 1);
        int h_idx = reduce_H ? tmp % H : 0; tmp /= (reduce_H ? H : 1);
        int n_idx = reduce_N ? tmp % N : n;
        
        int flat   = n_idx * C*H*W + c * H*W + h_idx * W + w_idx;
        float go   = static_cast<float>(grad_out[flat]);
        float xhat = (static_cast<float>(x[flat]) - mu) * sig_inv;
        
        sum_dgamma      += go * xhat;
        sum_dbeta       += go;
        sum_dxhat       += go * g;
        sum_xhat_dxhat  += xhat * go * g;
    }
    
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        sum_dgamma     += __shfl_down_sync(0xffffffff, sum_dgamma,     offset);
        sum_dbeta      += __shfl_down_sync(0xffffffff, sum_dbeta,      offset);
        sum_dxhat      += __shfl_down_sync(0xffffffff, sum_dxhat,      offset);
        sum_xhat_dxhat += __shfl_down_sync(0xffffffff, sum_xhat_dxhat, offset);
    }
    
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) {
        sh_dgamma[warp]     = sum_dgamma;
        sh_dbeta[warp]      = sum_dbeta;
        sh_dxhat[warp]      = sum_dxhat;
        sh_xhat_dxhat[warp] = sum_xhat_dxhat;
    }
    __syncthreads();
    
    if (warp == 0) {
        sum_dgamma     = lane < blockDim.x/warpSize ? sh_dgamma[lane]     : 0;
        sum_dbeta      = lane < blockDim.x/warpSize ? sh_dbeta[lane]      : 0;
        sum_dxhat      = lane < blockDim.x/warpSize ? sh_dxhat[lane]      : 0;
        sum_xhat_dxhat = lane < blockDim.x/warpSize ? sh_xhat_dxhat[lane] : 0;
        
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            sum_dgamma     += __shfl_down_sync(0xffffffff, sum_dgamma,     offset);
            sum_dbeta      += __shfl_down_sync(0xffffffff, sum_dbeta,      offset);
            sum_dxhat      += __shfl_down_sync(0xffffffff, sum_dxhat,      offset);
            sum_xhat_dxhat += __shfl_down_sync(0xffffffff, sum_xhat_dxhat, offset);
        }
        
        if (threadIdx.x == 0) {
            if (dgamma) dgamma[c] = sum_dgamma;
            if (dbeta)  dbeta[c]  = sum_dbeta;
            sh_sum_dxhat      = sum_dxhat;
            sh_sum_xhat_dxhat = sum_xhat_dxhat;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        int tmp   = i;
        int w_idx = reduce_W ? tmp % W : 0; tmp /= (reduce_W ? W : 1);
        int h_idx = reduce_H ? tmp % H : 0; tmp /= (reduce_H ? H : 1);
        int n_idx = reduce_N ? tmp % N : n;
        
        int flat   = n_idx * C*H*W + c * H*W + h_idx * W + w_idx;
        float go   = static_cast<float>(grad_out[flat]);
        float xhat = (static_cast<float>(x[flat]) - mu) * sig_inv;
        
        float dx_val = (1.0f / reduce_size) * sig_inv * (
            reduce_size * go * g
            - sh_sum_dxhat
            - xhat * sh_sum_xhat_dxhat
        );

        atomic_add(&dx[flat], dx_val);
    }
}

template<typename T>
void launch_batch_norm_backward(
    const T* grad_out, const T* x,
    const float* mean, const float* var, const float* gamma,
    float* dx, float* dgamma, float* dbeta,
    int N, int C, int H, int W,
    int reduce_N, int reduce_H, int reduce_W,
    float eps
) {
    int n_blocks = reduce_N ? C : N * C;
    batch_norm_backward_kernel<T><<<n_blocks, 256>>>(
        grad_out, x, mean, var, gamma, dx, dgamma, dbeta,
        N, C, H, W, reduce_N, reduce_H, reduce_W, eps
    );
}
   
template void launch_batch_norm_backward<float>(
    const float*, const float*, 
    const float*, const float*, const float*, 
    float*, float*, float*,
    int, int, int, int, int, int, int, float);
template void launch_batch_norm_backward<half>(
    const half*, const half*, 
    const float*, const float*, const float*, 
    float*, float*, float*,
    int, int, int, int, int, int, int, float);
template void launch_batch_norm_backward<uint8_t>(
    const uint8_t*, const uint8_t*, 
    const float*, const float*, const float*, 
    float*, float*, float*,
    int, int, int, int, int, int, int, float);
template void launch_batch_norm_backward<int32_t>(
    const int32_t*, const int32_t*, 
    const float*, const float*, const float*, 
    float*, float*, float*,
    int, int, int, int, int, int, int, float);


