#include "common/dtype.h"
#include "allocator_pool/allocator_pool.h"

/* KERNELS */

// AVERAGE POOL

template<typename T>
void launch_avg_pool1d_forward(
    T* input, T* output,
    int B, int C, int L, 
    int L_out, int K, int S, int P, 
    bool count_include_pad
);

template<typename T>
void launch_avg_pool1d_backward(
    T* input, T* output,
    int B, int C, int L, int L_out,
    int K, int S, int P,
    bool count_include_pad
);

template<typename T>
void launch_avg_pool2d_forward(
    T* input, T* output,
    int B, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW, 
    int SH, int SW,
    int PH, int PW, 
    bool count_include_pad
);

template<typename T>
void launch_avg_pool2d_backward(
    T* input, T* output,
    int B, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW, 
    int SH, int SW,
    int PH, int PW, 
    bool count_include_pad
);

template<typename T>
void launch_avg_pool3d_forward(
    T* input, T* output,
    int B, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int KD, int KH, int KW, 
    int SD, int SH, int SW,
    int PD, int PH, int PW, 
    bool count_include_pad
);

template<typename T>
void launch_avg_pool3d_backward(
    T* input, T* output, 
    int B, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int KD, int KH, int KW, 
    int SD, int SH, int SW,
    int PD, int PH, int PW, 
    bool count_include_pad
);

// MAX POOL 

template<typename T>
void launch_max_pool1d_forward(
    T* input, T* output, int32_t* indices,
    int B, int C, int L, int L_out, 
    int K, int S, int P, int D
);

template<typename T>
void launch_max_pool1d_backward(
    T* out_grad, int32_t* indices, T* grad_input,
    int B, int C, int L, int L_out
);

template<typename T>
void launch_max_pool2d_forward(
    T* input, T* output, int32_t* indices,
    int B, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW, int SH, int SW,
    int PH, int PW, int D
);

template<typename T>
void launch_max_pool2d_backward(
    T* out_grad, int32_t* indices, T* grad_input,
    int B, int C, int H, int W, int H_out, int W_out
);

template<typename T>
void launch_max_pool3d_forward(
    T* input, T* output, int32_t* indices,
    int B, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int KD, int KH, int KW, 
    int SD, int SH, int SW,
    int PD, int PH, int PW, 
    int Dil
);

template<typename T>
void launch_max_pool3d_backward(
    T* out_grad, int32_t* indices, T* grad_input,
    int B, int C, int D, int H, int W,
    int D_out, int H_out, int W_out
);

/* NAMESPACE OPS*/

namespace nectar {

    /* AVERAGE POOL */

    uintptr_t avg_pool1d_forward(
        uintptr_t input_ptr,
        int B, int C, int L, int L_out,
        int K, int S, int P,
        bool count_include_pad,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_output = static_cast<T*>(g_pool.alloc(B * C * L_out * sizeof(T)));
            launch_avg_pool1d_forward<T>(
                reinterpret_cast<T*>(input_ptr), d_output,
                B, C, L, L_out, K, S, P, count_include_pad);
            return reinterpret_cast<uintptr_t>(d_output);
        });
    }

    uintptr_t avg_pool1d_backward(
        uintptr_t out_grad_ptr,
        int B, int C, int L, int L_out,
        int K, int S, int P,
        bool count_include_pad,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input = static_cast<T*>(g_pool.alloc(B * C * L * sizeof(T)));
            cudaMemset(d_grad_input, 0, B * C * L * sizeof(T));
            launch_avg_pool1d_backward<T>(
                reinterpret_cast<T*>(out_grad_ptr), d_grad_input,
                B, C, L, L_out, K, S, P, count_include_pad);
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }

    uintptr_t avg_pool2d_forward(
        uintptr_t input_ptr,
        int B, int C, int H, int W,
        int H_out, int W_out,
        int KH, int KW,
        int SH, int SW,
        int PH, int PW,
        bool count_include_pad,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_output = static_cast<T*>(g_pool.alloc( B * C * H_out * W_out * sizeof(T)));
            launch_avg_pool2d_forward<T>(
                reinterpret_cast<T*>(input_ptr), d_output,
                B, C, H, W, H_out, W_out,
                KH, KW, SH, SW, PH, PW, count_include_pad);
            return reinterpret_cast<uintptr_t>(d_output);
        });
    }

    uintptr_t avg_pool2d_backward(
        uintptr_t out_grad_ptr,
        int B, int C, int H, int W,
        int H_out, int W_out,
        int KH, int KW,
        int SH, int SW,
        int PH, int PW,
        bool count_include_pad,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input = static_cast<T*>(g_pool.alloc(B * C * H * W * sizeof(T)));
            cudaMemset(d_grad_input, 0, B * C * H * W * sizeof(T));
            launch_avg_pool2d_backward<T>(
                reinterpret_cast<T*>(out_grad_ptr), d_grad_input,
                B, C, H, W, H_out, W_out,
                KH, KW, SH, SW, PH, PW, count_include_pad);
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }

    uintptr_t avg_pool3d_forward(
        uintptr_t input_ptr,
        int B, int C, int D, int H, int W,
        int D_out, int H_out, int W_out,
        int KD, int KH, int KW,
        int SD, int SH, int SW,
        int PD, int PH, int PW,
        bool count_include_pad,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_output = static_cast<T*>(g_pool.alloc(B * C * D_out * H_out * W_out * sizeof(T)));
            launch_avg_pool3d_forward<T>(
                reinterpret_cast<T*>(input_ptr), d_output,
                B, C, D, H, W, D_out, H_out, W_out,
                KD, KH, KW, SD, SH, SW, PD, PH, PW, count_include_pad);
            return reinterpret_cast<uintptr_t>(d_output);
        });
    }

    uintptr_t avg_pool3d_backward(
        uintptr_t out_grad_ptr,
        int B, int C, int D, int H, int W,
        int D_out, int H_out, int W_out,
        int KD, int KH, int KW,
        int SD, int SH, int SW,
        int PD, int PH, int PW,
        bool count_include_pad,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input = static_cast<T*>(g_pool.alloc(B * C * D * H * W * sizeof(T)));
            cudaMemset(d_grad_input, 0, B * C * D * H * W * sizeof(T));
            launch_avg_pool3d_backward<T>(
                reinterpret_cast<T*>(out_grad_ptr), d_grad_input,
                B, C, D, H, W, D_out, H_out, W_out,
                KD, KH, KW, SD, SH, SW, PD, PH, PW, count_include_pad);
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }

    /* MAX POOl */

    std::pair<uintptr_t, uintptr_t> max_pool1d_forward(
        uintptr_t input_ptr,
        int B, int C, int L, int L_out,
        int K, int S, int P, int D,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T*       d_output  = static_cast<T*>(g_pool.alloc(B * C * L_out * sizeof(T)));
            int32_t* d_indices = static_cast<int32_t*>(g_pool.alloc(B * C * L_out * sizeof(int32_t)));
            launch_max_pool1d_forward<T>(
                reinterpret_cast<T*>(input_ptr), d_output, d_indices,
                B, C, L, L_out, K, S, P, D);
            return {reinterpret_cast<uintptr_t>(d_output),
                    reinterpret_cast<uintptr_t>(d_indices)};
        });
    }

    uintptr_t max_pool1d_backward(
        uintptr_t out_grad_ptr,
        uintptr_t indices_ptr,
        int B, int C, int L, int L_out,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input = static_cast<T*>(g_pool.alloc(B * C * L * sizeof(T)));
            cudaMemset(d_grad_input, 0, B * C * L * sizeof(T));
            launch_max_pool1d_backward<T>(
                reinterpret_cast<T*>(out_grad_ptr),
                reinterpret_cast<int32_t*>(indices_ptr),
                d_grad_input,
                B, C, L, L_out);
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }

    std::pair<uintptr_t, uintptr_t> max_pool2d_forward(
        uintptr_t input_ptr,
        int B, int C, int H, int W,
        int H_out, int W_out,
        int KH, int KW,
        int SH, int SW,
        int PH, int PW, int D,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T*       d_output = static_cast<T*>(g_pool.alloc(B * C * H_out * W_out * sizeof(T)));
            int32_t* d_indices = static_cast<int32_t*>(g_pool.alloc(B * C * H_out * W_out * sizeof(int32_t)));
            launch_max_pool2d_forward<T>(
                reinterpret_cast<T*>(input_ptr), d_output, d_indices,
                B, C, H, W, H_out, W_out,
                KH, KW, SH, SW, PH, PW, D);
            return {reinterpret_cast<uintptr_t>(d_output),
                    reinterpret_cast<uintptr_t>(d_indices)};
        });
    }

    uintptr_t max_pool2d_backward(
        uintptr_t out_grad_ptr,
        uintptr_t indices_ptr,
        int B, int C, int H, int W,
        int H_out, int W_out,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input = static_cast<T*>(g_pool.alloc(B * C * H * W * sizeof(T)));
            cudaMemset(d_grad_input, 0, B * C * H * W * sizeof(T));
            launch_max_pool2d_backward<T>(
                reinterpret_cast<T*>(out_grad_ptr),
                reinterpret_cast<int32_t*>(indices_ptr),
                d_grad_input,
                B, C, H, W, H_out, W_out);
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }

    std::pair<uintptr_t, uintptr_t> max_pool3d_forward(
        uintptr_t input_ptr,
        int B, int C, int D, int H, int W,
        int D_out, int H_out, int W_out,
        int KD, int KH, int KW,
        int SD, int SH, int SW,
        int PD, int PH, int PW, int Dil,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T*       d_output = static_cast<T*>(g_pool.alloc(B * C * D_out * H_out * W_out * sizeof(T)));
            int32_t* d_indices = static_cast<int32_t*>(g_pool.alloc(B * C * D_out * H_out * W_out * sizeof(int32_t)));
            launch_max_pool3d_forward<T>(
                reinterpret_cast<T*>(input_ptr), d_output, d_indices,
                B, C, D, H, W, D_out, H_out, W_out,
                KD, KH, KW, SD, SH, SW, PD, PH, PW, Dil);
            return {reinterpret_cast<uintptr_t>(d_output),
                    reinterpret_cast<uintptr_t>(d_indices)};
        });
    }

    uintptr_t max_pool3d_backward(
        uintptr_t out_grad_ptr,
        uintptr_t indices_ptr,
        int B, int C, int D, int H, int W,
        int D_out, int H_out, int W_out,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_grad_input = static_cast<T*>(g_pool.alloc(B * C * D * H * W * sizeof(T)));
            cudaMemset(d_grad_input, 0, B * C * D * H * W * sizeof(T));
            launch_max_pool3d_backward<T>(
                reinterpret_cast<T*>(out_grad_ptr),
                reinterpret_cast<int32_t*>(indices_ptr),
                d_grad_input,
                B, C, D, H, W, D_out, H_out, W_out);
            return reinterpret_cast<uintptr_t>(d_grad_input);
        });
    }

}
