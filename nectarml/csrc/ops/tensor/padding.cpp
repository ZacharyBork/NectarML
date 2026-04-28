#include "include/common/dtype.h"
#include "pool/allocator_pool.h"

/* KERNELS */

template<typename T>
void launch_pad(
    T* input, T* output,
    int B, int C,
    int* in_sizes,
    int* out_sizes,
    int* pad_before,
    int n_dims,
    float constant_value,
    int total_out,
    const std::string& mode
);

void launch_pad_backward(
    float* grad_out,
    float* grad_in,
    int    B, 
    int    C,
    int*   in_sizes,
    int*   out_sizes,
    int*   pad_before,
    int    n_dims,
    int    total_out,
    const std::string& mode
);

namespace nectar {

    uintptr_t pad(
        uintptr_t          input_ptr,
        std::vector<int>   input_shape,
        std::vector<int>   pad_before,
        std::vector<int>   pad_after,
        const std::string& mode,
        float              constant_value,
        DType              dtype
    ) {
        int B      = input_shape[0];
        int C      = input_shape[1];
        int n_dims = input_shape.size() - 2;

        std::vector<int> in_sizes(input_shape.begin() + 2, input_shape.end());
        std::vector<int> out_sizes(n_dims);
        for (int d = 0; d < n_dims; d++)
            out_sizes[d] = in_sizes[d] + pad_before[d] + pad_after[d];

        int total_out = B * C;
        for (int s : out_sizes) total_out *= s;

        int memsize       = n_dims * sizeof(int);
        int *d_in_sizes   = static_cast<int*>(g_pool.alloc(memsize));
        int *d_out_sizes  = static_cast<int*>(g_pool.alloc(memsize));
        int *d_pad_before = static_cast<int*>(g_pool.alloc(memsize));
        cudaMemcpy(d_in_sizes,   in_sizes.data(),   memsize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out_sizes,  out_sizes.data(),  memsize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pad_before, pad_before.data(), memsize, cudaMemcpyHostToDevice);

        DISPATCH_DTYPE(dtype, T, {
            T* d_out = static_cast<T*>(g_pool.alloc(total_out * sizeof(T)));

            launch_pad<T>(
                reinterpret_cast<T*>(input_ptr), d_out,
                B, C,
                d_in_sizes, d_out_sizes, d_pad_before,
                n_dims, constant_value, total_out, mode);

            g_pool.free(d_in_sizes,   memsize);
            g_pool.free(d_out_sizes,  memsize);
            g_pool.free(d_pad_before, memsize);

            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

    void pad_backward(
        uintptr_t          grad_out_ptr,
        uintptr_t          grad_in_ptr,
        std::vector<int>   input_shape,
        std::vector<int>   pad_before,
        std::vector<int>   pad_after,
        const std::string& mode
    ) {
        int B      = input_shape[0];
        int C      = input_shape[1];
        int n_dims = input_shape.size() - 2;

        std::vector<int> in_sizes(input_shape.begin() + 2, input_shape.end());
        std::vector<int> out_sizes(n_dims);
        for (int d = 0; d < n_dims; d++)
            out_sizes[d] = in_sizes[d] + pad_before[d] + pad_after[d];

        int total_out = B * C;
        for (int s : out_sizes) total_out *= s;

        int memsize       = n_dims * sizeof(int);
        int* d_in_sizes   = static_cast<int*>(g_pool.alloc(memsize));
        int* d_out_sizes  = static_cast<int*>(g_pool.alloc(memsize));
        int* d_pad_before = static_cast<int*>(g_pool.alloc(memsize));
        cudaMemcpy(d_in_sizes,    in_sizes.data(),   memsize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out_sizes,   out_sizes.data(),  memsize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pad_before,  pad_before.data(), memsize, cudaMemcpyHostToDevice);

        launch_pad_backward(
            reinterpret_cast<float*>(grad_out_ptr),
            reinterpret_cast<float*>(grad_in_ptr),
            B, C, d_in_sizes, d_out_sizes, d_pad_before,
            n_dims, total_out, mode
        );

        g_pool.free(d_in_sizes,   memsize);
        g_pool.free(d_out_sizes,  memsize);
        g_pool.free(d_pad_before, memsize);
    }

}
