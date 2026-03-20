#include "common.h"

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

namespace nectar {

    uintptr_t pad(
        uintptr_t input_ptr,
        std::vector<int> input_shape,
        std::vector<int> pad_before,
        std::vector<int> pad_after,
        const std::string& mode,
        float constant_value,
        DType dtype
    ) {
        int B = input_shape[0];
        int C = input_shape[1];
        int n_dims = input_shape.size() - 2;

        std::vector<int> in_sizes(input_shape.begin() + 2, input_shape.end());
        std::vector<int> out_sizes(n_dims);
        for (int d = 0; d < n_dims; d++)
            out_sizes[d] = in_sizes[d] + pad_before[d] + pad_after[d];

        int total_out = B * C;
        for (int s : out_sizes) total_out *= s;

        int *d_in_sizes, *d_out_sizes, *d_pad_before;
        int memsize = n_dims * sizeof(int);
        cudaMalloc(&d_in_sizes,   n_dims * sizeof(int));
        cudaMalloc(&d_out_sizes,  n_dims * sizeof(int));
        cudaMalloc(&d_pad_before, n_dims * sizeof(int));
        cudaMemcpy(d_in_sizes,   in_sizes.data(),   memsize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out_sizes,  out_sizes.data(),  memsize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pad_before, pad_before.data(), memsize, cudaMemcpyHostToDevice);

        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, total_out * sizeof(T));

            launch_pad<T>(
                reinterpret_cast<T*>(input_ptr), d_out,
                B, C,
                d_in_sizes, d_out_sizes, d_pad_before,
                n_dims, constant_value, total_out, mode);

            cudaFree(d_in_sizes);
            cudaFree(d_out_sizes);
            cudaFree(d_pad_before);

            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

}
