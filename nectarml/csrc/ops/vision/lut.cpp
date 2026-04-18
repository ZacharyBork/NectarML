#include "common.h"
#include "allocator_pool/allocator_pool.h"

/* KERNELS */

template<typename T>
void launch_apply_lut(
    T* input, T* output,
    float* lut,
    int B, int H, int W,
    int lut_size
);

/* NAMESPACE OPS */

namespace nectar {

    uintptr_t apply_lut(
        uintptr_t in_ptr,
        uintptr_t lut_ptr,
        int B, int H, int W,
        int lut_size,
        DType dtype
    ) {
        float* d_lut = reinterpret_cast<float*>(lut_ptr);

        DISPATCH_DTYPE(dtype, T, {
            T* d_input = reinterpret_cast<T*>(in_ptr);
            T* d_output = static_cast<T*>(g_pool.alloc(B * 3 * H * W * sizeof(T)));

            launch_apply_lut<T>(
                d_input, d_output, d_lut,
                B, H, W, lut_size);

            return reinterpret_cast<uintptr_t>(d_output);
        });
    }

}

