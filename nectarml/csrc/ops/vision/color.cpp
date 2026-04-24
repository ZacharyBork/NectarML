#include "ops/common.h"
#include "include/common/dtype.h"
#include "pool/allocator_pool.h"

template<typename T>
void launch_hsv_adjust(
    T* d_in, int B, int C, int H, int W,
    float hue_shift, float saturation, float value
);

namespace nectar {
    
    uintptr_t hsv_adjust(
        uintptr_t in_ptr, std::vector<int> shape, 
        float hue_shift, float saturation, float value,
        DType dtype
    ) {
        size_t memsize = 1;
        for (int i : shape) { memsize *= i; }

        DISPATCH_DTYPE(dtype, T, {
            T* d_out = static_cast<T*>(g_pool.alloc(memsize * sizeof(T)));
            cudaMemcpy(d_out, reinterpret_cast<T*>(in_ptr), 
                   memsize * sizeof(T), cudaMemcpyDeviceToDevice);
            launch_hsv_adjust<T>(
                d_out, shape[0], shape[1], shape[2], shape[3],
                hue_shift, saturation, value);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

}


