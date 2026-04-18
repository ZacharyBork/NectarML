#include "common.h"
#include "allocator_pool/allocator_pool.h"

namespace py = pybind11;

template<typename T>
void launch_rotate(
    T* input, T* output,
    int B, int C, int H, int W,
    float angle_degrees, float fill_value
);

namespace nectar {
    
    uintptr_t rotate(
        uintptr_t in_ptr, std::vector<int> shape, 
        float angle, float fill_value,
        DType dtype
    ) {
        size_t memsize = 1;
        for (int i : shape) { memsize *= i; }

        DISPATCH_DTYPE(dtype, T, {
            T* d_out = static_cast<T*>(g_pool.alloc(memsize * sizeof(T)));
            launch_rotate<T>(
                reinterpret_cast<T*>(in_ptr), d_out, 
                shape[0], shape[1], shape[2], shape[3],
                angle, fill_value);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

}

