#include "common.h"
#include "allocator_pool/allocator_pool.h"

/* KERNELS */

template<typename T>
void sort_tensor(
    T* d_values, int32_t* d_indices,
    int dim_size, int outer, int inner,
    bool descending
);

/* NAMESPACE OPS */

namespace nectar {

    std::pair<uintptr_t, uintptr_t> sort(
        uintptr_t input_ptr,
        int total,
        int dim_size,
        int outer,
        int inner,
        bool descending,
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            T* d_values = static_cast<T*>(g_pool.alloc(total * sizeof(T)));
            int32_t* d_indices = static_cast<int32_t*>(g_pool.alloc(total * sizeof(int32_t)));
            cudaMemcpy(d_values, reinterpret_cast<T*>(input_ptr), 
                total * sizeof(T), cudaMemcpyDeviceToDevice);

            sort_tensor<T>(
                d_values, d_indices,
                dim_size, outer, inner, descending);

            return {
                reinterpret_cast<uintptr_t>(d_values),
                reinterpret_cast<uintptr_t>(d_indices)
            }; 
        });
    }

}
