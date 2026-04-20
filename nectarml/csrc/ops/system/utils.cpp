#include "ops/common.h"
#include "common/dtype.h"
#include "ops/policies/reductions.h"
#include "allocator_pool/allocator_pool.h"

/* KERNELS */

template<typename T, template<typename> class Op>
void launch_reduce(
    T* in_data, T* out_data, size_t n_elements, 
    float initial = std::numeric_limits<float>::quiet_NaN()
);

template<typename SrcT, typename DstT>
void launch_cast_kernel(SrcT* src, DstT* dst, size_t n_elements);

namespace nectar {

    float compute_tensor_min(
        uintptr_t device_ptr, 
        size_t n_elements, 
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            float result;
            T* d_out = static_cast<T*>(g_pool.alloc(sizeof(T)));
            launch_reduce<T, MinOp>(
                reinterpret_cast<T*>(device_ptr), d_out, n_elements);
            cudaMemcpy(&result, d_out, sizeof(T), cudaMemcpyDeviceToHost);
            g_pool.free(d_out, sizeof(T));
            return static_cast<double>(result);
        });
    }

    float compute_tensor_max(
        uintptr_t device_ptr, 
        size_t n_elements, 
        DType dtype
    ) {
        DISPATCH_DTYPE(dtype, T, {
            float result;
            T* d_out = static_cast<T*>(g_pool.alloc(sizeof(T)));
            launch_reduce<T, MaxOp>(
                reinterpret_cast<T*>(device_ptr), d_out, n_elements);
            cudaMemcpy(&result, d_out, sizeof(T), cudaMemcpyDeviceToHost);
            g_pool.free(d_out, sizeof(T));
            return static_cast<double>(result);
        });
    }

    std::vector<float> compute_tensor_range(
        uintptr_t device_ptr,
        size_t n_elements,
        DType dtype
    ) {
        std::vector<float> output;
        output.push_back(compute_tensor_min(device_ptr, n_elements, dtype));
        output.push_back(compute_tensor_max(device_ptr, n_elements, dtype));
        return output;
    }

}

