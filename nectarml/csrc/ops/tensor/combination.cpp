#include "ops/common.h"
#include "common/dtype.h"
#include "common/data_structures.h"
#include "allocator_pool/allocator_pool.h"

/* KERNELS */

template<typename T>
void launch_concatenate(
    ConcatInputs inputs, 
    TensorIndex out_idx, 
    T* out, int dim, size_t n_elements
);

namespace nectar {

    uintptr_t concatenate(
        py::list ptrs,
        std::vector<std::vector<int>> shapes, 
        int dim,
        DType dtype
    ) {
        const int count = ptrs.size();
        ConcatInputs inputs;
        inputs.n_inputs = count;

        DISPATCH_DTYPE(dtype, T, {        
            for(int i = 0; i < count; i++) {
                inputs.ptrs[i] = py::cast<uintptr_t>(ptrs[i]);
                std::vector<int> shape = shapes[i];
                
                TensorIndex t_index(shape.data(), shape.size());
                inputs.indices[i] = t_index;
                inputs.offsets[i] = ((i == 0) ?
                    0 : inputs.offsets[i-1] + inputs.indices[i-1].shape[dim]);
            }

            std::vector<int> out_shape = shapes[0];
            out_shape[dim] = 0;
            for (int i = 0; i < count; i++)
                out_shape[dim] += inputs.indices[i].shape[dim];

            TensorIndex out_idx(out_shape.data(), out_shape.size());

            size_t total_elements = 0;
            for (int i = 0; i < count; i++) {
                total_elements += inputs.indices[i].n_elements;
            }
            T* d_out = static_cast<T*>(
                g_pool.alloc(total_elements * sizeof(T)));
            launch_concatenate<T>(
                inputs, out_idx, d_out,
                dim, total_elements);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

}
