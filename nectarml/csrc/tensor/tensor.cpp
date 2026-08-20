#include "tensor/tensor.h"

Tensor::Tensor(
    uintptr_t        data_ptr, 
    std::vector<int> shape,
    Device           device,
    DType            dtype,
    bool             requires_grad,
    std::vector<uintptr_t> _children
) : data_ptr(data_ptr), 
    shape(shape), 
    device(device), 
    dtype(dtype),
    requires_grad(requires_grad), 
    _children(_children) 
{ }

