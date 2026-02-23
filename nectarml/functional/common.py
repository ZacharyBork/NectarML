from collections.abc import Callable

import numpy as np

from nectarml import Tensor

# ABSTRACTS

def _manipulate_shape(input: Tensor, new_data: np.ndarray) -> Tensor:
    old_shape = input.data.shape
    out = input._build_output_tensor(new_data, (input,))
    def _backward():
        if input.requires_grad:
            input.grad += out.grad.reshape(old_shape)
    out._backward = _backward
    return out

def _wrapper_base(
    input: Tensor,
    func: Callable[[np.ndarray], np.ndarray], 
    grad_func: Callable[[np.ndarray], np.ndarray], 
    children: tuple[Tensor, ...] | None = None,
    reduce_dim: int | None = None
) -> Tensor:
    if children is None: children = (input,)
    out = input._build_output_tensor(func(input.data), children)
    def _backward():
        if input.requires_grad:
            out_grad = out.grad
            if reduce_dim is not None and out_grad.ndim < input.data.ndim:
                out_grad = np.expand_dims(out_grad, axis=reduce_dim)
            out_grad = np.broadcast_to(out_grad, input.data.shape)
            input.grad += grad_func(input.data) * out_grad
    out._backward = _backward
    return out

