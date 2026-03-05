from collections.abc import Callable

import numpy as np

from nectarml.tensor import Tensor

# ABSTRACTS

def _eval_core_function(
    input: Tensor,
    func: Callable[
        [np.ndarray], 
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]],
    **kwargs
) -> Tensor:
    out_data, _backward = func(input.data, **kwargs)
    out = Tensor(out_data, out_data.shape, input.dtype, input.device,
        input.requires_grad, _children=(input,))
    def _backward_hook():
        if input.requires_grad:
            input.grad += _backward(out.grad)
    out._backward = _backward_hook
    return out

