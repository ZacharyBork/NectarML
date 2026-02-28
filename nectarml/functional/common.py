from collections.abc import Callable

import numpy as np

from nectarml import Tensor

# ABSTRACTS

def _eval_core_function(
    input: Tensor,
    func: Callable[
        [np.ndarray], 
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]],
    **kwargs
) -> Tensor:
    out_data, _backward = func(input.data, **kwargs)
    out = input._build_output_tensor(out_data, (input,))
    def _backward_hook():
        if input.requires_grad:
            input.grad += _backward(out.grad)
    out._backward = _backward_hook
    return out

