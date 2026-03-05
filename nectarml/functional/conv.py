from typing import Literal

from nectarml.tensor import Tensor
from nectarml._core import conv

def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int | Literal['valid', 'same'] = 0,
    dilation: int = 1,
    groups: int = 1
) -> Tensor:
    out_data, _backward = conv.conv1d(
        input.data, weight.data, bias.data if bias is not None else None, 
        stride, padding, dilation, groups)
    out = Tensor(out_data, out_data.shape, input.dtype, input.device,
        input.requires_grad, _children=(input,))
    def _backward_hook():
        grad_input, grad_weight, grad_bias = _backward(out.grad)
        if input.requires_grad: input.grad += grad_input
        if weight.requires_grad: weight.grad += grad_weight
        if bias is not None and bias.requires_grad: bias.grad += grad_bias
    out._backward = _backward_hook
    return out
                    
def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int | Literal['valid', 'same'] = 0,
    dilation: int = 1,
    groups: int = 1
) -> Tensor:
    pass
    
def conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int | Literal['valid', 'same'] = 0,
    dilation: int = 1,
    groups: int = 1
) -> Tensor:
    pass





