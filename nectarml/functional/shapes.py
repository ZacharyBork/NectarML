from collections.abc import Sequence

from nectarml.tensor import Tensor
from nectarml._core import shapes
from nectarml.functional.common import _eval_core_function

def reshape(input: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _eval_core_function(input, lambda x : shapes.reshape(x, shape))

def flatten(input: Tensor) -> Tensor:
    return _eval_core_function(input, shapes.flatten)

def squeeze(input: Tensor, dim: int | tuple[int, ...] | None) -> Tensor: 
    return _eval_core_function(input, lambda x : shapes.squeeze(x, dim))
    
def unsqueeze(input: Tensor, dim: int | tuple[int, ...]) -> Tensor:
    return _eval_core_function(input, lambda x : shapes.unsqueeze(x, dim))

def transpose(input: Tensor, dims: Sequence[int] | None) -> Tensor:
    return _eval_core_function(input, lambda x : shapes.transpose(x, dims))

def swapdims(input: Tensor, dim1: int, dim2: int) -> Tensor: 
    return _eval_core_function(
        input, lambda x : shapes.swapdims(x, dim1, dim2))

def permute(input: Tensor, dims: Sequence[int] | None) -> Tensor:
    return _eval_core_function(input, lambda x : shapes.permute(x, dims))

def expand(input: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _eval_core_function(input, lambda x : shapes.expand(x, shape))

def broadcast_to(input: Tensor, shape: tuple[int, ...]) -> Tensor:
    return expand(input, shape)

