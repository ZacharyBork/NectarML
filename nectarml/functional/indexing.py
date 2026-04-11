import builtins

from nectarml import typing
from nectarml.tensor import Tensor, BoolTensor

def gather(input: Tensor, dim: builtins.int, index: Tensor) -> Tensor:
    return input.gather(dim, index)

def scatter(
    input:  Tensor, 
    dim:    builtins.int, 
    index:  Tensor, 
    source: Tensor
) -> Tensor:
    return input.scatter(dim, index, source)

def scatter_add(
    input:  Tensor, 
    dim:    builtins.int, 
    index:  Tensor, 
    source: Tensor
) -> Tensor:
    return input.scatter_add(dim, index, source)

def where(
    condition: BoolTensor, 
    x: Tensor | builtins.int | builtins.float, 
    y: Tensor | builtins.int | builtins.float
) -> Tensor:
    if isinstance(x, Tensor):   template = x
    elif isinstance(y, Tensor): template = y
    else: raise ValueError('Either x or y input must be a Tensor.')
    assert template.dtype != typing.bool_, \
        'x and y cannot be BoolTensors, only numerical Tensors.'
    mask = condition.to(template.device, template.dtype)
    return (x * mask + y * (1 - mask))

def masked_fill(input: Tensor, mask: Tensor, value: builtins.float) -> Tensor:
    return input.masked_fill(mask, value)

def index_select(input: Tensor, dim: builtins.int, index: Tensor) -> Tensor:
    return input.index_select(dim, index)
