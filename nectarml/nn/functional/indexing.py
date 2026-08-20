import builtins

from nectarml      import typing
from nectarml.core import Tensor, BoolTensor, creation

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
    x:         Tensor | builtins.int | builtins.float, 
    y:         Tensor | builtins.int | builtins.float,
    dtype:              typing.dtype | None = None,
    device:    typing.DeviceLikeType | None = None
) -> Tensor:
    '''Fills tensor with one of two values based on a conditional.

    This function generally expects that either `x` or `y` is a tensor. If `x`
    if a tensor, it will return a tensor of the same DType and on the same
    device as `x`. If `x` is not a tensor, it will check `y`. If neither is a
    tensor, `dtype` and `device` must be provided, and the function will create
    a new tensor to serve as output.

    Args:
        condition : The condition to evaluate. For example:
                    ```
                    x = F.where(nectarml.rand(1, 4, 4) > 0.5, 1.0, 0.0)
                    ```
        x         : The value to fill with where the condition is met.
        y         : The value to fill with where the condition is not met.
        dtype     : The DType for the new tensor. Only required if neither `x` 
                    nor `y` is a tensor.
        device    : The device for the new tensor. Only required if neither `x`
                    nor `y` is a tensor.
        
    Returns:
        Tensor : The result of the conditional fill.
    '''
    if   isinstance(x, Tensor): template = x
    elif isinstance(y, Tensor): template = y
    else: 
        assert dtype is not None, \
            'where() requires a dtype if a template tensor is not provided.'
        assert device is not None, \
            'where() requires a device if a template tensor is not provided.'
        x = creation.full(
            condition.shape, fill_value=x, 
            dtype=dtype, device=device)
        template = x
    assert template.dtype != typing.bool_, \
        'x and y cannot be BoolTensors, only numerical Tensors.'
    
    mask = condition.to(template.device, template.dtype)
    return (x * mask + y * (1 - mask))

def masked_fill(
    input: Tensor, 
    mask:  Tensor | BoolTensor, 
    value: builtins.float | builtins.int
) -> Tensor:
    '''Fills a tensor with a given value based on a mask.

    The mask tensor must be on the same device as the input tensor, and their
    shapes must be broadcastable.
    
    Args:
        input : The tensor to apply the masked fill to.
        mask  : The mask tensor. If the tensor is numerical, the value range
                should be (0-1). Can also be a BoolTensor where True values
                will be filled, and False values will not.
        value : The value to fill the masked region with.
    
    Returns:
        Tensor : The resulting filled tensor.
    '''
    return input.masked_fill(mask, value)

def index_select(input: Tensor, dim: builtins.int, index: Tensor) -> Tensor:
    return input.index_select(dim, index)
