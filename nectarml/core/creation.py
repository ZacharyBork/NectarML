import builtins

import numpy as np

from nectarml        import typing
from nectarml.random import RNG, Random
from nectarml.core   import Tensor

###############################################################################
# TEMPLATING / DUPLICATION
###############################################################################

def clone(input: Tensor) -> Tensor:
    '''Creates a clone of a given tensor.

    Creates a new tensor with the same shape, dtype, and data as a given input
    tensor. Please note that unlike `tensor.clone()`, this function is NOT 
    differentiable. Clones created via this function will be severed from the
    computation graph. If you would like the cloned tensor to participate in
    backpropagation, please see `tensor.clone()` instead.

    Args:
        input : The tensor to clone.
                        
    Returns:
        Tensor : The newly created tensor.
    '''
    out               = input.clone()
    out._prev         = None
    out.requires_grad = False
    return out
    
def zeros_like(
    input:         Tensor, 
    requires_grad: builtins.bool | None = None
) -> Tensor:
    '''Creates a clone of a given tensor, filled with zeros.

    The resulting tensor will have the same shape and dtype as the input tensor
    and will have every element set to zero.

    Args:
        input         : The tensor to clone.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph. If `requires_grad` is None, the 
                        resulting tensor will require grad if the input 
                        requires grad, otherwise it will not.
                        
    Returns:
        Tensor : The newly created tensor.
    '''
    _grad = input.requires_grad if requires_grad is None else requires_grad
    data  = np.zeros(input.shape, dtype=input.dtype.numpy)
    return Tensor(data, input.shape, input.dtype, input.device, _grad)

def ones_like(
    input:         Tensor, 
    requires_grad: builtins.bool | None = None
) -> Tensor:
    '''Creates a clone of a given tensor, filled with ones.

    The resulting tensor will have the same shape and dtype as the input tensor
    and will have every element set to one.

    Args:
        input         : The tensor to clone.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph. If `requires_grad` is None, the 
                        resulting tensor will require grad if the input 
                        requires grad, otherwise it will not.
                        
    Returns:
        Tensor : The newly created tensor.
    '''
    _grad = input.requires_grad if requires_grad is None else requires_grad
    data  = np.ones(input.shape, dtype=input.dtype.numpy)
    return Tensor(data, input.shape, input.dtype, input.device, _grad)

def rand_like(
    input:         Tensor, 
    requires_grad: builtins.bool | None = None
) -> Tensor:
    '''Creates a clone of a given tensor, filled with random values.

    The random values will be drawn from a uniform distribution, and the 
    resulting tensor will have the same shape and dtype as the input tensor.

    Args:
        input         : The tensor to clone.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph. If `requires_grad` is None, the 
                        resulting tensor will require grad if the input 
                        requires grad, otherwise it will not.
                        
    Returns:
        Tensor : The newly created tensor.
    '''
    _grad = input.requires_grad if requires_grad is None else requires_grad 
    data  = RNG.random(input.shape, dtype=input.dtype)
    return Tensor(data, input.shape, input.dtype, input.device, _grad)

def randn_like(
    input:         Tensor, 
    requires_grad: builtins.bool | None = None
) -> Tensor:
    '''Creates a clone of a given tensor, filled with random values.

    The random values will be drawn from a normal distribution, and the 
    resulting tensor will have the same shape and dtype as the input tensor.

    Args:
        input         : The tensor to clone.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph. If `requires_grad` is None, the 
                        resulting tensor will require grad if the input 
                        requires grad, otherwise it will not.
                        
    Returns:
        Tensor : The newly created tensor.
    '''
    _grad = False if requires_grad is None else requires_grad
    data  = RNG.normal(loc=0.0, scale=1.0, size=input.shape)
    data  = data.astype(input.dtype.numpy)
    return Tensor(data, input.shape, input.dtype, input.device, _grad)

def full_like(
    input:         Tensor, 
    fill_value:    builtins.float | builtins.int,
    requires_grad: builtins.bool | None = None
) -> Tensor: 
    '''Creates a clone of a given tensor, filled with a specified value.

    The resulting tensor will have the same shape and dtype as the input tensor
    and will have every element set to a specified value.

    Args:
        input         : The tensor to clone.
        fill_value    : The value to fill the new tensor with.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph. If `requires_grad` is None, the 
                        resulting tensor will require grad if the input 
                        requires grad, otherwise it will not.
                        
    Returns:
        Tensor : The newly created tensor.
    '''
    _grad = input.requires_grad if requires_grad is None else requires_grad
    data  = np.full(input.shape, fill_value, dtype=input.dtype.numpy)
    return Tensor(data, input.shape, input.dtype, input.device, _grad)

def empty_like(
    input:         Tensor, 
    requires_grad: builtins.bool | None = None
) -> Tensor:
    '''Creates an empty tensor with the same shape and dtype as the input.

    Args:
        input         : The tensor to use as a template.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph. If `requires_grad` is None, the 
                        resulting tensor will require grad if the input 
                        requires grad, otherwise it will not.
                        
    Returns:
        Tensor : The newly created tensor.
    '''
    _grad = input.requires_grad if requires_grad is None else requires_grad
    data  = np.empty(input.shape, dtype=input.dtype.numpy)
    return Tensor(data, input.shape, input.dtype, input.device, _grad)

###############################################################################
# FIXED SHAPE
###############################################################################

def zeros(
    *shape:        typing.ShapeType,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: builtins.bool = False
) -> Tensor:
    '''Creates a new tensor of a given shape filled with zeros.

    Args:
        shape         : The shape of the new tensor.
        dtype         : The DType of the new tensor.
        device        : The device to build the new tensor on.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph.
                        
    Returns:
        Tensor : The newly created tensor.
    '''
    shape = Tensor._normalize_shape_input(*shape)
    return Tensor(
        np.zeros(shape, dtype=dtype.numpy), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def ones(
    *shape:        typing.ShapeType,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: builtins.bool = False
) -> Tensor: 
    '''Creates a new tensor of a given shape filled with ones.

    Args:
        shape         : The shape of the new tensor.
        dtype         : The DType of the new tensor.
        device        : The device to build the new tensor on.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph.
                        
    Returns:
        Tensor : The newly created tensor.
    '''
    shape = Tensor._normalize_shape_input(*shape)
    return Tensor(
        np.ones(shape, dtype=dtype.numpy), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def rand(
    *shape:        typing.ShapeType,
    seed:          builtins.int | None = None,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: builtins.bool = False
) -> Tensor: 
    '''Creates a tensor filled with random values from a uniform distribution.

    Args:
        shape         : The shape of the new tensor.
        seed          : The seed to use for the random value generation, or 
                        None to use the globally seeded random generator.
        dtype         : The DType of the new tensor.
        device        : The device to build the new tensor on.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph.
                        
    Returns:
        Tensor : The newly created tensor.
    '''
    shape = Tensor._normalize_shape_input(*shape)
    rng   = Random(seed) if seed is not None else RNG
    return Tensor(
        rng.random(shape, dtype=dtype), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def randn(
    *shape:        typing.ShapeType,
    seed:          builtins.int | None = None,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: builtins.bool = False
) -> Tensor: 
    '''Creates a tensor filled with random values from a normal distribution.

    Args:
        shape         : The shape of the new tensor.
        seed          : The seed to use for the random value generation, or 
                        None to use the globally seeded random generator.
        dtype         : The DType of the new tensor.
        device        : The device to build the new tensor on.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph.
                        
    Returns:
        Tensor : The newly created tensor.
    '''
    shape = Tensor._normalize_shape_input(*shape)
    rng   = Random(seed) if seed is not None else RNG
    return Tensor(
        rng.standard_normal(shape, dtype=dtype), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def full(
    *shape:        typing.ShapeType,
    fill_value:    builtins.float | builtins.int,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: builtins.bool = False
) -> Tensor: 
    '''Creates a tensor of a given shapes filled with a specified value.

    Args:
        shape         : The shape of the new tensor.
        fill_value    : The value to fill the new tensor with.
        dtype         : The DType of the new tensor.
        device        : The device to build the new tensor on.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph.
                        
    Returns:
        Tensor : The newly created tensor.
    '''
    shape = Tensor._normalize_shape_input(*shape)
    return Tensor(
        np.full(shape, fill_value, dtype.numpy), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def empty(
    *shape:        typing.ShapeType,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: builtins.bool = False
) -> Tensor:
    '''Creates a new empty tensor of a specified shape.

    Args:
        shape         : The shape of the new tensor.
        dtype         : The DType of the new tensor.
        device        : The device to build the new tensor on.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph.
                        
    Returns:
        Tensor : The newly created tensor.
    '''
    shape = Tensor._normalize_shape_input(*shape)
    return Tensor(
        np.empty(shape, dtype.numpy), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

###############################################################################
# OTHER
###############################################################################

def eye(
    n:             builtins.int,
    m:             builtins.int | None = None,
    k:             builtins.int = 0,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: builtins.bool = False
) -> Tensor: 
    '''Creates a new tensor filled with values from an identity matrix.

    Args:
        n             : The number of rows in the output.
        m             : The number of columns in the output. If None, defaults
                        to the number of rows (`n`).
        k             : The index of the diagonal. Positive values indicate
                        upper diagonals, negative values indicate lower.
        dtype         : The DType of the new tensor.
        device        : The device to build the new tensor on.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph.
    Returns:
        Tensor : The newly created tensor.
    '''
    return Tensor(
        np.eye(n, m, k, dtype.numpy), dtype=dtype, 
        device=device, requires_grad=requires_grad)
    
def arange(
    start:         builtins.float,
    stop:          builtins.float | None = None,
    step:          builtins.float = 1.0,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: builtins.bool = False
) -> Tensor: 
    '''Creates a tensor filled with evenly stepped values over a given range.

    Args:
        start         : The starting point of the value range (inclusive).
        stop          : The ending point of the value range (non-inclusive).
        step          : The size of the step between each value in the range.
        dtype         : The DType of the new tensor.
        device        : The device to build the new tensor on.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph.
    Returns:
        Tensor : The newly created tensor.
    '''
    if stop is None: data = np.arange(start, step=step, dtype=dtype.numpy)
    else: data = np.arange(start, stop, step, dtype=dtype.numpy)
    return Tensor(data, data.shape, dtype, device, requires_grad)

def linspace(
    start:         builtins.float,
    stop:          builtins.float,
    num_elements:  builtins.int = 50,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: builtins.bool = False
) -> Tensor: 
    '''Creates a new tensor filled values evenly distributed over a range.

    Args:
        start         : The starting point of the value range.
        end           : The ending point of the value range.
        num_elements  : The number of elements to distribute along the range.
        dtype         : The DType of the new tensor.
        device        : The device to build the new tensor on.
        requires_grad : Whether the newly created tensor should be connected to
                        the computation graph.
    Returns:
        Tensor : The newly created tensor.
    '''
    data = np.linspace(start, stop, num_elements, dtype=dtype.numpy)
    return Tensor(data, data.shape, dtype, device, requires_grad)

def tril(
    size:          builtins.int, 
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: builtins.bool = False
) -> Tensor:
    data = np.tril(np.ones((size, size), dtype=dtype.numpy))
    return Tensor(data, data.shape, dtype, device, requires_grad)
