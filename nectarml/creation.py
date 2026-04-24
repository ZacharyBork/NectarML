
import numpy as np

from nectarml        import typing
from nectarml.random import RNG, Random
from nectarml.core import Tensor

# CREATION / DUPLICATION

def clone(input: Tensor, requires_grad: bool | None = None) -> Tensor:
    out = input.clone()
    out._prev = None
    out.requires_grad = input.requires_grad \
        if requires_grad is None else requires_grad
    return out
    
def zeros_like(input: Tensor, requires_grad: bool | None = None) -> Tensor:
    _grad = input.requires_grad if requires_grad is None else requires_grad
    data = np.zeros(input.shape, dtype=input.dtype.numpy)
    return Tensor(data, input.shape, input.dtype, input.device, _grad)

def ones_like(input: Tensor, requires_grad: bool | None = None) -> Tensor: 
    _grad = input.requires_grad if requires_grad is None else requires_grad
    data = np.ones(input.shape, dtype=input.dtype.numpy)
    return Tensor(data, input.shape, input.dtype, input.device, _grad)

def rand_like(input: Tensor, requires_grad: bool | None = None) -> Tensor:
    _grad = input.requires_grad if requires_grad is None else requires_grad 
    data = RNG.random(input.shape, dtype=input.dtype.numpy)
    return Tensor(data, input.shape, input.dtype, input.device, _grad)

def full_like(
    input:         Tensor, 
    fill_value:    float | int,
    requires_grad: bool | None = None
) -> Tensor: 
    _grad = input.requires_grad if requires_grad is None else requires_grad
    data = np.full(input.shape, fill_value, dtype=input.dtype.numpy)
    return Tensor(data, input.shape, input.dtype, input.device, _grad)

def empty_like(input: Tensor, requires_grad: bool | None = None) -> Tensor: 
    _grad = input.requires_grad if requires_grad is None else requires_grad
    data = np.empty(input.shape, dtype=input.dtype.numpy)
    return Tensor(data, input.shape, input.dtype, input.device, _grad)

# FIXED SHAPE

def zeros(
    shape:         tuple[int, ...], 
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.zeros(shape, dtype=dtype.numpy), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def ones(
    shape:         tuple[int, ...], 
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.ones(shape, dtype=dtype.numpy), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def rand(
    shape:         tuple[int, ...], 
    seed:          int | None = None,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    rng = Random(seed) if seed is not None else RNG
    return Tensor(
        rng.random(shape, dtype=dtype), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def randn(
    shape:         tuple[int, ...], 
    seed:          int | None = None,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    rng = Random(seed) if seed is not None else RNG
    return Tensor(
        rng.standard_normal(shape, dtype=dtype.numpy), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def full(
    shape:         tuple[int, ...], 
    fill_value:    float | int,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.full(shape, fill_value, dtype.numpy), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def empty(
    shape:         tuple[int, ...], 
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.empty(shape, dtype.numpy), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def eye(
    n:             int,
    m:             int | None = None,
    k:             int = 0,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.eye(n, m, k, dtype.numpy), dtype=dtype, 
        device=device, requires_grad=requires_grad)
    
def arange(
    start:         float,
    stop:          float | None = None,
    step:          float = 1.0,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    if stop is None: data = np.arange(start, step=step, dtype=dtype.numpy)
    else: data = np.arange(start, stop, step, dtype=dtype.numpy)
    return Tensor(data, data.shape, dtype, device, requires_grad)

def linspace(
    start:         float,
    stop:          float,
    num_elements:  int = 50,
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    data = np.linspace(start, stop, num_elements, dtype=dtype.numpy)
    return Tensor(data, data.shape, dtype, device, requires_grad)

def tril(
    size:          int, 
    dtype:         typing.dtype = typing.float32,
    device:        typing.DeviceLikeType = 'cpu',
    requires_grad: bool = False
) -> Tensor:
    data = np.tril(np.ones((size, size), dtype=dtype.numpy))
    return Tensor(data, data.shape, dtype, device, requires_grad)
