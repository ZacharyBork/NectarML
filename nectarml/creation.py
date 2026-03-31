from typing import Literal

import numpy as np
from nectarml.typing import DTypeLike, float32

from nectarml.tensor import Tensor
from nectarml.random import RNG

# CREATION / DUPLICATION

def clone(input: Tensor, requires_grad: bool | None = None) -> Tensor:
    _grad = input.requires_grad if requires_grad is None else requires_grad
    data = input.numpy()
    out = Tensor(data, input.shape, input.dtype, requires_grad=_grad)
    return out.to(input.device)
    
def zeros_like(input: Tensor, requires_grad: bool | None = None) -> Tensor:
    _grad = input.requires_grad if requires_grad is None else requires_grad
    data = np.zeros(input.shape, dtype=input.dtype)
    out = Tensor(data, input.shape, input.dtype, requires_grad=_grad)
    return out.to(input.device)

def ones_like(input: Tensor, requires_grad: bool | None = None) -> Tensor: 
    _grad = input.requires_grad if requires_grad is None else requires_grad
    data = np.ones(input.shape, dtype=input.dtype)
    out = Tensor(data, input.shape, input.dtype, requires_grad=_grad)
    return out.to(input.device)

def rand_like(input: Tensor, requires_grad: bool | None = None) -> Tensor:
    _grad = input.requires_grad if requires_grad is None else requires_grad 
    data = RNG.random(input.shape, dtype=input.dtype)
    out = Tensor(data, input.shape, input.dtype, requires_grad=_grad)
    return out.to(input.device)

def full_like(
    input: Tensor, 
    fill_value: float | int,
    requires_grad: bool | None = None
) -> Tensor: 
    _grad = input.requires_grad if requires_grad is None else requires_grad
    data = np.full(input.shape, fill_value, dtype=input.dtype)
    out = Tensor(data, input.shape, input.dtype, requires_grad=_grad)
    return out.to(input.device)

def empty_like(input: Tensor, requires_grad: bool | None = None) -> Tensor: 
    _grad = input.requires_grad if requires_grad is None else requires_grad
    data = np.empty(input.shape, dtype=input.dtype)
    out = Tensor(data, input.shape, input.dtype, requires_grad=_grad)
    return out.to(input.device)

# FIXED SHAPE

def zeros(
    shape: tuple[int, ...], 
    dtype: DTypeLike = float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.zeros(shape, dtype=dtype), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def ones(
    shape: tuple[int, ...], 
    dtype: DTypeLike = float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.ones(shape, dtype=dtype), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def rand(
    shape: tuple[int, ...], 
    seed: int | None = None,
    dtype: DTypeLike = float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    rng = np.random.default_rng(seed) if seed is not None else RNG
    return Tensor(
        rng.random(shape, dtype=dtype), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def randn(
    shape: tuple[int, ...], 
    seed: int | None = None,
    dtype: DTypeLike = float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    rng = np.random.default_rng(seed) if seed is not None else RNG
    return Tensor(
        rng.standard_normal(shape, dtype=dtype), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def full(
    shape: tuple[int, ...], 
    fill_value: float | int,
    dtype: DTypeLike = float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.full(shape, fill_value, dtype), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def empty(
    shape: tuple[int, ...], 
    dtype: DTypeLike = float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.empty(shape, dtype), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def eye(
    n: int,
    m: int | None = None,
    k: int = 0,
    dtype: DTypeLike = float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.eye(n, m, k, dtype), dtype=dtype, 
        device=device, requires_grad=requires_grad)
    
def arange(
    start: float,
    stop: float | None = None,
    step: float = 1.0,
    dtype: DTypeLike = float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    if stop is None: data = np.arange(start, step=step, dtype=dtype)
    else: data = np.arange(start, stop, step, dtype=dtype)
    out = Tensor(data, dtype=dtype, requires_grad=requires_grad)
    return out.to(device)

def linspace(
    start: float,
    stop: float,
    num_elements: int = 50,
    dtype: DTypeLike = float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    data = np.linspace(start, stop, num_elements, dtype=dtype)
    out = Tensor(data, dtype=dtype, requires_grad=requires_grad)
    return out.to(device)

def tril(
    size: int, 
    dtype: DTypeLike = float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor:
    data = np.tril(np.ones((size, size), dtype=dtype))
    out = Tensor(data, requires_grad=requires_grad)
    return out.to(device=device, dtype=dtype)
