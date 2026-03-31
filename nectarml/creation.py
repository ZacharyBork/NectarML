from typing import Literal

import numpy as np
from nectarml.typing import DTypeLike, float32

import nectarml.cuda as cuda
from nectarml.tensor import Tensor
from nectarml.random import RNG

# ABSTRACTS

def _build_tensor(
    input: Tensor, 
    data: np.ndarray,
    requires_grad: bool
) -> Tensor:
    return Tensor(data, None, input.dtype, input.device, requires_grad)

# CREATION / DUPLICATION

def clone(input: Tensor, requires_grad: bool | None = None) -> Tensor:
    return _build_tensor(input, input.data, requires_grad)

def zeros_like(input: Tensor, requires_grad: bool | None = None) -> Tensor:
    _grad = input.requires_grad if requires_grad is None else requires_grad
    if input.device == 'cuda': 
        ptr = cuda.alloc_cuda_full(input.size, input.dtype, 0.0)
        return Tensor(ptr, input.shape, input.dtype, 'cuda', _grad)
    else: 
        data = np.zeros_like(input.data, dtype=input.dtype)
        return _build_tensor(input, data, _grad)
    
def ones_like(input: Tensor, requires_grad: bool | None = None) -> Tensor: 
    _grad = input.requires_grad if requires_grad is None else requires_grad
    if input.device == 'cuda': 
        ptr = cuda.alloc_cuda_full(input.size, input.dtype, 1.0)
        return Tensor(ptr, input.shape, input.dtype, 'cuda', _grad)
    else: 
        data = np.ones_like(input.data, dtype=input.dtype)
        return _build_tensor(input, data, _grad)

def rand_like(input: Tensor, requires_grad: bool | None = None) -> Tensor:
    data = RNG.random(input.shape, dtype=input.dtype)
    _grad = input.requires_grad if requires_grad is None else requires_grad 
    if input.device == 'cuda':
        ptr = cuda.to_cuda(data, dtype=input.dtype)
        return Tensor(ptr, input.shape, input.dtype, 'cuda', _grad)
    else: return _build_tensor(input, data, _grad)

def full_like(
    input: Tensor, 
    fill_value: float | int,
    requires_grad: bool | None = None
) -> Tensor: 
    _grad = input.requires_grad if requires_grad is None else requires_grad
    if input.device == 'cuda':
        ptr = cuda.alloc_cuda_full(input.size, input.dtype, fill_value)
        return Tensor(ptr, input.shape, input.dtype, 'cuda', _grad)
    else: 
        data = np.full_like(input.data, fill_value, dtype=input.dtype)
        return _build_tensor(input, data, _grad)

def empty_like(input: Tensor, requires_grad: bool | None = None) -> Tensor: 
    _grad = input.requires_grad if requires_grad is None else requires_grad
    if input.device == 'cuda':
        ptr = cuda.alloc_cuda_empty(input.size, input.dtype)
        return Tensor(ptr, input.shape, input.dtype, 'cuda', _grad)
    else: return _build_tensor(
        input, np.empty_like(input.data, dtype=input.dtype), _grad)

def tril(
    size: int, 
    dtype: DTypeLike = float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor:
    data = np.tril(np.ones((size, size), dtype=dtype))
    out = Tensor(data, requires_grad=requires_grad)
    return out.to(device=device, dtype=dtype)

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
    return Tensor(
        data, dtype=dtype, device=device, requires_grad=requires_grad)

def linspace(
    start: float,
    stop: float,
    num_elements: int = 50,
    dtype: DTypeLike = float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.linspace(start, stop, num_elements, dtype=dtype), dtype=dtype, 
        device=device, requires_grad=requires_grad)

