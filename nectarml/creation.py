from typing import Literal

import numpy as np
from numpy.typing import DTypeLike

from nectarml import Tensor

_rng = np.random.default_rng()

# ABSTRACTS

def _build_tensor(
    input: Tensor, 
    data: np.ndarray,
    requires_grad: bool | None
) -> Tensor:
    return Tensor(
        data=data, dtype=input.dtype, device=input.device, 
        requires_grad=input.requires_grad if requires_grad is None \
            else requires_grad)

# CREATION / DUPLICATION

def clone(input: Tensor, requires_grad: bool | None = None) -> Tensor:
    return _build_tensor(input, input.data, requires_grad)

def zeros_like(input: Tensor, requires_grad: bool | None = None) -> Tensor:
    return _build_tensor(input, np.zeros_like(input.data), requires_grad)
    
def ones_like(input: Tensor, requires_grad: bool | None = None) -> Tensor: 
    return _build_tensor(input, np.ones_like(input.data), requires_grad)

def rand_like(input: Tensor, requires_grad: bool | None = None) -> Tensor: 
    return _build_tensor(
        input, _rng.random(input.data.shape), requires_grad)

def full_like(
    input: Tensor, 
    fill_value: float | int,
    requires_grad: bool | None = None
) -> Tensor: 
    return _build_tensor(
        input, np.full_like(input.data, fill_value), requires_grad)

def empty_like(input: Tensor, requires_grad: bool | None = None) -> Tensor: 
    return _build_tensor(input, np.empty_like(input.data), requires_grad)

# FIXED SHAPE

def zeros(
    shape: tuple[int, ...], 
    dtype: DTypeLike = np.float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.zeros(shape), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def ones(
    shape: tuple[int, ...], 
    dtype: DTypeLike = np.float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.ones(shape), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def rand(
    shape: tuple[int, ...], 
    dtype: DTypeLike = np.float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        _rng.random(shape), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def randn(
    shape: tuple[int, ...], 
    seed: int | None = None,
    dtype: DTypeLike = np.float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    rng = np.random.default_rng(seed) if seed is not None else _rng
    return Tensor(
        rng.standard_normal(shape), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def full(
    shape: tuple[int, ...], 
    fill_value: float | int,
    dtype: DTypeLike = np.float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.full(shape, fill_value), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def empty(
    shape: tuple[int, ...], 
    dtype: DTypeLike = np.float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.empty(shape), shape=shape, dtype=dtype, 
        device=device, requires_grad=requires_grad)

def eye(
    n: int,
    m: int | None = None,
    k: int = 0,
    dtype: DTypeLike = np.float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.eye(n, m, k), dtype=dtype, 
        device=device, requires_grad=requires_grad)

def arange(
    start: float,
    stop: float | None = None,
    step: float = 1.0,
    dtype: DTypeLike = np.float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    if stop is None: data = np.arange(start, step=step)
    else: data = np.arange(start, stop, step)
    return Tensor(
        data, dtype=dtype, device=device, requires_grad=requires_grad)

def linspace(
    start: float,
    stop: float,
    num_elements: int = 50,
    dtype: DTypeLike = np.float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    requires_grad: bool = False
) -> Tensor: 
    return Tensor(
        np.linspace(start, stop, num_elements), dtype=dtype, 
        device=device, requires_grad=requires_grad)

