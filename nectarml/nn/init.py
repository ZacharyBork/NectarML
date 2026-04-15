from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import builtins
from typing import Literal

import numpy as np

from nectarml import cuda
from nectarml.random import RNG
    
### UTILS ###

def calculate_gain(
    nonlinearity: Literal[
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d', 'sigmoid', 'tanh',
        'relu', 'leaky_relu', 'selu'
    ] = 'leaky_relu',
    a: builtins.float = 0.0
) -> builtins.float:
    if nonlinearity in [
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d', 'sigmoid'
    ]: return 1.0
    match nonlinearity:
        case 'tanh':       return 5/3
        case 'relu':       return np.sqrt(2)
        case 'leaky_relu': return np.sqrt(2 / (1 + a * a))
        case 'selu':       return 3/4
        case _: raise ValueError(f'Invalid nonlinearity: {nonlinearity}')
        
def _set_weights(weights: Tensor, data: np.ndarray) -> None:
    data = data.astype(weights.dtype)
    if weights.device == 'cuda':
        old_buffer = weights._buffer
        weights._buffer = cuda.memory.CudaBuffer(data, weights.dtype)
        old_buffer.decrement()
    else: weights.data = data
    weights.zero_grad()

### CONSTANT ###

def zeros_(weights: Tensor) -> None: 
    _set_weights(weights, np.zeros(weights.shape, dtype=weights.dtype))

def ones_(weights: Tensor) -> None: 
    _set_weights(weights, np.ones(weights.shape, dtype=weights.dtype))

def constant_(weights: Tensor, value: builtins.float) -> None: 
    _set_weights(weights, np.full(weights.shape, value, dtype=weights.dtype))

def eye_(weights: Tensor) -> None: 
    assert weights.ndim == 2, \
        'eye_ init only valid for 2 dimensional tensor.'
    s = weights.shape
    _set_weights(weights, np.eye(N=s[0], M=s[1], k=0, dtype=weights.dtype))
    
def dirac_(weights: Tensor, groups: builtins.int = 1) -> None: 
    assert weights.ndim >= 3, \
        'dirac_ init expects {3, 4, 5} dimensional tensors.'
    out_channels, in_channels = weights.shape[0], weights.shape[1]
    spatial_dims = weights.shape[2:]
    center = tuple(s // 2 for s in spatial_dims)
    data = np.zeros(weights.shape, dtype=weights.dtype)
    for i in range(min(out_channels, in_channels // groups)):
        data[(i, i) + center] = 1
    _set_weights(weights, data)

### RANDOM ###

def uniform_(
    weights: Tensor, 
    a: builtins.float = 0.0, 
    b: builtins.float = 1.0
) -> None: 
    _set_weights(weights, RNG.uniform(low=a, high=b, size=weights.shape))

def normal_(
    weights: Tensor,
    mean: builtins.float = 0.0, 
    std:  builtins.float = 1.0
) -> None: 
    _set_weights(weights, RNG.normal(loc=mean, scale=std, size=weights.shape))

### VARIANCE SCALING ###

def xavier_uniform_(weights: Tensor, gain: builtins.float = 1.0) -> None: 
    fan_in = weights.shape[-1]
    fan_out = weights.shape[0]
    
    std_dev = gain * np.sqrt(6 / (fan_in + fan_out))
    data = RNG.uniform(low=-std_dev, high=std_dev, size=weights.shape)
    _set_weights(weights, data)

def xavier_normal_(weights: Tensor, gain: builtins.float = 1.0) -> None: 
    s = weights.shape
    fan_in, fan_out = s[-1], s[0]
    std_dev = gain * np.sqrt(2 / (fan_in + fan_out))
    _set_weights(weights, RNG.normal(loc=0.0, scale=std_dev, size=s))

def kaiming_uniform_(
    weights:      Tensor, 
    a:            builtins.float = 0.0, 
    mode:         Literal['fan_in', 'fan_out'] = 'fan_in',
    nonlinearity: Literal[
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d', 'sigmoid', 'tanh',
        'relu', 'leaky_relu', 'selu'
    ] = 'leaky_relu'
) -> None: 
    s = weights.shape
    match mode:
        case 'fan_in':  features = s[1]
        case 'fan_out': features = s[0]
        case _: raise ValueError(f'Invalid init mode: {mode}')
    
    gain = calculate_gain(nonlinearity, a)
    std_dev = np.sqrt(3.0) * gain / np.sqrt(features)
    _set_weights(weights, RNG.uniform(low=-std_dev, high=std_dev, size=s))

def kaiming_normal_(
    weights:      Tensor, 
    a:            builtins.float = 0.0, 
    mode:         Literal['fan_in', 'fan_out'] = 'fan_in',
    nonlinearity: Literal[
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d', 'sigmoid', 'tanh',
        'relu', 'leaky_relu', 'selu'
    ] = 'leaky_relu'    
) -> None: 
    s = weights.shape
    match mode:
        case 'fan_in':  features = s[1]
        case 'fan_out': features = s[0]
        case _: raise ValueError(f'Invalid init mode: {mode}')
    
    gain = calculate_gain(nonlinearity, a)
    std_dev = gain / np.sqrt(features)
    _set_weights(weights, RNG.normal(loc=0.0, scale=std_dev, size=s))

### OTHER ###

def trunc_normal_(
    weights: Tensor, 
    mean:    builtins.float = 0.0, 
    std:     builtins.float = 1.0, 
    a:       builtins.float = -2.0, 
    b:       builtins.float = 2.0
) -> None:
    data = RNG.normal(mean, std, size=weights.shape)
    while True:
        invalid = (data < a) | (data > b)
        if not invalid.any(): break
        data[invalid] = RNG.normal(mean, std, size=invalid.sum())
    _set_weights(weights, data)
    
def orthogonal_(weights: Tensor, gain: builtins.float = 1.0) -> None: 
    shape = weights.shape
    flat_shape = (shape[0], np.prod(shape[1:]))
    Q, R = np.linalg.qr(RNG.normal(loc=0.5, scale=0.5, size=flat_shape))
    Q *= np.sign(np.diag(R))
    if flat_shape[0] < flat_shape[1]: Q = Q.T
    _set_weights(weights, (gain * Q).reshape(shape))
    
def sparse_(
    weights:  Tensor, 
    sparsity: builtins.float, 
    std:      builtins.float = 0.01) -> None: 
    assert 0 <= sparsity <= 1, 'Sparsity must be between 0 and 1.'
    data = np.zeros(weights.shape, dtype=weights.dtype)
    rows, cols = weights.shape[0], weights.shape[1]
    num_zeros = builtins.int(np.ceil(sparsity * rows))
    for col in range(cols):
        indices = RNG.choice(rows, size=num_zeros, replace=False)
        data[indices, col] = RNG.normal(loc=0, scale=std, size=num_zeros)
    _set_weights(weights, data)
    

