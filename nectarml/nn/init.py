from typing import Literal

import numpy as np

from nectarml import Tensor

### RANDOM / SEEDING/

rng = np.random.default_rng()

def manual_seed(seed: int) -> None:
    global rng
    rng = np.random.default_rng(seed)
    
### UTILS ###

def calculate_gain(
    nonlinearity: Literal[
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d', 'sigmoid', 'tanh',
        'relu', 'leaky_relu', 'selu'
    ] = 'leaky_relu',
    a: float = 0.0
) -> float:
    if nonlinearity in [
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d', 'sigmoid'
    ]: return 1.0
    match nonlinearity:
        case 'tanh': return 5/3
        case 'relu': return np.sqrt(2)
        case 'leaky_relu': return np.sqrt(2 / (1 + a * a))
        case 'selu': return 3/4
        case _: raise ValueError(f'Invalid nonlinearity: {nonlinearity}')

### CONSTANT ###

def zeros_(weights: Tensor) -> None: 
    weights.data = np.zeros(weights.shape, dtype=weights.dtype)

def ones_(weights: Tensor) -> None: 
    weights.data = np.ones(weights.shape, dtype=weights.dtype)

def constant_(weights: Tensor, value: float) -> None: 
    weights.data = np.full(weights.shape, value, dtype=weights.dtype)

def eye_(weights: Tensor) -> None: 
    assert weights.data.ndim == 2, \
        'eye_ init only valid for 2 dimensional tensor.'
    weights.data = np.eye(
        N=weights.shape[0], M=weights.shape[1], k=0, dtype=weights.dtype)
    
def dirac_(weights: Tensor, groups: int = 1) -> None: 
    assert weights.data.ndim >= 3, \
        'dirac_ init expects {3, 4, 5} dimensional tensors.'
    out_channels, in_channels = weights.shape[0], weights.shape[1]
    spatial_dims = weights.shape[2:]
    center = tuple(s // 2 for s in spatial_dims)
    _zeros = np.zeros(weights.shape, dtype=weights.dtype)
    for i in range(min(out_channels, in_channels // groups)):
        _zeros[(i, i) + center] = 1
    weights.data = _zeros

### RANDOM ###

def uniform_(weights: Tensor, a: float = 0.0, b: float = 1.0) -> None: 
    weights.data = rng.uniform(low=a, high=b, size=weights.shape)

def normal_(weights: Tensor, mean: float = 0.0, std: float = 1.0) -> None: 
    weights.data = rng.normal(loc=mean, scale=std, size=weights.shape)

### VARIANCE SCALING ###

def xavier_uniform_(weights: Tensor, gain: float = 1.0) -> None: 
    fan_in = weights.shape[-1]
    fan_out = weights.shape[0]
    
    std_dev = gain * np.sqrt(6 / (fan_in + fan_out))
    weights.data = rng.uniform(low=-std_dev, high=std_dev, size=weights.shape)

def xavier_normal_(weights: Tensor, gain: float = 1.0) -> None: 
    fan_in = weights.shape[-1]
    fan_out = weights.shape[0]
    
    std_dev = gain * np.sqrt(2 / (fan_in + fan_out))
    weights.data = rng.normal(loc=0.0, scale=std_dev, size=weights.shape)

def kaiming_uniform_(
    weights: Tensor, 
    a: float = 0.0, 
    mode: Literal['fan_in', 'fan_out'] = 'fan_in',
    nonlinearity: Literal[
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d', 'sigmoid', 'tanh',
        'relu', 'leaky_relu', 'selu'
    ] = 'leaky_relu'
) -> None: 
    match mode:
        case 'fan_in':  features = weights.shape[1]
        case 'fan_out': features = weights.shape[0]
        case _: raise ValueError(f'Invalid init mode: {mode}')
    
    gain = calculate_gain(nonlinearity, a)
    std_dev = np.sqrt(3.0) * gain / np.sqrt(features)
    weights.data = rng.uniform(low=-std_dev, high=std_dev, size=weights.shape)

def kaiming_normal_(
    weights: Tensor, 
    a: float = 0.0, 
    mode: Literal['fan_in', 'fan_out'] = 'fan_in',
    nonlinearity: Literal[
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d', 'sigmoid', 'tanh',
        'relu', 'leaky_relu', 'selu'
    ] = 'leaky_relu'    
) -> None: 
    match mode:
        case 'fan_in':  features = weights.shape[1]
        case 'fan_out': features = weights.shape[0]
        case _: raise ValueError(f'Invalid init mode: {mode}')
    
    gain = calculate_gain(nonlinearity, a)
    std_dev = gain / np.sqrt(features)
    weights.data = rng.normal(loc=0.0, scale=std_dev, size=weights.shape)

### OTHER ###

def trunc_normal_(
    weights: Tensor, 
    mean: float = 0.0, 
    std: float = 1.0, 
    a: float = -2.0, 
    b: float = 2.0
) -> None:
    data = rng.normal(mean, std, size=weights.shape)
    while True:
        invalid = (data < a) | (data > b)
        if not invalid.any(): break
        data[invalid] = rng.normal(mean, std, size=invalid.sum())
    weights.data = data.astype(weights.dtype)
    
def orthogonal_(weights: Tensor, gain: float = 1.0) -> None: 
    shape = weights.shape
    flat_shape = (shape[0], np.prod(shape[1:]))
    Q, R = np.linalg.qr(rng.normal(size=flat_shape))
    Q *= np.sign(np.diag(R))
    if flat_shape[0] < flat_shape[1]: Q = Q.T
    weights.data = (gain * Q).reshape(shape)
    
def sparse_(weights: Tensor, sparsity: float, std: float = 0.01) -> None: 
    assert 0 <= sparsity <= 1, 'Sparsity must be between 0 and 1.'
    _zeros = np.zeros(weights.shape, dtype=weights.dtype)
    rows, cols = weights.shape[0], weights.shape[1]
    num_zeros = int(np.ceil(sparsity * rows))
    for col in range(cols):
        indices = rng.choice(rows, size=num_zeros, replace=False)
        _zeros[indices, col] = rng.normal(loc=0, scale=std, size=num_zeros)
    weights.data = _zeros
    



