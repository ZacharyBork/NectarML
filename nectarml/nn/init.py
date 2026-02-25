from typing import Literal

from nectarml import Tensor

### CONSTANT ###

def zeros_(input: Tensor) -> None: 
    pass

def ones_(input: Tensor) -> None: 
    pass

def constant_(input: Tensor, value: float) -> None: 
    pass

def eye_(input: Tensor) -> None: 
    pass

def dirac_(input: Tensor, groups: int = 1) -> None: 
    pass

### RANDOM ###

def uniform_(input: Tensor, a: float = 0.0, b: float = 1.0) -> None: 
    pass

def normal_(input: Tensor, mean: float = 0.0, std: float = 1.0) -> None: 
    pass

### VARIANCE SCALING ###

def xavier_uniform_(input: Tensor, gain: float = 1.0) -> None: 
    pass

def xavier_normal_(input: Tensor, gain: float = 1.0) -> None: 
    pass

def kaiming_uniform_(
    input: Tensor, 
    a: float = 0.0, 
    mode: Literal['fan_in', 'fan_out'] = 'fan_in',
    nonlinearity: Literal[
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d', 'sigmoid', 'tanh',
        'relu', 'leaky_relu', 'selu'
    ] = 'leaky_relu'
) -> None: 
    pass

def kaiming_normal_(
    input: Tensor, 
    a: float = 0.0, 
    mode: Literal['fan_in', 'fan_out'] = 'fan_in',
    nonlinearity: Literal[
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d', 'sigmoid', 'tanh',
        'relu', 'leaky_relu', 'selu'
    ] = 'leaky_relu'    
) -> None: 
    pass

### OTHER ###

def trunc_normal_(
    input: Tensor, 
    mean: float = 1.0, 
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0
) -> None: 
    pass

def othogonal_(input: Tensor, gain: float = 1.0) -> None: 
    pass

def sparse_(input: Tensor, sparsity: float, std: float = 0.01) -> None: 
    pass



