from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import builtins
from   typing import Literal

import numpy as np

from nectarml        import cuda
from nectarml.random import RNG
    
### UTILS ###

def calculate_gain(
    nonlinearity: Literal[
        'linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'selu',
        'conv1d', 'conv2d', 'conv3d', 
        'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d'
    ] = 'leaky_relu',
    a: builtins.float = 0.0
) -> builtins.float:
    '''Calculates gain for various nonlinear functions.
    
    Args:
        nonlinearity : The nonlinear function to calculate the gain for.
        a            : Argument to pass to the nonlinear function. Currently
                       only used for negative slope for `leaky_relu`.
            
    Returns:
        float : The calculated gain value.
    '''
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
    '''Sets the values of a given weight tensor in-place to the provided data.

    If the given `weights` tensor's device is CPU, the given data simply 
    replaces the tensor's data directly. If the `weights` tensor's device is
    CUDA, however, the data is moved to device memory, then the `weights`
    tensor has its CudaBuffer decremented and replaced with a new one pointing
    to the data in device memory.
    
    Args:
        weights : The weights tensor to overwrite the data of.
        data    : The data to overwrite the `weights` tensor's data with.
    '''
    data = data.astype(weights.dtype.cpu)
    if weights.device == 'cuda':
        ptr = cuda.data_to_cuda(data, weights.size, weights.dtype)
        weights._buffer.decrement()
        weights._buffer = cuda.memory.CudaBuffer(ptr, data.size, weights.dtype)
    else: weights.data = data
    weights.zero_grad()

### CONSTANT ###

def zeros_(weights: Tensor) -> None: 
    '''Sets all elements of a given `weights` tensor to 0.0.

    Note: This function modifies the provided `weights` tensor in-place.
    
    Args:
        weights : The weights tensor to initialize the values of.
    '''
    _set_weights(weights, np.zeros(weights.shape, dtype=weights.dtype.cpu))

def ones_(weights: Tensor) -> None: 
    '''Sets all elements of a given `weights` tensor to 1.0.

    Note: This function modifies the provided `weights` tensor in-place.
    
    Args:
        weights : The weights tensor to initialize the values of.
    '''
    _set_weights(weights, np.ones(weights.shape, dtype=weights.dtype.cpu))

def constant_(weights: Tensor, value: builtins.float) -> None: 
    '''Sets all elements of a given `weights` tensor to a constant value. 

    Note: This function modifies the provided `weights` tensor in-place.
    
    Args:
        weights : The weights tensor to initialize the values of.
        value   : The constant value to set all elements of the `weights` 
                  tensor's data to.
    '''
    _set_weights(
        weights, np.full(weights.shape, value, dtype=weights.dtype.cpu))

def eye_(weights: Tensor) -> None: 
    '''Initializes a 2-dimensional `weights` tensor with an identity matrix.

    Note: This function modifies the provided `weights` tensor in-place.
    
    Args:
        weights : The weights tensor to initialize the values of.
    '''
    assert weights.ndim == 2, \
        'eye_ init only valid for 2 dimensional tensor.'
    s = weights.shape
    _set_weights(weights, np.eye(N=s[0], M=s[1], k=0, dtype=weights.dtype.cpu))
    
def dirac_(weights: Tensor, groups: builtins.int = 1) -> None:
    '''Fills a given `weights` tensor with the Dirac delta function.

    Note: This function modifies the provided `weights` tensor in-place.

    Initializes the `weights` tensor with a convolutional filter, causing it to
    act as an identity early in training. Useful for helping layers learn
    deviations from the baseline. Conceptually similar to residual connections
    for deep learning. 

    Args:
        weights : The weights tensor to initialize the values of.
        groups  : The number of groups in the convolutional layer.
    '''
    assert weights.ndim >= 3, \
        'dirac_ init expects {3, 4, 5} dimensional tensors.'
    out_channels, in_channels = weights.shape[0], weights.shape[1]
    spatial_dims = weights.shape[2:]
    center       = tuple(s // 2 for s in spatial_dims)
    data         = np.zeros(weights.shape, dtype=weights.dtype.cpu)
    for i in range(min(out_channels, in_channels // groups)):
        data[(i, i) + center] = 1
    _set_weights(weights, data)

### RANDOM ###

def uniform_(
    weights: Tensor, 
    a:       builtins.float = 0.0, 
    b:       builtins.float = 1.0
) -> None: 
    '''Fills a tensor with random values drawn from a uniform distribution.

    Note: This function modifies the provided `weights` tensor in-place.
    
    Args:
        weights : The weights tensor to initialize the values of.
        a       : The min value (inclusive) for the uniform distribution.
        b       : The max value (non-inclusive) for the uniform distribution.
    '''
    _set_weights(weights, RNG.uniform(low=a, high=b, size=weights.shape))

def normal_(
    weights: Tensor,
    mean: builtins.float = 0.0, 
    std:  builtins.float = 1.0
) -> None: 
    '''Fills a tensor with random values drawn from a normal distribution.

    Note: This function modifies the provided `weights` tensor in-place.
    
    Args:
        weights : The weights tensor to initialize the values of.
        mean    : The mean value of the normal distribution.
        std     : The standard deviation of the normal distribution.
    '''
    _set_weights(weights, RNG.normal(loc=mean, scale=std, size=weights.shape))

### VARIANCE SCALING ###

def xavier_uniform_(weights: Tensor, gain: builtins.float = 1.0) -> None:
    '''Fills a given tensor with a Xavier (Glorot) uniform distribution.

    Note: This function modifies the provided `weights` tensor in-place.

    Draws random values from a uniform distribution and scales them by the 
    number of input and outputs of the given layer (fan-in, fan-out). Generally 
    used for saturating activation functions such as tanh and sigmoid.

    Reference: 
        Glorot, Xavier and Bengio, Yoshua. "Understanding the difficulty of 
        training deep feedforward neural networks.", (2010). 
        https://proceedings.mlr.press/v9/glorot10a.html
    
    Args:
        weights : The weights tensor to initialize the values of.
        gain    : Scaling factor for the weights.
    ''' 
    s           = weights.shape
    kernel_size = int(np.prod(s[2:])) if len(s) > 2 else 1
    fan_in      = weights.shape[-1] * kernel_size
    fan_out     = weights.shape[0]  * kernel_size
    
    std_dev = gain * np.sqrt(6 / (fan_in + fan_out))
    data = RNG.uniform(low=-std_dev, high=std_dev, size=weights.shape)
    _set_weights(weights, data)

def xavier_normal_(weights: Tensor, gain: builtins.float = 1.0) -> None: 
    '''Fills a given tensor with a Xavier (Glorot) normal distribution.

    Note: This function modifies the provided `weights` tensor in-place.

    Draws random values from a normal distribution and scales them by the 
    number of input and outputs of the given layer (fan-in, fan-out). Generally
    used for saturating activation functions such as tanh and sigmoid.

    Reference: 
        Glorot, Xavier and Bengio, Yoshua. "Understanding the difficulty of 
        training deep feedforward neural networks.", (2010). 
        https://proceedings.mlr.press/v9/glorot10a.html
    
    Args:
        weights : The weights tensor to initialize the values of.
        gain    : Scaling factor for the weights.
    ''' 
    s = weights.shape
    kernel_size     = int(np.prod(s[2:])) if len(s) > 2 else 1
    fan_in, fan_out = s[-1] * kernel_size, s[0] * kernel_size
    std_dev         = gain * np.sqrt(2 / (fan_in + fan_out))
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
    '''Fills a given tensor with a Kaiming (He) uniform distribution.

    Note: This function modifies the provided `weights` tensor in-place.

    Draws random values from a uniform distribution and scales them in one
    direction (`fan_in` or `fan_out`) to account for the loss of input
    information from ReLU-style activation function.

    Reference: 
        He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Delving deep 
        into rectifiers: Surpassing human-level performance on imagenet 
        classification.", (2015). https://arxiv.org/abs/1502.01852
    
    Args:
        weights      : The weights tensor to initialize the values of.
        a            : Optional argument for the non-linear function when
                       calculating gain. Only used for negative slope when
                       `nonlinearity`=`leaky_relu`.
        mode         : The direction to scale the weights. Options are 
                       [`fan_in`, `fan_out`]. `fan_in` better preserves the 
                       weights in the forward pass, where `fan_out` better 
                       preserves the weights in the backward pass.
        nonlinearity : The nonlinear activation function to calculate the
                       weights gain for.
    ''' 
    s           = weights.shape
    kernel_size = int(np.prod(s[2:])) if len(s) > 2 else 1
    match mode:
        case 'fan_in':  features = s[1] * kernel_size
        case 'fan_out': features = s[0] * kernel_size
        case _: raise ValueError(f'Invalid init mode: {mode}')
    
    gain    = calculate_gain(nonlinearity, a)
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
    '''Fills a given tensor with a Kaiming (He) normal distribution.

    Note: This function modifies the provided `weights` tensor in-place.

    Draws random values from a normal distribution and scales them in one
    direction (`fan_in` or `fan_out`) to account for the loss of input
    information from ReLU-style activation function.

    Reference: 
        He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Delving deep 
        into rectifiers: Surpassing human-level performance on imagenet 
        classification.", (2015). https://arxiv.org/abs/1502.01852
    
    Args:
        weights      : The weights tensor to initialize the values of.
        a            : Optional argument for the non-linear function when
                       calculating gain. Only used for negative slope when
                       `nonlinearity`=`leaky_relu`.
        mode         : The direction to scale the weights. Options are 
                       [`fan_in`, `fan_out`]. `fan_in` better preserves the 
                       weights in the forward pass, where `fan_out` better 
                       preserves the weights in the backward pass.
        nonlinearity : The nonlinear activation function to calculate the
                       weights gain for.
    ''' 
    s           = weights.shape
    kernel_size = int(np.prod(s[2:])) if len(s) > 2 else 1
    match mode:
        case 'fan_in':  features = s[1] * kernel_size
        case 'fan_out': features = s[0] * kernel_size
        case _: raise ValueError(f'Invalid init mode: {mode}')
    
    gain    = calculate_gain(nonlinearity, a)
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
    '''Fills a given tensor with a truncated normal distribution.

    Note: This function modifies the provided `weights` tensor in-place.
    
    Args:
        weights : The weights tensor to initialize the values of.
        mean    : The mean value for the normal distribution.
        std     : The standard deviation for the normal distribution.
        a       : The minimum value for the weight initialization.
        b       : The maximum value for the weight initialization.
    ''' 
    data = RNG.normal(mean, std, size=weights.shape)
    while True:
        invalid = (data < a) | (data > b)
        if not invalid.any(): break
        data[invalid] = RNG.normal(mean, std, size=invalid.sum())
    _set_weights(weights, data)
    
def orthogonal_(weights: Tensor, gain: builtins.float = 1.0) -> None:
    '''Fills a given tensor with a random orthogonal matrix.

    Note: This function modifies the provided `weights` tensor in-place.

    Rows and columns which are perpendicular to one another in an orthogonal 
    matrix will have unit lenth. Thus, multiplying by the weight matrix will
    preserve the length of the multiplicand vector exactly. Useful to help 
    avoid vanishing/exploding gradients in networks where you repeatedly 
    multiply a vector by the same weight matrix.

    Reference:
        Saxe, Andrew M., James L. McClelland, and Surya Ganguli. "Exact 
        solutions to the nonlinear dynamics of learning in deep linear neural 
        networks.", (2013). https://arxiv.org/abs/1312.6120

    Args:
        weights : The weights tensor to initialize the values of.
        gain    : Scaling factor for the weights.
    ''' 
    shape      = weights.shape
    flat_shape = (shape[0], np.prod(shape[1:]))
    Q, R       = np.linalg.qr(RNG.normal(loc=0.5, scale=0.5, size=flat_shape))
    Q         *= np.sign(np.diag(R))
    
    if flat_shape[0] < flat_shape[1]: Q = Q.T
    _set_weights(weights, (gain * Q).reshape(shape))
    
def sparse_(
    weights:  Tensor, 
    sparsity: builtins.float, 
    std:      builtins.float = 0.01
) -> None: 
    '''Fills tensor sparsely with values drawn from a normal distribution.

    Note: This function modifies the provided `weights` tensor in-place.

    Most values in the given `weights` tensor will be set to 0.0, with a small
    subset being instead set to random values drawn from a normal distribution.
    This initialization technique functions on the hypothesis that the sparse
    connectivity of the resulting network encourages neurons to learn more
    independently of one another, allowing them to specialize in certain 
    features.

    Reference:
        Martens, James. "Deep learning via Hessian-free optimization.", (2010).
        https://dl.acm.org/doi/10.5555/3104322.3104416

    Args:
        weights : The weights tensor to initialize the values of.
        spasity : The ratio (0-1) of 0.0 weights to normal weights.
        std     : The standard deviation of the normal distribution.
    '''
    assert 0 <= sparsity <= 1, 'Sparsity must be between 0 and 1.'
    data       = np.zeros(weights.shape, dtype=weights.dtype.cpu)
    rows, cols = weights.shape[0], weights.shape[1]
    num_zeros  = builtins.int(np.ceil(sparsity * rows))
    
    for col in range(cols):
        indices = RNG.choice(rows, size=num_zeros, replace=False)
        data[indices, col] = RNG.normal(loc=0, scale=std, size=num_zeros)
    _set_weights(weights, data)
    

