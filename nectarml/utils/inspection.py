import numpy as np

from nectarml.core import Tensor
from nectarml.cuda.utils import (
    is_inf    as cuda_is_inf,
    is_finite as cuda_is_finite,
    is_nan    as cuda_is_nan,
    has_nan   as cuda_has_nan,
    has_inf   as cuda_has_inf
)

def is_inf(tensor: Tensor) -> bool:
    '''Checks whether every element of a given Tensor's data is infinite.
    
    Args:
        Tensor : The Tensor to check.
    
    Returns:
        bool : True if all elements of the given Tensor's data are infinite,
            otherwise False.
    '''
    if tensor.device == 'cuda': return cuda_is_inf(tensor)
    else: return np.all(np.isinf(tensor.data))

def is_finite(tensor: Tensor) -> bool:
    '''Checks whether every element of a given Tensor's data is finite.
    
    Args:
        Tensor : The Tensor to check.
    
    Returns:
        bool : True if all elements of the given Tensor's data are finite,
            otherwise False.
    '''
    if tensor.device == 'cuda': return cuda_is_finite(tensor)
    else: return np.all(np.isfinite(tensor.data))

def is_nan(tensor: Tensor) -> bool:
    '''Checks whether every element of a given Tensor's data is not a number.
    
    Args:
        Tensor : The Tensor to check.
    
    Returns:
        bool : True if all elements of the given Tensor's data are NaN,
            otherwise False.
    '''
    if tensor.device == 'cuda': return cuda_is_nan(tensor)
    else: return np.all(np.isnan(tensor.data))

def has_nan(tensor: Tensor) -> bool:
    '''Checks whether any element of a given Tensor's data is not a number.
    
    Args:
        Tensor : The Tensor to check.
    
    Returns:
        bool : True if any element of the given Tensor's data is NaN,
            otherwise False.
    '''
    if tensor.device == 'cuda': return cuda_has_nan(tensor)
    else: return np.any(np.isnan(tensor.data))

def has_inf(tensor: Tensor) -> bool:
    '''Checks whether any element of a given Tensor's data is infinite.
    
    Args:
        Tensor : The Tensor to check.
    
    Returns:
        bool : True if any element of the given Tensor's data is infinite,
            otherwise False.
    '''
    if tensor.device == 'cuda': return cuda_has_inf(tensor)
    else: return np.any(np.isinf(tensor.data))


