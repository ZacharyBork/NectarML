from nectarml.core import Tensor

def is_inf(tensor: Tensor) -> bool:
    '''Checks whether every element of a given Tensor's data is infinite.
    
    Args:
        Tensor : The Tensor to check.
    
    Returns:
        bool : True if all elements of the given Tensor's data are infinite,
            otherwise False.
    '''
    return tensor.is_inf()

def is_finite(tensor: Tensor) -> bool:
    '''Checks whether every element of a given Tensor's data is finite.
    
    Args:
        Tensor : The Tensor to check.
    
    Returns:
        bool : True if all elements of the given Tensor's data are finite,
            otherwise False.
    '''
    return tensor.is_finite()

def is_nan(tensor: Tensor) -> bool:
    '''Checks whether every element of a given Tensor's data is not a number.
    
    Args:
        Tensor : The Tensor to check.
    
    Returns:
        bool : True if all elements of the given Tensor's data are NaN,
            otherwise False.
    '''
    return tensor.is_nan()

def has_nan(tensor: Tensor) -> bool:
    '''Checks whether any element of a given Tensor's data is not a number.
    
    Args:
        Tensor : The Tensor to check.
    
    Returns:
        bool : True if any element of the given Tensor's data is NaN,
            otherwise False.
    '''
    return tensor.has_nan()

def has_inf(tensor: Tensor) -> bool:
    '''Checks whether any element of a given Tensor's data is infinite.
    
    Args:
        Tensor : The Tensor to check.
    
    Returns:
        bool : True if any element of the given Tensor's data is infinite,
            otherwise False.
    '''
    return tensor.has_inf()


