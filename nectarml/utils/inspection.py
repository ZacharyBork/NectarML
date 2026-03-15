import numpy as np

from nectarml.tensor import Tensor
from nectarml.cuda.utils import is_inf, is_finite, is_nan, has_inf, has_nan

def is_inf(tensor: Tensor) -> bool:
    if tensor.device == 'cuda': return is_inf(tensor)
    else: return all(np.isinf(tensor.data))

def is_finite(tensor: Tensor) -> bool:
    if tensor.device == 'cuda': return is_finite(tensor)
    else: return all(np.isfinite(tensor.data))

def is_nan(tensor: Tensor) -> bool:
    if tensor.device == 'cuda': return is_nan(tensor)
    else: return all(np.isnan(tensor.data))

def has_nan(tensor: Tensor) -> bool:
    if tensor.device == 'cuda': return has_nan(tensor)
    else: return any(np.isnan(tensor.data))

def has_inf(tensor: Tensor) -> bool:
    if tensor.device == 'cuda': return has_inf(tensor)
    else: return any(np.isinf(tensor.data))


