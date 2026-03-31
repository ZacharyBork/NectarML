import numpy as np

from nectarml.tensor import Tensor
from nectarml.cuda.utils import (
    is_inf    as cuda_is_inf,
    is_finite as cuda_is_finite,
    is_nan    as cuda_is_nan,
    has_nan   as cuda_has_nan,
    has_inf   as cuda_has_inf
)

def is_inf(tensor: Tensor) -> bool:
    if tensor.device == 'cuda': return cuda_is_inf(tensor)
    else: return np.all(np.isinf(tensor.data))

def is_finite(tensor: Tensor) -> bool:
    if tensor.device == 'cuda': return cuda_is_finite(tensor)
    else: return np.all(np.isfinite(tensor.data))

def is_nan(tensor: Tensor) -> bool:
    if tensor.device == 'cuda': return cuda_is_nan(tensor)
    else: return np.all(np.isnan(tensor.data))

def has_nan(tensor: Tensor) -> bool:
    if tensor.device == 'cuda': return cuda_has_nan(tensor)
    else: return np.any(np.isnan(tensor.data))

def has_inf(tensor: Tensor) -> bool:
    if tensor.device == 'cuda': return cuda_has_inf(tensor)
    else: return np.any(np.isinf(tensor.data))


