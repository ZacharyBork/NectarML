from typing import Literal

from nectarml.tensor import Tensor
from nectarml._core import padding
from nectarml.functional.common import _eval_core_function

def pad(
    input: Tensor, 
    pad: tuple, 
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    value: float = 0.0
) -> Tensor:
    return _eval_core_function(
        input, lambda x : padding.pad(x, pad, mode, value))
    

