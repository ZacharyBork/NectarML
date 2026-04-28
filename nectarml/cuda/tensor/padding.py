from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import math
import builtins
from   typing import Literal

import _nectarml
from   nectarml import typing

def pad(
    input:      Tensor,
    pad_before: list[builtins.int],
    pad_after:  list[builtins.int],
    mode:       Literal[
        'constant', 'reflect', 'replicate', 'circular'
    ] = 'constant',
    value:      builtins.float = 0.0
) -> builtins.int:
    return _nectarml.tensor.padding.pad(
        input._data_ptr, list(input.shape), pad_before, pad_after,
        mode, value, input.dtype.cuda)

def pad_backward(
    grad_out:    Tensor,
    input_shape: list[int],
    pad_before:  list[int],
    pad_after:   list[int],
    mode:        Literal['constant', 'reflect', 'replicate', 'circular']
) -> builtins.int:
    dx_ptr = _nectarml.alloc_cuda_full(
        math.prod(input_shape), typing.float32.cuda, 0.0)
    _nectarml.tensor.padding.pad_backward(
        grad_out._data_ptr, dx_ptr, 
        list(input_shape), pad_before, pad_after, mode)
    return dx_ptr
