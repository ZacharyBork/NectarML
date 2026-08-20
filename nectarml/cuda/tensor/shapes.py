from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import math

import _nectarml

### TENSOR RESHAPING ###

def permute(input: Tensor, dims: tuple[int, ...] | None) -> int:
    return _nectarml.tensor.shapes.permute(
        input._data_ptr, input.shape, list(dims), input.dtype.cuda)

def expand(input: Tensor, shape: tuple[int, ...]) -> int:
    return _nectarml.tensor.shapes.expand(
        input._data_ptr, input.shape, list(shape), input.dtype.cuda)

def flip(input: Tensor, dim: int) -> int:
    dim      = dim if dim >= 0 else input.ndim + dim
    outer    = int(math.prod(input.shape[:dim]))
    dim_size = input.shape[dim]
    inner    = int(math.prod(input.shape[dim+1:]))
    total    = input.numel()
    return _nectarml.tensor.shapes.flip(
        input._data_ptr, total, dim_size,
        outer, inner,
        input.dtype.cuda)
    
### IM2COL / COL2IM WRAPPERS ###
    
def im2col_1d(
    input: Tensor,
    size: int,
    step: int = 1
) -> int:
    B, C, L = input.shape
    return _nectarml.tensor.conv.im2col_1d(
        input._data_ptr,
        B, C, L, 1, size, step, 0, 1, 1,
        input.dtype.cuda)

def col2im_1d(
    grad: Tensor,
    B: int, C: int, L: int,
    size: int,
    L_out: int,
    step: int = 1
) -> int:
    return _nectarml.tensor.conv.col2im_1d(
        grad._data_ptr,
        B, C, L, size, L_out, step, 0, 1, 1,
        input.dtype.cuda)

def im2col_2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    step: int | tuple[int, int] = 1
) -> int:
    B, C, H, W = input.shape
    KH, KW = (kernel_size, kernel_size) \
        if isinstance(kernel_size, int) else kernel_size
    SH, SW = (step, step) \
        if isinstance(step, int) else step

    return _nectarml.tensor.conv.im2col_2d(
        input._data_ptr,
        B, C, H, W, 1,
        KH, KW, SH, SW, 0, 0, 1, 1,
        input.dtype.cuda)

def col2im_2d(
    input: Tensor,
    B: int, C: int, H: int, W: int,
    kernel_size: int | tuple[int, int],
    H_out: int, W_out: int,
    step: int | tuple[int, int] = 1
) -> int:
    KH, KW = (kernel_size, kernel_size) \
        if isinstance(kernel_size, int) else kernel_size
    SH, SW = (step, step) \
        if isinstance(step, int) else step

    return _nectarml.tensor.conv.col2im_2d(
        input._data_ptr,
        B, C, H, W, KH, KW,
        H_out, W_out, SH, SW, 
        0, 0, 1, 1, input.dtype.cuda)


