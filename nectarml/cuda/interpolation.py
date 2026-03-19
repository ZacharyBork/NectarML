from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml
from nectarml.cuda.utils import map_dtype

### NEAREST NEIGHBOR ###

def upsample_nearest_1d(
    input: Tensor,
    L_out: int
) -> int:
    assert input.ndim == 3, (
        'nectarml.cuda.upsample_nearest_1d input must have ndim==3.')
    return _nectarml.upsample_nearest_1d(
        input._data_ptr, input.shape[0], input.shape[1],
        input.shape[2], L_out, map_dtype(input.dtype))

def upsample_nearest_1d_backward(
    grad: Tensor,
    L_in: int, L_out: int
) -> int:
    assert grad.ndim == 3, (
        'nectarml.cuda.upsample_nearest_1d_backward grad must have ndim==3.')
    return _nectarml.upsample_nearest_1d_backward(
        grad._data_ptr, grad.shape[0], grad.shape[1],
        L_in, L_out, map_dtype(grad.dtype))

def upsample_nearest_2d(
    input: Tensor,
    H_out: int,
    W_out: int
) -> int:
    assert input.ndim == 4, (
        'nectarml.cuda.upsample_nearest_2d input must have ndim==4.')
    return _nectarml.upsample_nearest_2d(
        input._data_ptr, input.shape[0], input.shape[1],
        input.shape[2], input.shape[3], H_out, W_out, map_dtype(input.dtype))

def upsample_nearest_2d_backward(
    grad: Tensor,
    H_in: int, W_in: int,
    H_out: int, W_out: int
) -> int:
    assert grad.ndim == 4, (
        'nectarml.cuda.upsample_nearest_2d_backward grad must have ndim==4.')
    return _nectarml.upsample_nearest_2d_backward(
        grad._data_ptr, grad.shape[0], grad.shape[1],
        H_in, W_in, H_out, W_out, map_dtype(grad.dtype))

def upsample_nearest_3d(
    input: Tensor, 
    D_out: int, H_out: int, W_out: int
) -> int:
    assert input.ndim == 5, \
        'upsample_nearest_3d input must have ndim==5.'
    return _nectarml.upsample_nearest_3d(
        input._data_ptr, input.shape[0], input.shape[1],
        input.shape[2], input.shape[3], input.shape[4],
        D_out, H_out, W_out, map_dtype(input.dtype))

def upsample_nearest_3d_backward(
    grad: Tensor, 
    D_in: int, H_in: int, W_in: int,
    D_out: int, H_out: int, W_out: int
) -> int:
    assert grad.ndim == 5, \
        'upsample_nearest_3d_backward grad must have ndim==5.'
    return _nectarml.upsample_nearest_3d_backward(
        grad._data_ptr, grad.shape[0], grad.shape[1],
        D_in, H_in, W_in, D_out, H_out, W_out, map_dtype(grad.dtype))

### LINEAR ###

def upsample_linear(
    input: Tensor,
    L_out: int,
    align_corners: bool = False
) -> int:
    assert input.ndim == 3, (
        'nectarml.cuda.upsample_linear input must have ndim==3.')
    return _nectarml.upsample_linear(
        input._data_ptr, input.shape[0], input.shape[1],
        input.shape[2], L_out, align_corners, map_dtype(input.dtype))

def upsample_linear_backward(
    grad: Tensor,
    L_in: int, L_out: int,
    align_corners: bool = False
) -> int:
    assert grad.ndim == 3, (
        'nectarml.cuda.upsample_linear_backward grad must have ndim==3.')
    return _nectarml.upsample_linear_backward(
        grad._data_ptr, grad.shape[0], grad.shape[1],
        L_in, L_out, align_corners, map_dtype(grad.dtype))

def upsample_bilinear(
    input: Tensor,
    H_out: int, W_out: int,
    align_corners: bool = False
) -> int:
    assert input.ndim == 4, (
        'nectarml.cuda.upsample_bilinear input must have ndim==4.')
    return _nectarml.upsample_bilinear(
        input._data_ptr, input.shape[0], input.shape[1],
        input.shape[2], input.shape[3], H_out, W_out, 
        align_corners, map_dtype(input.dtype))

def upsample_bilinear_backward(
    grad: Tensor,
    H_in: int, W_in: int,
    H_out: int, W_out: int,
    align_corners: bool = False
) -> int:
    assert grad.ndim == 4, (
        'nectarml.cuda.upsample_bilinear_backward grad must have ndim==4.')
    return _nectarml.upsample_bilinear_backward(
        grad._data_ptr, grad.shape[0], grad.shape[1],
        H_in, W_in, H_out, W_out, align_corners, map_dtype(grad.dtype))

def upsample_trilinear(
    input: Tensor, 
    D_out: int, H_out: int, W_out: int,
    align_corners: bool = False
) -> int:
    assert input.ndim == 5, \
        'upsample_trilinear input must have ndim==5.'
    return _nectarml.upsample_trilinear(
        input._data_ptr, input.shape[0], input.shape[1],
        input.shape[2], input.shape[3], input.shape[4],
        D_out, H_out, W_out, align_corners, map_dtype(input.dtype))

def upsample_trilinear_backward(
    grad: Tensor, 
    D_in: int, H_in: int, W_in: int,
    D_out: int, H_out: int, W_out: int,
    align_corners: bool = False
) -> int:
    assert grad.ndim == 5, \
        'upsample_trilinear_backward grad must have ndim==5.'
    return _nectarml.upsample_trilinear_backward(
        grad._data_ptr, grad.shape[0], grad.shape[1],
        D_in, H_in, W_in, D_out, H_out, W_out, 
        align_corners, map_dtype(grad.dtype))

### CUBIC ###

def upsample_bicubic(
    input: Tensor,
    H_out: int, W_out: int,
    a: float = -0.75,
    align_corners: bool = False
) -> int:
    assert input.ndim == 4, (
        'nectarml.cuda.upsample_bicubic input must have ndim==4.')
    return _nectarml.upsample_bicubic(
        input._data_ptr, input.shape[0], input.shape[1],
        input.shape[2], input.shape[3], H_out, W_out, 
        a, align_corners, map_dtype(input.dtype))

def upsample_bicubic_backward(
    grad: Tensor,
    H_in: int, W_in: int,
    H_out: int, W_out: int,
    a: float = -0.75,
    align_corners: bool = False
) -> int:
    assert grad.ndim == 4, (
        'nectarml.cuda.upsample_bicubic_backward grad must have ndim==4.')
    return _nectarml.upsample_bicubic_backward(
        grad._data_ptr, grad.shape[0], grad.shape[1],
        H_in, W_in, H_out, W_out, a, align_corners, map_dtype(grad.dtype))

