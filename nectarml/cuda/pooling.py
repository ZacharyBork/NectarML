from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml

### AVERAGE POOL ###

def avg_pool1d_forward(
    input: Tensor,
    B: int, C: int, L: int, L_out: int,
    K: int, S: int, P: int,
    count_include_pad: bool
) -> int:
    return _nectarml.tensor.pooling.avg_pool1d_forward(
        input._data_ptr,
        B, C, L, L_out, K, S, P,
        count_include_pad,
        input.dtype.cuda)

def avg_pool1d_backward(
    out_grad: Tensor,
    B: int, C: int, L: int, L_out: int,
    K: int, S: int, P: int,
    count_include_pad: bool
) -> int:
    return _nectarml.tensor.pooling.avg_pool1d_backward(
        out_grad._data_ptr,
        B, C, L, L_out, K, S, P,
        count_include_pad,
        out_grad.dtype.cuda)

def avg_pool2d_forward(
    input: Tensor,
    B: int, C: int, H: int, W: int,
    H_out: int, W_out: int,
    KH: int, KW: int,
    SH: int, SW: int,
    PH: int, PW: int,
    count_include_pad: bool
) -> int:
    return _nectarml.tensor.pooling.avg_pool2d_forward(
        input._data_ptr,
        B, C, H, W, H_out, W_out,
        KH, KW, SH, SW, PH, PW,
        count_include_pad,
        input.dtype.cuda)

def avg_pool2d_backward(
    out_grad: Tensor,
    B: int, C: int, H: int, W: int,
    H_out: int, W_out: int,
    KH: int, KW: int,
    SH: int, SW: int,
    PH: int, PW: int,
    count_include_pad: bool
) -> int:
    return _nectarml.tensor.pooling.avg_pool2d_backward(
        out_grad._data_ptr,
        B, C, H, W, H_out, W_out,
        KH, KW, SH, SW, PH, PW,
        count_include_pad,
        out_grad.dtype.cuda)

def avg_pool3d_forward(
    input: Tensor,
    B: int, C: int, D: int, H: int, W: int,
    D_out: int, H_out: int, W_out: int,
    KD: int, KH: int, KW: int,
    SD: int, SH: int, SW: int,
    PD: int, PH: int, PW: int,
    count_include_pad: bool
) -> int:
    return _nectarml.tensor.pooling.avg_pool3d_forward(
        input._data_ptr,
        B, C, D, H, W, D_out, H_out, W_out,
        KD, KH, KW, SD, SH, SW, PD, PH, PW,
        count_include_pad,
        input.dtype.cuda)

def avg_pool3d_backward(
    out_grad: Tensor,
    B: int, C: int, D: int, H: int, W: int,
    D_out: int, H_out: int, W_out: int,
    KD: int, KH: int, KW: int,
    SD: int, SH: int, SW: int,
    PD: int, PH: int, PW: int,
    count_include_pad: bool
) -> int:
    return _nectarml.tensor.pooling.avg_pool3d_backward(
        out_grad._data_ptr,
        B, C, D, H, W, D_out, H_out, W_out,
        KD, KH, KW, SD, SH, SW, PD, PH, PW,
        count_include_pad,
        out_grad.dtype.cuda)

### MAX POOL ###

def max_pool1d_forward(
    input: Tensor,
    B: int, C: int, L: int, L_out: int,
    K: int, S: int, P: int, D: int
) -> tuple[int, int]:
    return _nectarml.tensor.pooling.max_pool1d_forward(
        input._data_ptr,
        B, C, L, L_out, K, S, P, D,
        input.dtype.cuda)

def max_pool1d_backward(
    out_grad: Tensor,
    indices_ptr: int,
    B: int, C: int, L: int, L_out: int
) -> int:
    return _nectarml.tensor.pooling.max_pool1d_backward(
        out_grad._data_ptr, indices_ptr,
        B, C, L, L_out,
        out_grad.dtype.cuda)

def max_pool2d_forward(
    input: Tensor,
    B: int, C: int, H: int, W: int,
    H_out: int, W_out: int,
    KH: int, KW: int,
    SH: int, SW: int,
    PH: int, PW: int, D: int
) -> tuple[int, int]:
    return _nectarml.tensor.pooling.max_pool2d_forward(
        input._data_ptr,
        B, C, H, W, H_out, W_out,
        KH, KW, SH, SW, PH, PW, D,
        input.dtype.cuda)

def max_pool2d_backward(
    out_grad: Tensor,
    indices_ptr: int,
    B: int, C: int, H: int, W: int,
    H_out: int, W_out: int
) -> int:
    return _nectarml.tensor.pooling.max_pool2d_backward(
        out_grad._data_ptr, indices_ptr,
        B, C, H, W, H_out, W_out,
        out_grad.dtype.cuda)

def max_pool3d_forward(
    input: Tensor,
    B: int, C: int, D: int, H: int, W: int,
    D_out: int, H_out: int, W_out: int,
    KD: int, KH: int, KW: int,
    SD: int, SH: int, SW: int,
    PD: int, PH: int, PW: int, Dil: int
) -> tuple[int, int]:
    return _nectarml.tensor.pooling.max_pool3d_forward(
        input._data_ptr,
        B, C, D, H, W, D_out, H_out, W_out,
        KD, KH, KW, SD, SH, SW, PD, PH, PW, Dil,
        input.dtype.cuda)

def max_pool3d_backward(
    out_grad: Tensor,
    indices_ptr: int,
    B: int, C: int, D: int, H: int, W: int,
    D_out: int, H_out: int, W_out: int
) -> int:
    return _nectarml.tensor.pooling.max_pool3d_backward(
        out_grad._data_ptr, indices_ptr,
        B, C, D, H, W, D_out, H_out, W_out,
        out_grad.dtype.cuda)


