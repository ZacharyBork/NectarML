from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml
from nectarml.cuda.utils import map_dtype

### 1-Dimensional ###

def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    B: int, C_in: int, L: int,
    C_out: int, K: int,
    stride: int, 
    padding: int, 
    dilation: int,
    groups: int
) -> int:
    assert groups == 1, \
        'Grouped convolution is not currently supported for CUDA Tensors.'

    return _nectarml.tensor.conv.conv1d(
        input._data_ptr, weight._data_ptr,
        bias._data_ptr if bias is not None else 0,
        B, C_in, L, C_out, K,
        stride, padding, dilation, groups,
        map_dtype(input.dtype))
    
def conv_transpose1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    B: int, C_in: int, L: int,
    C_out: int, K: int,
    stride: int, 
    padding: int, 
    output_padding: int,
    dilation: int,
    groups: int
) -> int:
    assert groups == 1, \
        'Grouped convolution is not currently supported for CUDA Tensors.'

    return _nectarml.tensor.conv.conv_transpose1d(
        input._data_ptr, weight._data_ptr,
        bias._data_ptr if bias is not None else 0,
        B, C_in, L, C_out, K,
        stride, padding, dilation, output_padding, groups,
        map_dtype(input.dtype))

def conv1d_backward_input(
    out_grad: Tensor,
    weight: Tensor,
    B: int, C_in: int, L: int,
    C_out: int, K: int, L_out: int,
    stride: int, padding: int, dilation: int, groups: int
) -> int:
    return _nectarml.tensor.conv.conv1d_backward_input(
        out_grad._data_ptr, weight._data_ptr,
        B, C_in, L, C_out, K, L_out,
        stride, padding, dilation, groups,
        map_dtype(out_grad.dtype))
    
def conv_transpose1d_backward_input(
    out_grad: Tensor,
    weight: Tensor,
    B: int, C_in: int, L: int,
    C_out: int, K: int, L_out: int,
    stride: int, padding: int, dilation: int, groups: int
) -> int:
    return _nectarml.tensor.conv.conv_transpose1d_backward_input(
        out_grad._data_ptr, weight._data_ptr,
        B, C_in, L, C_out, K, L_out,
        stride, padding, dilation, groups,
        map_dtype(out_grad.dtype))
    
def conv1d_backward_weight(
    out_grad: Tensor,
    input: Tensor,
    B: int, C_in: int, L: int,
    C_out: int, K: int, L_out: int,
    stride: int, padding: int, dilation: int
) -> int:
    return _nectarml.tensor.conv.conv1d_backward_weight(
        out_grad._data_ptr, input._data_ptr,
        B, C_in, L, C_out, K, L_out,
        stride, padding, dilation,
        map_dtype(out_grad.dtype))
    
def conv_transpose1d_backward_weight(
    out_grad: Tensor,
    input: Tensor,
    B: int, C_in: int, L: int,
    C_out: int, K: int, L_out: int,
    stride: int, padding: int, dilation: int
) -> int:
    return _nectarml.tensor.conv.conv_transpose1d_backward_weight(
        out_grad._data_ptr, input._data_ptr,
        B, C_in, L, C_out, K, L_out,
        stride, padding, dilation,
        map_dtype(out_grad.dtype))
    
### 2-Dimensional ###

def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    B: int, C_in: int, H: int, W: int,
    C_out: int, KH: int, KW: int,
    stride_h: int, stride_w: int, 
    padding_h: int, padding_w: int, 
    dilation_h: int, dilation_w: int,
    groups: int
) -> int:
    assert groups == 1, \
        'Grouped convolution is not currently supported for CUDA Tensors.'
        
    return _nectarml.tensor.conv.conv2d(
        input._data_ptr, weight._data_ptr,
        bias._data_ptr if bias is not None else 0,
        B, C_in, H, W, C_out, KH, KW,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, groups, map_dtype(input.dtype))
   
def conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    B: int, C_in: int, H: int, W: int,
    C_out: int, KH: int, KW: int,
    stride_h: int, stride_w: int, 
    padding_h: int, padding_w: int, 
    dilation_h: int, dilation_w: int,
    output_padding_h: int, output_padding_w: int,
    groups: int
) -> int:
    assert groups == 1, \
        'Grouped convolution is not currently supported for CUDA Tensors.'
        
    return _nectarml.tensor.conv.conv_transpose2d(
        input._data_ptr, weight._data_ptr,
        bias._data_ptr if bias is not None else 0,
        B, C_in, H, W, C_out, KH, KW,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, 
        output_padding_h, output_padding_w,
        groups, map_dtype(input.dtype))
    
def conv2d_backward_input(
    out_grad: Tensor,
    weight: Tensor,
    B: int, C_in: int, H: int, W: int,
    C_out: int, KH: int, KW: int,
    H_out: int, W_out: int,
    stride_h: int, stride_w: int, 
    padding_h: int, padding_w: int, 
    dilation_h: int, dilation_w: int,
    groups: int
) -> int:
    return _nectarml.tensor.conv.conv2d_backward_input(
        out_grad._data_ptr, weight._data_ptr,
        B, C_in, H, W, C_out, KH, KW, H_out, W_out,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, groups, map_dtype(out_grad.dtype))

def conv_transpose2d_backward_input(
    out_grad: Tensor,
    weight: Tensor,
    B: int, C_in: int, H: int, W: int,
    C_out: int, KH: int, KW: int,
    H_out: int, W_out: int,
    stride_h: int, stride_w: int, 
    padding_h: int, padding_w: int, 
    dilation_h: int, dilation_w: int,
    groups: int
) -> int:
    return _nectarml.tensor.conv.conv_transpose2d_backward_input(
        out_grad._data_ptr, weight._data_ptr,
        B, C_in, H, W, C_out, KH, KW, H_out, W_out,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, groups, map_dtype(out_grad.dtype))
    
def conv2d_backward_weight(
    out_grad: Tensor,
    input: Tensor,
    B: int, C_in: int, H: int, W: int,
    C_out: int, KH: int, KW: int,
    H_out: int, W_out: int,
    stride_h: int, stride_w: int, 
    padding_h: int, padding_w: int, 
    dilation_h: int, dilation_w: int
) -> int:
    return _nectarml.tensor.conv.conv2d_backward_weight(
        out_grad._data_ptr, input._data_ptr,
        B, C_in, H, W, C_out, KH, KW, H_out, W_out,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, map_dtype(out_grad.dtype))
    
def conv_transpose2d_backward_weight(
    out_grad: Tensor,
    input: Tensor,
    B: int, C_in: int, H: int, W: int,
    C_out: int, KH: int, KW: int,
    H_out: int, W_out: int,
    stride_h: int, stride_w: int, 
    padding_h: int, padding_w: int, 
    dilation_h: int, dilation_w: int
) -> int:
    return _nectarml.tensor.conv.conv_transpose2d_backward_weight(
        out_grad._data_ptr, input._data_ptr,
        B, C_in, H, W, C_out, KH, KW, H_out, W_out,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, map_dtype(out_grad.dtype))

### 3-Dimensional ###

