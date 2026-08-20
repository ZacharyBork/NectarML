from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml
from   nectarml.typing       import Size, float32
from   nectarml.amp.autocast import autocast_state

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
        input.dtype.cuda)
    
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
        input.dtype.cuda)

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
        out_grad.dtype.cuda)
    
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
        out_grad.dtype.cuda)
    
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
        input.dtype.cuda)
    
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
        input.dtype.cuda)
    
### 2-Dimensional ###

def conv2d(
    input:       Tensor,
    weight:      Tensor,
    bias:        Tensor | None,
    input_shape: tuple[int, int, int, int],
    C_out: int, KH: int, KW: int, H_out: int, W_out: int,
    stride:   tuple[int, int],
    padding:  tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> Tensor:
    assert groups == 1, \
        'Grouped convolution is not currently supported for CUDA Tensors.'

    B, C_in, H, W        = input_shape
    input_requires_grad  = input.requires_grad
    weight_requires_grad = weight.requires_grad
    bias_requires_grad   = bias.requires_grad if bias is not None else False
    
    _requires_grad = input_requires_grad or weight_requires_grad
    _children      = [input, weight]
    if bias is not None: 
        _children.append(bias)
        _requires_grad = _requires_grad or bias_requires_grad
        
    output_size    = Size([B, C_out, H_out, W_out])
    state          = autocast_state()
    half_precision = state.enabled and state.context == 'cuda'
    
    out_data = _nectarml.tensor.conv.conv2d(
        input._data_ptr,  input.size, 
        weight._data_ptr, weight.size,
        bias._data_ptr if bias is not None else 0,
        bias.size      if bias is not None else 0,
        output_size.numel(), B, C_in, H, W, C_out, KH, KW,
        stride[0], stride[1], padding[0], padding[1],
        dilation[0], dilation[1], groups, input.dtype.cuda,
        half_precision)
    
    out = input._new(out_data, output_size, input.dtype, 
        input.device, requires_grad=_requires_grad, _children=tuple(_children))
    
    def _backward() -> None:
        out_grad         = out.grad.contiguous()
        B, C_in, H, W    = input.shape
        C_out, _, KH, KW = weight.shape

        if input_requires_grad:
            grad_input_ptr = _nectarml.tensor.conv.conv2d_backward_input(
                out_grad._data_ptr, weight._data_ptr,
                B, C_in, H, W, C_out, KH, KW, H_out, W_out,
                stride[0], stride[1], padding[0], padding[1],
                dilation[0], dilation[1], groups, float32.cuda)
            input.grad += input._new(
                grad_input_ptr, input.shape, float32, input.device)
        
        if weight_requires_grad:
            grad_weight_ptr = \
                _nectarml.tensor.conv.conv2d_backward_weight(
                    out_grad._data_ptr, input._data_ptr,
                    B, C_in, H, W, C_out, KH, KW, H_out, W_out,
                    stride[0], stride[1], padding[0], padding[1],
                    dilation[0], dilation[1], float32.cuda) 
            weight.grad += input._new(
                grad_weight_ptr, weight.shape, float32, weight.device)
        
        if bias is not None and bias_requires_grad:
            bias.grad += out_grad.sum(dim=(0, 2, 3))
            
    out._backward = _backward
    return out

def conv_transpose2d(
    input:       Tensor,
    weight:      Tensor,
    bias:        Tensor | None,
    input_shape: tuple[int, int, int, int],
    C_out: int, KH: int, KW: int, H_out: int, W_out: int, 
    stride:         tuple[int, int],
    padding:        tuple[int, int],
    output_padding: tuple[int, int],
    dilation:       tuple[int, int],
    groups:         int
) -> Tensor:
    assert groups == 1, \
        'Grouped convolution is not currently supported for CUDA Tensors.'

    B, C_in, H, W        = input_shape
    input_requires_grad  = input.requires_grad
    weight_requires_grad = weight.requires_grad
    bias_requires_grad   = bias.requires_grad if bias is not None else False

    _requires_grad = input_requires_grad or weight_requires_grad
    _children = [input, weight]
    if bias is not None:
        _children.append(bias)
        _requires_grad = _requires_grad or bias_requires_grad

    output_size    = Size([B, C_out, H_out, W_out])
    state          = autocast_state()
    half_precision = state.enabled and state.context == 'cuda'
    
    out_data = _nectarml.tensor.conv.conv_transpose2d(
        input._data_ptr,  input.size, 
        weight._data_ptr, weight.size,
        bias._data_ptr if bias is not None else 0,
        bias.size      if bias is not None else 0,
        output_size.numel(), B, C_in, H, W, C_out, KH, KW,
        stride[0], stride[1], padding[0], padding[1],
        dilation[0], dilation[1], 
        output_padding[0], output_padding[1],
        groups, input.dtype.cuda, half_precision)
    
    out = input._new(out_data, (B, C_out, H_out, W_out), input.dtype, 
        input.device, requires_grad=_requires_grad, _children=tuple(_children))

    def _backward() -> None:
        out_grad         = out.grad.contiguous()
        B, C_in, H, W    = input.shape
        _, C_out, KH, KW = weight.shape
        H_out, W_out     = out.shape[2], out.shape[3]

        if input_requires_grad:
            grad_input_ptr = \
                _nectarml.tensor.conv.conv_transpose2d_backward_input(
                    out_grad._data_ptr, weight._data_ptr,
                    B, C_in, H, W, C_out, KH, KW, H_out, W_out,
                    stride[0], stride[1], padding[0], padding[1],
                    dilation[0], dilation[1], groups, float32.cuda)
            input.grad += input._new(
                grad_input_ptr, input.shape, float32, input.device)

        if weight_requires_grad:
            grad_weight_ptr = \
                _nectarml.tensor.conv.conv_transpose2d_backward_weight(
                    out_grad._data_ptr, input._data_ptr,
                    B, C_in, H, W, C_out, KH, KW, H_out, W_out,
                    stride[0], stride[1], padding[0], padding[1],
                    dilation[0], dilation[1], input.dtype.cuda)
            weight.grad += input._new(
                grad_weight_ptr, weight.shape, float32, weight.device)

        if bias is not None and bias_requires_grad:
            bias.grad += out_grad.sum(dim=(0, 2, 3))

    out._backward = _backward
    return out

### 3-Dimensional ###

