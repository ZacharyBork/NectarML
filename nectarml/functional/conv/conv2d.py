from typing import Literal

from nectarml.tensor import Tensor
from nectarml import cpu, cuda
from nectarml.functional.padding import pad
from nectarml.amp.precision import run_cast_float16

### CPU ###

def _conv2d_cpu(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    B: int, C_in: int, H: int, W: int,
    C_out: int, KH: int, KW: int,
    H_out: int, W_out: int,
    stride_h: int, stride_w: int,
    padding_h: int, padding_w: int,
    dilation_h: int, dilation_w: int,
    groups: int,
) -> Tensor:
    input_requires_grad = input.requires_grad
    weight_requires_grad = weight.requires_grad
    bias_requires_grad = bias.requires_grad if bias is not None else False
    
    _requires_grad = input.requires_grad or weight.requires_grad
    _children = [input, weight]
    if bias is not None: 
        _children.append(bias)
        _requires_grad = _requires_grad or bias.requires_grad

    out_data, input_padded = cpu.conv.conv2d(
        input.data, weight.data, bias.data if bias is not None else None,
        B, C_in, H, W, C_out, KH, KW, H_out, W_out,
        stride_h, stride_w, padding_h, padding_w, 
        dilation_h, dilation_w, groups)
    out = Tensor._new(
        out_data, (B, C_out, H_out, W_out), input.dtype, input.device,
        requires_grad=_requires_grad, _children=tuple(_children))
    
    def _backward() -> None:
        out_grad = out.grad.contiguous()

        if input_requires_grad:
            grad_input = cpu.conv.conv2d_backward_input(
                out_grad.data, input_padded, weight.data,
                KH, KW, H_out, W_out,
                stride_h, stride_w, padding_h, padding_w, 
                dilation_h, dilation_w, groups)
            input.grad += Tensor._new(
                grad_input, input.shape, input.dtype, 'cpu')
        
        if weight_requires_grad:
            grad_weight = cpu.conv.conv2d_backward_weight(
                out_grad.data, input_padded, weight.data,
                KH, KW, H_out, W_out,
                stride_h, stride_w, dilation_h, dilation_w, groups)
            weight.grad += Tensor._new(
                grad_weight, weight.shape, weight.dtype, 'cpu')
        
        if bias is not None and bias_requires_grad:
            bias.grad += Tensor._new(
                out_grad.data.sum(axis=(0, 2, 3)), 
                bias.shape, bias.dtype, 'cpu')
                
    out._backward = _backward
    return out

### CUDA ###

def _conv2d_cuda(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    B: int, C_in: int, H: int, W: int,
    C_out: int, KH: int, KW: int,
    H_out: int, W_out: int,
    stride_h: int, stride_w: int,
    padding_h: int, padding_w: int,
    dilation_h: int, dilation_w: int,
    groups: int,
) -> Tensor:
    input_requires_grad  = input.requires_grad
    weight_requires_grad = weight.requires_grad
    bias_requires_grad   = bias.requires_grad if bias is not None else False
    
    _requires_grad = input.requires_grad or weight.requires_grad
    _children = [input, weight]
    if bias is not None: 
        _children.append(bias)
        _requires_grad = _requires_grad or bias.requires_grad
        
    out_data = run_cast_float16(
        cuda.conv.conv2d,
        input, weight, bias, 
        B, C_in, H, W, C_out, KH, KW,
        stride_h, stride_w, padding_h, padding_w, 
        dilation_h, dilation_w, groups)
    out = Tensor._new(out_data, (B, C_out, H_out, W_out), input.dtype, input.device,
        requires_grad=_requires_grad, _children=tuple(_children))
    
    def _backward() -> None:
        out_grad = out.grad.contiguous()
        
        B, C_in, H, W = input.shape
        C_out, _, KH, KW = weight.shape
    
        if input_requires_grad:
            grad_input_ptr = cuda.conv.conv2d_backward_input(
                out_grad, weight, 
                B, C_in, H, W, C_out, KH, KW, H_out, W_out,
                stride_h, stride_w, padding_h, padding_w, 
                dilation_h, dilation_w, groups)
            input.grad += Tensor._new(
                grad_input_ptr, input.shape, input.dtype, input.device)
        
        if weight_requires_grad:
            grad_weight_ptr = cuda.conv.conv2d_backward_weight(
                out_grad, input, 
                B, C_in, H, W, C_out, KH, KW, H_out, W_out,
                stride_h, stride_w, padding_h, padding_w, 
                dilation_h, dilation_w)
            weight.grad += Tensor._new(
                grad_weight_ptr, weight.shape, weight.dtype, weight.device)
        
        if bias is not None and bias_requires_grad:
            bias.grad += out_grad.sum(dim=(0, 2, 3))
            
    out._backward = _backward
    return out

### WRAPPER ###

def conv2d(
    input:        Tensor,
    weight:       Tensor,
    bias:         Tensor | None = None,
    stride:       int | tuple[int, int] = 1,
    padding:      int | tuple[int, int] = 0,
    dilation:     int | tuple[int, int] = 1,
    groups:       int = 1,
    padding_mode: Literal[
        'zeros', 'reflect', 'replicate', 'circular'
    ] = 'zeros'
) -> Tensor:
    B, C_in, H_in, W_in = input.shape
    C_out, _, KH, KW = weight.shape

    assert C_in % groups == 0, \
        f'Input channels [{C_in}] must be divisible by groups [{groups}].'
    assert C_out % groups == 0, \
        f'Output channels [{C_out}] must be divisible by groups [{groups}].'

    stride_h, stride_w     = (stride, stride) \
        if isinstance(stride, int) else stride
    dilation_h, dilation_w = (dilation, dilation) \
        if isinstance(dilation, int) else dilation
    padding_h, padding_w   = (padding, padding) \
        if isinstance(padding, int) else padding

    if padding_mode != 'zeros' and (padding_h > 0 or padding_w > 0):
        input = pad(
            input, (padding_w, padding_w, padding_h, padding_h),
            mode=padding_mode)
        B, C_in, H_in, W_in = input.shape
        padding_h = padding_w = 0

    H_out = (H_in + 2*padding_h - dilation_h*(KH-1) - 1) // stride_h + 1
    W_out = (W_in + 2*padding_w - dilation_w*(KW-1) - 1) // stride_w + 1

    if input.device == 'cuda':
        return _conv2d_cuda(
            input, weight, bias,
            B, C_in, H_in, W_in,
            C_out, KH, KW, H_out, W_out,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w, groups)
    else:
        return _conv2d_cpu(
            input, weight, bias,
            B, C_in, H_in, W_in,
            C_out, KH, KW, H_out, W_out,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w, groups)
