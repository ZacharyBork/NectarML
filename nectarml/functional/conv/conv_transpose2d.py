from typing import Literal

from nectarml import cpu, cuda
from nectarml.tensor import Tensor
from nectarml.typing import float32
from nectarml.amp.precision import run_cast_float16

### CPU ###

def _conv_transpose2d_cpu(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    B: int, C_in: int, H_in: int, W_in: int,
    H_out: int, W_out: int, C_out: int, KH: int, KW: int,
    stride_h: int, stride_w: int,
    padding_h: int, padding_w: int,
    output_padding_h: int, output_padding_w: int,
    dilation_h: int, dilation_w: int,
    groups: int
) -> Tensor:
    input_requires_grad  = input.requires_grad
    weight_requires_grad = weight.requires_grad
    bias_requires_grad   = bias.requires_grad if bias is not None else False

    _requires_grad = input_requires_grad or weight_requires_grad
    _children = [input, weight]
    if bias is not None:
        _children.append(bias)
        _requires_grad = _requires_grad or bias_requires_grad

    out_data = cpu.conv.conv_transpose2d(
        input.data, weight.data,
        bias.data if bias is not None else None,
        B, C_in, H_in, W_in, C_out, KH, KW,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w,
        output_padding_h, output_padding_w, groups)
    out = Tensor._new(out_data, (B, C_out, H_out, W_out), input.dtype, 
        input.device, requires_grad=_requires_grad, _children=tuple(_children))

    def _backward() -> None:
        out_grad = out.grad.contiguous()
        B, C_in, H_in, W_in = input.shape
        C_in_w, C_out, KH, KW = weight.shape
        H_out, W_out = out.shape[2], out.shape[3]

        if input_requires_grad:
            grad_input = cpu.conv.conv_transpose2d_backward_input(
                out_grad.data, weight.data,
                B, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out,
                stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w, groups)
            input.grad += Tensor._new(
                grad_input, input.shape, float32, 'cpu')

        if weight_requires_grad:
            grad_weight = cpu.conv.conv_transpose2d_backward_weight(
                out_grad.data, input.data,
                B, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out,
                stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w, groups)
            weight.grad += Tensor._new(
                grad_weight, weight.shape, float32, 'cpu')

        if bias is not None and bias_requires_grad:
            bias.grad += Tensor._new(
                out_grad.data.sum(axis=(0, 2, 3)),
                bias.shape, float32, 'cpu')

    out._backward = _backward
    return out

### CUDA ###

def _conv_transpose2d_cuda(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    B: int, C_in: int, H_in: int, W_in: int,
    H_out: int, W_out: int, C_out: int, KH: int, KW: int,
    stride_h: int, stride_w: int,
    padding_h: int, padding_w: int,
    output_padding_h: int, output_padding_w: int,
    dilation_h: int, dilation_w: int,
    groups: int
) -> Tensor:
    input_requires_grad  = input.requires_grad
    weight_requires_grad = weight.requires_grad
    bias_requires_grad   = bias.requires_grad if bias is not None else False

    _requires_grad = input_requires_grad or weight_requires_grad
    _children = [input, weight]
    if bias is not None:
        _children.append(bias)
        _requires_grad = _requires_grad or bias_requires_grad

    out_data = run_cast_float16(
        cuda.conv.conv_transpose2d,
        input, weight, bias,
        B, C_in, H_in, W_in, C_out, KH, KW,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w,
        output_padding_h, output_padding_w, groups)
    out = Tensor._new(out_data, (B, C_out, H_out, W_out), input.dtype, 
        input.device, requires_grad=_requires_grad, _children=tuple(_children))

    def _backward() -> None:
        out_grad = out.grad.contiguous()
        B, C_in, H_in, W_in = input.shape
        C_in_w, C_out, KH, KW = weight.shape
        H_out, W_out = out.shape[2], out.shape[3]

        if input_requires_grad:
            grad_input_ptr = cuda.conv.conv_transpose2d_backward_input(
                out_grad, weight,
                B, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out,
                stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w, groups)
            input.grad += Tensor._new(
                grad_input_ptr, input.shape, float32, input.device)

        if weight_requires_grad:
            grad_weight_ptr = cuda.conv.conv_transpose2d_backward_weight(
                out_grad, input,
                B, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out,
                stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w)
            weight.grad += Tensor._new(
                grad_weight_ptr, weight.shape, float32, weight.device)

        if bias is not None and bias_requires_grad:
            bias.grad += out_grad.sum(dim=(0, 2, 3))

    out._backward = _backward
    return out

### WRAPPER ###

def conv_transpose2d(
    input:          Tensor,
    weight:         Tensor,
    bias:           Tensor | None = None,
    stride:         int | tuple[int, int] = 1,
    padding:        int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation:       int | tuple[int, int] = 1,
    groups:         int = 1,
    padding_mode:   Literal['zeros'] = 'zeros'
) -> Tensor:
    B, C_in, H_in, W_in = input.shape
    C_in_w, C_out, KH, KW = weight.shape
    
    if padding_mode != 'zeros':
        raise ValueError(
            'Only padding_mode="zeros" is supported for conv_transpose2d.')
    stride_h, stride_w     = (stride, stride) \
        if isinstance(stride, int) else stride
    dilation_h, dilation_w = (dilation, dilation) \
        if isinstance(dilation, int) else dilation
    padding_h, padding_w   = (padding, padding) \
        if isinstance(padding, int) else padding
    op_h, op_w             = (output_padding, output_padding) \
        if isinstance(output_padding, int) else output_padding

    assert C_in % groups == 0, \
        f'Input channels [{C_in}] must be divisible by groups [{groups}].'
    assert C_in_w == C_in, \
        f'Weight input channels [{C_in_w}] must match input channels [{C_in}].'
    assert C_out % groups == 0, \
        f'Output channels [{C_out}] must be divisible by groups [{groups}].'
    assert 0 <= op_h < stride_h, \
        f'output_padding_h [{op_h}] must be in [0, stride_h-1].'
    assert 0 <= op_w < stride_w, \
        f'output_padding_w [{op_w}] must be in [0, stride_w-1].'

    H_out = (H_in-1)*stride_h - 2*padding_h + dilation_h*(KH-1) + 1 + op_h
    W_out = (W_in-1)*stride_w - 2*padding_w + dilation_w*(KW-1) + 1 + op_w

    if input.device == 'cuda':
        return _conv_transpose2d_cuda(
            input, weight, bias,
            B, C_in, H_in, W_in, H_out, W_out, C_out, KH, KW,
            stride_h, stride_w,
            padding_h, padding_w,
            op_h, op_w,
            dilation_h, dilation_w, groups)
    else:
        return _conv_transpose2d_cpu(
            input, weight, bias,
            B, C_in, H_in, W_in, H_out, W_out, C_out, KH, KW,
            stride_h, stride_w,
            padding_h, padding_w,
            op_h, op_w,
            dilation_h, dilation_w, groups)

