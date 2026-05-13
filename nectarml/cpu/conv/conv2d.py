from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor
    
import numpy as np

from nectarml.typing import float32

### WRAPPERS ###

def conv2d(
    input:       Tensor,
    weight:      Tensor,
    bias:        Tensor | None,
    input_shape: tuple[int, int, int, int],
    C_out: int, KH: int, KW: int,
    H_out: int, W_out: int,
    stride:   tuple[int, int],
    padding:  tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> Tensor:
    B, C_in, H, W        = input_shape
    input_requires_grad  = input.requires_grad
    weight_requires_grad = weight.requires_grad
    bias_requires_grad   = bias.requires_grad if bias is not None else False
    
    _requires_grad = input.requires_grad or weight.requires_grad
    _children = [input, weight]
    if bias is not None: 
        _children.append(bias)
        _requires_grad = _requires_grad or bias.requires_grad

    out_data, input_padded = _conv2d(
        input.data, weight.data, bias.data if bias is not None else None,
        B, C_in, H, W, C_out, KH, KW, H_out, W_out,
        stride[0], stride[1], padding[0], padding[1], 
        dilation[0], dilation[1], groups)
    out = input._new(
        out_data, (B, C_out, H_out, W_out), input.dtype, input.device,
        requires_grad=_requires_grad, _children=tuple(_children))
    
    def _backward() -> None:
        out_grad = out.grad.contiguous()

        if input_requires_grad:
            grad_input = _conv2d_backward_input(
                out_grad.data, input_padded, weight.data,
                KH, KW, H_out, W_out,
                stride[0], stride[1], padding[0], padding[1], 
                dilation[0], dilation[1], groups)
            input.grad += input._new(
                grad_input, input.shape, float32, 'cpu')
        
        if weight_requires_grad:
            grad_weight = _conv2d_backward_weight(
                out_grad.data, input_padded, weight.data,
                KH, KW, H_out, W_out,
                stride[0], stride[1], dilation[0], dilation[1], groups)
            weight.grad += input._new(
                grad_weight, weight.shape, float32, 'cpu')
        
        if bias is not None and bias_requires_grad:
            bias.grad += input._new(
                out_grad.data.sum(axis=(0, 2, 3)), 
                bias.shape, float32, 'cpu')
                
    out._backward = _backward
    return out

def conv_transpose2d(
    input:       Tensor,
    weight:      Tensor,
    bias:        Tensor | None,
    input_shape: tuple[int, int, int, int],
    C_out: int, KH: int, KW: int, 
    H_out: int, W_out: int, 
    stride:         tuple[int, int],
    padding:        tuple[int, int],
    output_padding: tuple[int, int],
    dilation:       tuple[int, int],
    groups:         int
) -> Tensor:
    B, C_in, H, W        = input_shape
    input_requires_grad  = input.requires_grad
    weight_requires_grad = weight.requires_grad
    bias_requires_grad   = bias.requires_grad if bias is not None else False

    _requires_grad = input_requires_grad or weight_requires_grad
    _children = [input, weight]
    if bias is not None:
        _children.append(bias)
        _requires_grad = _requires_grad or bias_requires_grad

    out_data = _conv_transpose2d(
        input.data, weight.data,
        bias.data if bias is not None else None,
        B, C_in, H, W, C_out, KH, KW,
        stride[0], stride[1], padding[0], padding[1],
        dilation[0], dilation[1],
        output_padding[0], output_padding[1], groups)
    
    out = input._new(out_data, (B, C_out, H_out, W_out), input.dtype, 
        input.device, requires_grad=_requires_grad, _children=tuple(_children))

    def _backward() -> None:
        out_grad         = out.grad.contiguous()
        B, C_in, H, W    = input.shape
        _, C_out, KH, KW = weight.shape
        H_out, W_out     = out.shape[2], out.shape[3]

        if input_requires_grad:
            grad_input = _conv_transpose2d_backward_input(
                out_grad.data, weight.data,
                B, C_in, H, W, C_out, KH, KW, H_out, W_out,
                stride[0], stride[1], padding[0], padding[1],
                dilation[0], dilation[1], groups)
            input.grad += input._new(
                grad_input, input.shape, float32, 'cpu')

        if weight_requires_grad:
            grad_weight = _conv_transpose2d_backward_weight(
                out_grad.data, input.data,
                B, C_in, H, W, C_out, KH, KW, H_out, W_out,
                stride[0], stride[1], padding[0], padding[1],
                dilation[0], dilation[1], groups)
            weight.grad += input._new(
                grad_weight, weight.shape, float32, 'cpu')

        if bias is not None and bias_requires_grad:
            bias.grad += input._new(
                out_grad.data.sum(axis=(0, 2, 3)),
                bias.shape, float32, 'cpu')

    out._backward = _backward
    return out

### CONVOLUTION FUNCTIONS ###

def _conv2d(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    B: int, C_in: int, H: int, W: int,
    C_out: int, KH: int, KW: int,
    H_out: int, W_out: int,
    stride_h: int, stride_w: int,
    padding_h: int, padding_w: int,
    dilation_h: int, dilation_w: int,
    groups: int
) -> tuple[np.ndarray, np.ndarray]:
    if padding_h > 0 or padding_w > 0:
        input_padded = np.pad(
            input, ((0,0),(0,0),(padding_h,padding_h),(padding_w,padding_w)))
    else:
        input_padded = input

    output = np.zeros((B, C_out, H_out, W_out), dtype=input.dtype)
    group_in  = C_in  // groups
    group_out = C_out // groups

    for g in range(groups):
        in_start  = g * group_in
        out_start = g * group_out
        for b in range(B):
            for co in range(group_out):
                for h in range(H_out):
                    for w in range(W_out):
                        s = bias[out_start + co] if bias is not None else 0.0
                        for ci in range(group_in):
                            for kh in range(KH):
                                for kw in range(KW):
                                    s += (
                                        input_padded[
                                            b, in_start + ci,
                                            h*stride_h + kh*dilation_h,
                                            w*stride_w + kw*dilation_w]
                                        * weight[out_start + co, ci, kh, kw])
                        output[b, out_start + co, h, w] = s

    return output, input_padded

def _conv_transpose2d(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    B: int, C_in: int, H_in: int, W_in: int,
    C_out: int, KH: int, KW: int,
    stride_h: int, stride_w: int,
    padding_h: int, padding_w: int,
    dilation_h: int, dilation_w: int,
    output_padding_h: int, output_padding_w: int,
    groups: int
) -> np.ndarray:
    H_full = (H_in - 1) * stride_h + dilation_h * (KH - 1) + 1
    W_full = (W_in - 1) * stride_w + dilation_w * (KW - 1) + 1
    H_out  = H_full - 2 * padding_h + output_padding_h
    W_out  = W_full - 2 * padding_w + output_padding_w

    output = np.zeros((B, C_out, H_full, W_full), dtype=input.dtype)

    group_in  = C_in  // groups
    group_out = C_out // groups

    for g in range(groups):
        in_start  = g * group_in
        out_start = g * group_out
        g_input  = input[:, in_start:in_start + group_in, :, :]
        g_weight = weight[in_start:in_start + group_in, :, :, :]

        for h in range(H_in):
            for w in range(W_in):
                for kh in range(KH):
                    for kw in range(KW):
                        out_h = h * stride_h + kh * dilation_h
                        out_w = w * stride_w + kw * dilation_w
                        if 0 <= out_h < H_full and 0 <= out_w < W_full:
                            output[
                                :, out_start:out_start + group_out, out_h, out_w
                            ] += g_input[:, :, h, w] @ g_weight[:, :, kh, kw]

    output = output[
        :, :,
        padding_h : padding_h + H_out,
        padding_w : padding_w + W_out]

    if bias is not None:
        output += bias[np.newaxis, :, np.newaxis, np.newaxis]
    return output

def _conv2d_backward_input(
    grad_output: np.ndarray,
    input_padded: np.ndarray,
    weight: np.ndarray,
    KH: int, KW: int,
    H_out: int, W_out: int,
    stride_h: int, stride_w: int,
    padding_h: int, padding_w: int,
    dilation_h: int, dilation_w: int,
    groups: int
) -> np.ndarray:
    B, C_in_padded, H_padded, W_padded = input_padded.shape
    C_out, C_in, _, _ = weight.shape

    H_in = H_padded - 2 * padding_h if padding_h > 0 else H_padded
    W_in = W_padded - 2 * padding_w if padding_w > 0 else W_padded

    group_in  = C_in_padded // groups
    group_out = C_out        // groups

    grad_input_padded = np.zeros_like(input_padded)

    for g in range(groups):
        in_start  = g * group_in
        out_start = g * group_out
        g_grad   = grad_output[:, out_start:out_start + group_out, :, :]
        g_weight = weight[out_start:out_start + group_out, :, :, :]

        for h in range(H_out):
            for w in range(W_out):
                for kh in range(KH):
                    for kw in range(KW):
                        grad_input_padded[
                            :, in_start:in_start + group_in,
                            h*stride_h + kh*dilation_h,
                            w*stride_w + kw*dilation_w
                        ] += np.einsum(
                            'bo,oi->bi',
                            g_grad[:, :, h, w],
                            g_weight[:, :, kh, kw])

    if padding_h > 0 or padding_w > 0:
        return grad_input_padded[
            :, :C_in,
            padding_h:padding_h + H_in,
            padding_w:padding_w + W_in]
    return grad_input_padded[:, :C_in, :, :]

def _conv_transpose2d_backward_input(
    grad_output: np.ndarray,
    weight: np.ndarray,
    B: int, C_in: int, H_in: int, W_in: int,
    C_out: int, KH: int, KW: int,
    H_out: int, W_out: int,
    stride_h: int, stride_w: int,
    padding_h: int, padding_w: int,
    dilation_h: int, dilation_w: int,
    groups: int
) -> np.ndarray:
    result, _ = _conv2d(
        grad_output, weight, None,
        B, C_out, H_out, W_out,
        C_in, KH, KW,
        H_in, W_in,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups)
    return result

def _conv2d_backward_weight(
    grad_output: np.ndarray,
    input_padded: np.ndarray,
    weight: np.ndarray,
    KH: int, KW: int,
    H_out: int, W_out: int,
    stride_h: int, stride_w: int,
    dilation_h: int, dilation_w: int,
    groups: int
) -> np.ndarray:
    B, C_in_padded, _, _ = input_padded.shape
    C_out, _, _, _ = weight.shape

    group_in  = C_in_padded // groups
    group_out = C_out        // groups

    grad_weight = np.zeros_like(weight)

    for g in range(groups):
        in_start  = g * group_in
        out_start = g * group_out
        g_input = input_padded[:, in_start:in_start + group_in, :, :]
        g_grad  = grad_output[:, out_start:out_start + group_out, :, :]

        for kh in range(KH):
            for kw in range(KW):
                patches = g_input[
                    :, :,
                    kh*dilation_h : kh*dilation_h + H_out*stride_h : stride_h,
                    kw*dilation_w : kw*dilation_w + W_out*stride_w : stride_w]
                grad_weight[
                    out_start:out_start + group_out, :, kh, kw
                ] = np.einsum('bohw,bihw->oi', g_grad, patches)

    return grad_weight

def _conv_transpose2d_backward_weight(
    grad_output: np.ndarray,
    input_padded: np.ndarray,
    B: int, C_in: int, H_in: int, W_in: int,
    C_out: int, KH: int, KW: int,
    H_out: int, W_out: int,
    stride_h: int, stride_w: int,
    padding_h: int, padding_w: int,
    dilation_h: int, dilation_w: int,
    groups: int
) -> np.ndarray:
    if padding_h > 0 or padding_w > 0:
        grad_output_full = np.pad(
            grad_output,
            ((0,0),(0,0),(padding_h,padding_h),(padding_w,padding_w)))
    else:
        grad_output_full = grad_output

    group_in  = C_in  // groups
    group_out = C_out // groups

    grad_weight = np.zeros((C_in, C_out, KH, KW), dtype=grad_output.dtype)

    for g in range(groups):
        in_start  = g * group_in
        out_start = g * group_out
        g_input = input_padded[:, in_start:in_start + group_in, :, :]
        g_grad  = grad_output_full[:, out_start:out_start + group_out, :, :]

        for kh in range(KH):
            for kw in range(KW):
                patches = g_grad[
                    :, :,
                    kh*dilation_h : kh*dilation_h + H_in*stride_h : stride_h,
                    kw*dilation_w : kw*dilation_w + W_in*stride_w : stride_w]
                grad_weight[
                    in_start:in_start + group_in,
                    out_start:out_start + group_out, kh, kw
                ] = np.einsum('bihw,bohw->io', g_input, patches)

    return grad_weight
    

