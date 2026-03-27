from typing import Literal

from nectarml.tensor import Tensor
from nectarml import cpu, cuda

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
    out = Tensor(
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
            input.grad += Tensor(
                grad_input, input.shape, input.dtype, 'cpu')
        
        if weight_requires_grad:
            grad_weight = cpu.conv.conv2d_backward_weight(
                out_grad.data, input_padded, weight.data,
                KH, KW, H_out, W_out,
                stride_h, stride_w, dilation_h, dilation_w, groups)
            weight.grad += Tensor(
                grad_weight, weight.shape, weight.dtype, 'cpu')
        
        if bias is not None and bias_requires_grad:
            bias.grad += Tensor(
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
        
    out_data = cuda.conv.conv2d(
        input, weight, bias, 
        B, C_in, H, W, C_out, KH, KW,
        stride_h, stride_w, padding_h, padding_w, 
        dilation_h, dilation_w, groups)
    out = Tensor(out_data, (B, C_out, H_out, W_out), input.dtype, input.device,
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
            input.grad += Tensor(
                grad_input_ptr, input.shape, input.dtype, input.device)
        
        if weight_requires_grad:
            grad_weight_ptr = cuda.conv.conv2d_backward_weight(
                out_grad, input, 
                B, C_in, H, W, C_out, KH, KW, H_out, W_out,
                stride_h, stride_w, padding_h, padding_w, 
                dilation_h, dilation_w)
            weight.grad += Tensor(
                grad_weight_ptr, weight.shape, weight.dtype, weight.device)
        
        if bias is not None and bias_requires_grad:
            bias.grad += out_grad.sum(dim=(0, 2, 3))
            
    out._backward = _backward
    return out

### WRAPPER ###

def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] | Literal['valid', 'same'] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1
) -> Tensor:
    B, C_in, H_in, W_in = input.shape
    C_out, _, KH, KW = weight.shape
    
    assert C_in % groups == 0, (
        f'Input channel count [{C_in}] must be evenly divisible by number '
        f'of groups [{groups}].')
    assert C_out % groups == 0, (
        f'Weight channel count [{C_out}] must be evenly divisible by number '
        f'of groups [{groups}].')
    
    if not isinstance(stride, tuple): 
        stride_h = stride_w = stride
    else: stride_h, stride_w = stride
    if not isinstance(dilation, tuple): 
        dilation_h = dilation_w = dilation
    else: dilation_h, dilation_w = dilation
    
    if padding == 'valid': padding_h = padding_w = 0
    elif padding == 'same': 
        padding_h = ((H_in - 1) * stride_h - H_in + dilation * (KH-1) + 1) // 2
        padding_w = ((W_in - 1) * stride_w - W_in + dilation * (KW-1) + 1) // 2
    elif not isinstance(padding, tuple): 
        padding_h = padding_w = padding
    else: padding_h, padding_w = padding
    
    H_out = (H_in + 2*padding_h - dilation_h*(KH-1) - 1) // stride_h + 1
    W_out = (W_in + 2*padding_w - dilation_w*(KW-1) - 1) // stride_w + 1
    
    if input.device == 'cuda':
        return _conv2d_cuda(
            input, weight, bias, 
            B, C_in, H_in, W_in, 
            C_out, KH, KW, H_out, W_out, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w, groups)
    else:
        return _conv2d_cpu(
            input, weight, bias,
            B, C_in, H_in, W_in, 
            C_out, KH, KW, H_out, W_out,
            stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w, groups)
   
       
