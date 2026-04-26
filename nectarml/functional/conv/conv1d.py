from typing import Literal

from nectarml                    import cpu, cuda, typing
from nectarml.core             import Tensor
from nectarml.functional.padding import pad

### CPU ###

def _conv1d_cpu(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    B: int, C_in: int, L_out: int,
    C_out: int, K: int,
    stride: int,
    padding: int,
    dilation: int,
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

    out_data, input_padded = cpu.conv.conv1d(
        input.data, weight.data, bias.data if bias is not None else None,
        B, C_in, L_out, C_out, K, stride, padding, dilation, groups)
    out = Tensor._new(
        out_data, (B, C_out, L_out), input.dtype, input.device,
        requires_grad=_requires_grad, _children=tuple(_children))
    
    def _backward() -> None:
        out_grad = out.grad.contiguous()

        if input_requires_grad:
            grad_input = cpu.conv.conv1d_backward_input(
                out_grad.data, input_padded, weight.data,
                stride, padding, dilation, groups)
            input.grad += Tensor._new(
                grad_input, input.shape, typing.float32, 'cpu')
        
        if weight_requires_grad:
            grad_weight = cpu.conv.conv1d_backward_weight(
                out_grad.data, input_padded, weight.data,
                stride, padding, dilation, groups)
            weight.grad += Tensor._new(
                grad_weight, weight.shape, typing.float32, 'cpu')
        
        if bias is not None and bias_requires_grad:
            bias.grad += Tensor._new(
                out_grad.data.sum(axis=(0, 2)), 
                bias.shape, typing.float32, 'cpu')
                
    out._backward = _backward
    return out

### CUDA ###

def _conv1d_cuda(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    B: int, C_in: int, L_in: int,
    L_out: int, C_out: int, K: int,
    stride: int,
    padding: int,
    dilation: int,
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
        
    out_data = cuda.conv.conv1d(
        input, weight, bias, B, C_in, L_in, C_out, K,
        stride, padding, dilation, groups)
    out = Tensor._new(out_data, (B, C_out, L_out), input.dtype, input.device,
        requires_grad=_requires_grad, _children=tuple(_children))
        
    def _backward() -> None:
        out_grad = out.grad.contiguous()
        
        B, C_in, L_in = input.shape
        C_out, _, K = weight.shape
        L_out = out.shape[2]
    
        if input_requires_grad:
            grad_input_ptr = cuda.conv.conv1d_backward_input(
                out_grad, Tensor._fake(weight, typing.float32), 
                B, C_in, L_in, C_out, K, L_out,
                stride, padding, dilation, groups)
            grad_tensor = Tensor._new(
                grad_input_ptr, input.shape, typing.float32, input.device)
        
            input.grad += grad_tensor
        
        if weight_requires_grad:
            grad_weight_ptr = cuda.conv.conv1d_backward_weight(
                out_grad, Tensor._fake(input, typing.float32), 
                B, C_in, L_in, C_out, K, L_out,
                stride, padding, dilation)
            weight.grad += Tensor._new(
                grad_weight_ptr, weight.shape, typing.float32, weight.device)
        
        if bias is not None and bias_requires_grad:
            bias.grad += out_grad.sum(dim=(0, 2))
            
    out._backward = _backward
    return out

### WRAPPER ###

def conv1d(
    input:        Tensor,
    weight:       Tensor,
    bias:         Tensor | None = None,
    stride:       int = 1,
    padding:      int = 0,
    dilation:     int = 1,
    groups:       int = 1,
    padding_mode: Literal[
        'zeros', 'reflect', 'replicate', 'circular'
    ] = 'zeros'
) -> Tensor:
    assert input.ndim == 3, 'conv1d only valid for 1D (B, C, L) input.'
    B, C_in, L_in = input.shape
    C_out, _, K = weight.shape
    
    assert C_in % groups == 0, (
        f'Input channel count [{C_in}] must be evenly divisible by number '
        f'of groups [{groups}].')
    assert C_out % groups == 0, (
        f'Weight channel count [{C_out}] must be evenly divisible by number '
        f'of groups [{groups}].')
    
    if padding_mode != 'zeros' and (padding > 0):
        input = pad(input, (padding, padding), mode=padding_mode)
        B, C_in, L_in = input.shape
        padding = 0
    
    L_out = (L_in + 2*padding - dilation*(K-1) - 1) // stride + 1
    
    if input.device == 'cuda':
        return _conv1d_cuda(
            input, weight, bias, 
            B, C_in, L_in, L_out, C_out, K,
            stride, padding, dilation, groups)
    else:
        return _conv1d_cpu(
            input, weight, bias, 
            B, C_in, L_out, C_out, K,
            stride, padding, dilation, groups)
        

