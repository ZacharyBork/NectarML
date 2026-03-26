from typing import Literal

from nectarml.tensor import Tensor
from nectarml import cpu, cuda

### CPU ###

def _conv_transpose1d_cpu(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    B: int, C_in: int, L_in: int,
    L_out: int, C_out: int, K: int,
    stride: int,
    padding: int,
    output_padding: int,
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

    out_data = cpu.conv.conv_transpose1d(
        input.data, weight.data, bias.data if bias is not None else None,
        B, C_in, L_in, C_out, K, stride, padding, output_padding,
        dilation, groups)
    out = Tensor(
        out_data, (B, C_out, L_out), input.dtype, input.device,
        requires_grad=_requires_grad, _children=tuple(_children))
    
    def _backward() -> None:
        out_grad = out.grad.contiguous()
        B, C_in, L_in = input.shape
        C_in_w, C_out, K = weight.shape
        L_out = out.shape[2]

        if input_requires_grad:
            grad_input = cpu.conv.conv_transpose1d_backward_input(
                out_grad.data, weight.data,
                B, C_in, L_in, C_out, K, L_out,
                stride, padding, dilation, groups)
            input.grad += Tensor(
                grad_input, input.shape, input.dtype, 'cpu')

        if weight_requires_grad:
            grad_weight = cpu.conv.conv_transpose1d_backward_weight(
                out_grad.data, input.data,
                B, C_in, L_in, C_out, K, L_out,
                stride, padding, dilation, groups)
            weight.grad += Tensor(
                grad_weight, weight.shape, weight.dtype, 'cpu')

        if bias is not None and bias_requires_grad:
            bias.grad += Tensor(
                out_grad.data.sum(axis=(0, 2)),
                bias.shape, bias.dtype, 'cpu')
                
    out._backward = _backward
    return out

### CUDA ###

def _conv_transpose1d_cuda(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    B: int, C_in: int, L_in: int,
    L_out: int, C_out: int, K: int,
    stride: int,
    padding: int,
    output_padding: int,
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
        
    out_data = cuda.conv.conv_transpose1d(
        input, weight, bias, B, C_in, L_in, C_out, K,
        stride, padding, output_padding, dilation, groups)
    out = Tensor(out_data, (B, C_out, L_out), input.dtype, input.device,
        requires_grad=_requires_grad, _children=tuple(_children))
    
    def _backward() -> None:
        out_grad = out.grad.contiguous()
        B, C_in, L_in = input.shape
        C_in_w, C_out, K = weight.shape
        L_out = out.shape[2]

        if input_requires_grad:
            grad_input_ptr = cuda.conv.conv_transpose1d_backward_input(
                out_grad, weight, B, C_in, L_in, C_out, K, L_out,
                stride, padding, dilation, groups)
            input.grad += Tensor(
                grad_input_ptr, input.shape, input.dtype, input.device)

        if weight_requires_grad:
            grad_weight_ptr = cuda.conv.conv_transpose1d_backward_weight(
                out_grad, input, B, C_in, L_in, C_out, K, L_out,
                stride, padding, dilation)
            weight.grad += Tensor(
                grad_weight_ptr, weight.shape, weight.dtype, weight.device)

        if bias is not None and bias_requires_grad:
            bias.grad += out_grad.sum(dim=(0, 2))
            
    out._backward = _backward
    return out

### WRAPPER ###

def conv_transpose1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int | Literal['valid', 'same'] = 0,
    output_padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> Tensor:
    B, C_in, L_in = input.shape
    C_in_w, C_out, K = weight.shape
    
    assert C_in % groups == 0, (
        f'Input channel count [{C_in}] must be evenly divisible by number '
        f'of groups [{groups}].')
    assert C_in_w == C_in, \
        f'Weight input channels {C_in_w} must match input channels {C_in}'
    assert C_out % groups == 0, (
        f'Weight channel count [{C_out}] must be evenly divisible by number '
        f'of groups [{groups}].')
    assert 0 <= output_padding < stride, \
        f'output_padding must be in [0, stride-1] but got {output_padding} ' \
        f'with stride {stride}'
    
    if padding == 'valid': padding = 0
    elif padding == 'same': 
        raise ValueError('padding=same is not supported for conv_transpose1d')
    L_out = (L_in - 1) * stride - 2*padding + dilation*(K-1) + 1+output_padding
    
    
    if input.device == 'cuda':
        return _conv_transpose1d_cuda(
            input, weight, bias, 
            B, C_in, L_in, L_out, C_out, K,
            stride, padding, output_padding,
            dilation, groups)
    else:
        return _conv_transpose1d_cpu(
            input, weight, bias,
            B, C_in, L_in, L_out, C_out, K,
            stride, padding, output_padding,
            dilation, groups)
        
        
