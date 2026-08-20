from typing import Literal

from nectarml                       import cpu, cuda, typing
from nectarml.core                  import Tensor
from nectarml.nn.functional.padding import pad

###############################################################################
# 1-DIMENSIONAL
###############################################################################

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
        B, C_in, L_in, C_out, K, 
        stride, padding, dilation, 
        output_padding, groups)
    out = Tensor._new(
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
            input.grad += Tensor._new(
                grad_input, input.shape, typing.float32, 'cpu')

        if weight_requires_grad:
            grad_weight = cpu.conv.conv_transpose1d_backward_weight(
                out_grad.data, input.data,
                B, C_in, L_in, C_out, K, L_out,
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
    out = Tensor._new(out_data, (B, C_out, L_out), input.dtype, input.device,
        requires_grad=_requires_grad, _children=tuple(_children))
    
    def _backward() -> None:
        out_grad = out.grad.contiguous()
        B, C_in, L_in = input.shape
        C_in_w, C_out, K = weight.shape
        L_out = out.shape[2]

        if input_requires_grad:
            grad_input_ptr = cuda.conv.conv_transpose1d_backward_input(
                out_grad, Tensor._fake(weight, typing.float32), 
                B, C_in, L_in, C_out, K, L_out,
                stride, padding, dilation, groups)
            input.grad += Tensor._new(
                grad_input_ptr, input.shape, typing.float32, input.device)

        if weight_requires_grad:
            grad_weight_ptr = cuda.conv.conv_transpose1d_backward_weight(
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

def conv_transpose1d(
    input:          Tensor,
    weight:         Tensor,
    bias:           Tensor | None = None,
    stride:         int = 1,
    padding:        int = 0,
    output_padding: int = 0,
    dilation:       int = 1,
    groups:         int = 1,
    padding_mode:   Literal['zeros'] = 'zeros'
) -> Tensor:
    assert input.ndim == 3, \
        'conv_transpose1d only valid for 1D (B, C, L) input.'
        
    B, C_in, L_in = input.shape
    C_in_w, C_out, K = weight.shape
    
    if padding_mode != 'zeros':
        raise ValueError(
            'Only padding_mode="zeros" is supported for conv_transpose1d.')
    assert C_in % groups == 0, (
        f'Input channel count [{C_in}] must be evenly divisible by number '
        f'of groups [{groups}].')
    assert C_out % groups == 0, (
        f'Weight channel count [{C_out}] must be evenly divisible by number '
        f'of groups [{groups}].')
    assert C_in_w == C_in, \
        f'Weight input channels {C_in_w} must match input channels {C_in}'
    assert 0 <= output_padding < stride, \
        f'output_padding must be in [0, stride-1] but got {output_padding} ' \
        f'with stride {stride}'
        
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


###############################################################################
# 2-DIMENSIONAL
###############################################################################

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
    '''Performs convolution on 4-dimensional inputs.

    Expects input tensors to have shape (B, C, H, W).
    
    Args:
        input        : The tensor to convolve.
        weight       : The weight tensor to use for the convolution operation.
        bias         : The bias tensor to use for the convolution, or None to 
                       not apply bias.
        stride       : Step size for the kernel.
        padding      : Padding width for inputs. Can be a single integer for 
                       HW, or a tuple of two integers for (H, W).
        dilation     : Size of the gap between the filter's sampling points. 
                       Larger values allow the filter to cover a larger spatial 
                       area without increasing parameter count.
        groups       : Number of groups to split the input channels into before 
                       computing the convolution. Note: `groups`>1 is currently 
                       only valid for CPU tensors.
        padding_mode : The padding mode to use. Options are:
                       1. `zeros`     : Fill the padded area with zeros.
                       2. `reflect`   : Mirrors edge pixels in the padded
                                       area.
                       3. `replicate` : Replicates the edge pixels into the
                                       padded area.
                       4. `circular`  : Wraps the opposite edges pixels to
                                       fill the padded area.
    
    Returns:
        Tensors : The resulting tensor from the convolution operation.
    '''
    input_shape = input.shape
    C_out, _, KH, KW = weight.shape

    assert input_shape[1] % groups == 0, (
        f'Input channels [{input_shape[1]}] must be divisible '
        f'by groups [{groups}].')
    assert C_out % groups == 0, \
        f'Output channels [{C_out}] must be divisible by groups [{groups}].'

    stride   = (stride,   stride)   if isinstance(stride,   int) else stride
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    padding  = (padding,  padding)  if isinstance(padding,  int) else padding

    if padding_mode != 'zeros' and (padding[0] > 0 or padding[1] > 0):
        input = pad(
            input, (padding[1], padding[1], padding[0], padding[0]),
            mode=padding_mode)
        input_shape = input.shape
        padding     = (0, 0)

    H_out = (input_shape[2] + 2*padding[0]-dilation[0]*(KH-1)-1) // stride[0]+1
    W_out = (input_shape[3] + 2*padding[1]-dilation[1]*(KW-1)-1) // stride[1]+1

    conv_fn = cuda.conv.conv2d if input.device == 'cuda' else cpu.conv.conv2d
    return conv_fn(
        input, weight, bias, input_shape,
        C_out, KH, KW, H_out, W_out,
        stride, padding, dilation, groups)

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
    '''Performs transposed convolution on 4-dimensional inputs.

    Expects input tensors to have shape (B, C, H, W).
    
    Args:
        input          : The tensor to convolve.
        weight         : The weight tensor to use for the transposed 
                         convolution.
        bias           : The bias tensor to use for the transposed convolution, 
                         or None to not apply bias.
        stride         : Step size for the kernel.
        padding        : Padding width for inputs. Can be a single integer for 
                         HW, or a tuple of two integers for (H, W).
        output_padding : Number of rows and columns to add to the output tensor
                         after the transposed convolution. Can be a single 
                         integer for HW, or a tuple of two integers for (H, W).
        dilation       : Size of the gap between the filter's sampling points. 
                         Larger values allow the filter to cover a larger 
                         spatial area without increasing parameter count.
        groups         : Number of groups to split the input channels into 
                         before computing the convolution. Note: `groups`>1 is 
                         currently only valid for CPU tensors.
        padding_mode   : The padding mode to use. `zeros` is the only valid
                         padding mode for transposed convolution.
    
    Returns:
        Tensors : The resulting tensor from the transposed convolution.
    '''
    input_shape           = input.shape
    C_in_w, C_out, KH, KW = weight.shape
    
    if padding_mode != 'zeros':
        raise ValueError(
            'Only padding_mode="zeros" is supported for conv_transpose2d.')
    stride   = (stride,   stride)   if isinstance(stride,   int) else stride
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    padding  = (padding,  padding)  if isinstance(padding,  int) else padding
    output_padding = (output_padding, output_padding) \
        if isinstance(output_padding, int) else output_padding

    assert input_shape[1] % groups == 0, (
        f'Input channels [{input.shape[1]}] must be divisible '
        f'by groups [{groups}].')
    assert C_in_w == input_shape[1], (
        f'Weight input channels [{C_in_w}] must match input '
        f'channels [{input_shape[1]}].')
    assert C_out % groups == 0, \
        f'Output channels [{C_out}] must be divisible by groups [{groups}].'
    assert 0 <= output_padding[0] < stride[0], \
        f'output_padding_h [{output_padding[0]}] must be in [0, stride_h-1].'
    assert 0 <= output_padding[1] < stride[1], \
        f'output_padding_w [{output_padding[1]}] must be in [0, stride_w-1].'

    H_out = (input_shape[2]-1)*stride[0] \
          - 2*padding[0] + dilation[0]*(KH-1) + 1 + output_padding[0]
    W_out = (input_shape[3]-1)*stride[1] \
          - 2*padding[1] + dilation[1]*(KW-1) + 1 + output_padding[1]

    conv_fn = cuda.conv.conv_transpose2d \
           if input.device == 'cuda' else cpu.conv.conv_transpose2d
    return conv_fn(
        input, weight, bias, input_shape,
        C_out, KH, KW, H_out, W_out, 
        stride, padding, output_padding, dilation, groups)

###############################################################################
# 3-DIMENSIONAL
###############################################################################

def conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int | Literal['valid', 'same'] = 0,
    dilation: int = 1,
    groups: int = 1
) -> Tensor:
    raise NotImplementedError('3D convolution is currently not supported.')

def conv_transpose3d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int | Literal['valid', 'same'] = 0,
    dilation: int = 1,
    groups: int = 1
) -> Tensor:
    raise NotImplementedError('3D convolution is currently not supported.')


