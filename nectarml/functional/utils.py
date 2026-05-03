import builtins
from typing import Literal

import nectarml.nn.functional as F
from   nectarml.core import tensor, Tensor
from   nectarml      import typing

###############################################################################
# SHAPE
###############################################################################

def is_shape_broadcastable(
    a: tensor | typing.Size, 
    b: tensor | typing.Size
) -> bool:
    '''Checks two shapes to see if they are broadcastable.

    Two shapes are considered broadcastable if:
    1. All dimensions from each shape match.

    **OR** 

    2. Any dimensions which do not match have a size of 1.
    
    Args:
        a : The first tensor or Size object for the check.
        b : The second tensor or Size object for the check.
        
    Returns:
        bool : True if the shapes are broadcastable, otherwise False.
    '''
    a_shape  = a.shape if isinstance(a, tensor) else a
    b_shape  = b.shape if isinstance(b, tensor) else b
    ndim     = max(len(a_shape), len(b_shape))
    
    a_padded = (1,) * (ndim - len(a_shape)) + tuple(a_shape)
    b_padded = (1,) * (ndim - len(b_shape)) + tuple(b_shape)
    
    for a, b in zip(a_padded, b_padded):
        if a != 1 and b != 1 and a != b: return False
    return True

###############################################################################
# CONVOLUTION
###############################################################################

def apply_kernel_2d(input: Tensor, kernel: Tensor) -> Tensor:
    '''Convolves a given input tensor with a given kernel tensor.

    Args:
        input  : The tensor to convolve.
        kernel : The kernel to use for the convolution.

    Returns:
        Tensor : The resulting tensor from the convolution.
    '''
    B, C, H, W = input.shape
    KH, KW     = kernel.shape
    kernel     = kernel.unsqueeze(0).unsqueeze(0)
    image_flat = input.reshape((B * C, 1, H, W))
    result     = F.conv2d(image_flat, kernel, padding=(KH//2, KW//2), groups=1)
    return result.reshape((B, C, H, W))

def convolve2d(
    input:  Tensor, 
    kernel: list[list[builtins.int | builtins.float]] = [
        [-2, -1,  0],
        [-1,  1,  1],
        [ 0,  1,  2]
    ]
) -> Tensor:
    '''Convolves input tensor with specified kernel.
    
    Args:
        input  : The tensor to convolve.
        kernel : The kernel to use for the convolution.
        
    Returns:
        Tensor : The resulting tensor from the convolution.
    '''
    kernel = Tensor(kernel, dtype=input.dtype, device=input.device)
    return apply_kernel_2d(input, kernel)

def edge_detect(
    input:       Tensor,
    mode:        Literal['sobel', 'prewitt', 'laplacian'] = 'sobel',
    scharr:      builtins.bool = False,
    per_channel: builtins.bool = False,
    eps:         float = 1e-8
) -> Tensor:
    '''Applies an edge detection algorithm to an input tensor.

    Input tensor must have ndim=3 (C, H, W) or ndim=4 (B, C, H, W)/
    
    Args:
        mode        : The edge detection algorithm to use. Options are 
                      ['sobel`, `prewitt`, `laplacian`].
        scharr      : If True and the mode is `sobel`, the edge detection will
                      use Scharr operator kernels rather than the traditional 
                      Sobel-Feldman kernels.
        per_channel : If True, the channels of the input will be split and the 
                      Sobel filter will be applied to each channel 
                      independently, then the results will be joined to form 
                      the output image.
        eps         : Epsilon to add to the result of the convolution operation
                      to avoid division by zero errors.
    
    Returns:
        Tensor : The result of the edge detection.
    '''
    assert input.ndim in [3, 4], (
        'Edge detect expects input tensor to have ndim=3 (C, H, W) ' 
        'or ndim=4 (B, C, H, W).')
    channel_dim = 0 if input.ndim == 3 else 1
    in_channels = input.shape[channel_dim]
    
    match mode:
        case 'sobel':
            if not scharr:
                kx = [[ 1, 0, -1], [ 2,  0,  -2], [ 1,   0, -1]]
                ky = [[ 1, 2,  1], [ 0,  0,   0], [-1,  -2, -1]]
            else:
                kx = [[3,  0, -3], [10,  0, -10], [ 3,   0, -3]]
                ky = [[3, 10,  3], [ 0,  0,   0], [-3, -10, -3]]
        case 'prewitt':
                kx = [[1,  0, -1], [ 1,  0,  -1], [ 1,   0, -1]]
                ky = [[1,  1,  1], [ 0,  0,   0], [-1,  -1, -1]]
        case 'laplacian':
                kx = [[0,  1,  0], [ 1, -4,   1], [ 0,   1,  0]]
                ky = None
        case _: raise ValueError(f'Edge detect mode not valid: {mode}')

    kernel_x = Tensor(kx, dtype=input.dtype, device=input.device)
    kernel_y = Tensor(ky, dtype=input.dtype, device=input.device)

    def _apply(_input: Tensor) -> Tensor:
        convolved = apply_kernel_2d(_input, kernel_x)**2
        if ky is not None: convolved += apply_kernel_2d(convolved, kernel_y)**2
        return convolved

    if in_channels == 1: return _apply(input, kernel_x, kernel_y)
    
    if not per_channel:
        gray    = input.mean(dim=channel_dim, keepdim=True)
        outputs = [F.sqrt(_apply(gray) + eps)] * in_channels
    else:
        outputs, channels = [], input.unbind(dim=channel_dim, keepdim=True)
        for ch in channels: outputs.append(F.sqrt(_apply(ch) + eps))
    
    return F.cat(outputs, dim=channel_dim)

###############################################################################
# INTERPOLATION
###############################################################################

def lerp(a: Tensor, b: Tensor, weight: Tensor) -> Tensor:
    '''Linearly interpolates between `a` and `b` tensors based on `weight`.

    All tensors must be on the same device and have the same DType, and the 
    weight tensor shape must be broadcastable to `a` and `b`. If the weight
    tensor `weight` only has a single channel, the same weights will be applied
    when interpolating every channel of the `a` and `b` tensors. If the weight 
    tensor has more than 1 channel, the channel number should match that of `a` 
    and `b`, and in this case, each channel of the weight tensor will weight 
    the interpolation of the corresponding channel of the input tensor.

    Args:
        a      : The tensor to interpolate from.
        b      : The tensor to interpolate to.
        weight : The weight tensor. Should be in range [0:1]. Weights the 
                 interpolation between the `a` tensor at 0.0 weight, and the 
                 `b` tensor at 1.0.
        
    Returns:
        Tensor : The resulting tensor from the linear interpolation.
    '''
    return a + weight * (b - a)

def lerp3(a: Tensor, b: Tensor, c: Tensor, weight: Tensor) -> Tensor:
    '''Linearly interpolated between 3 tensor's values by `weight`

    All tensors must be on the same device and have the same DType, and have
    broadcastable shapes. If the weight tensor `weight` only has a single 
    channel, the same weights will be applied when interpolating every channel 
    of the other tensors. If the weight tensor has more than 1 channel, the
    channel number should match that of the other tensors, and in this case,
    each channel of the weight tensor will weight the interpolation of the 
    corresponding channel of the input tensor.

    Args:
        a      : The first tensor.
        b      : The second tensor.
        c      : The third tensor.
        weight : The weight tensor. Should be in range [0:1]. Weights the 
                 interpolation between the `a` tensor at 0.0 weight, the `b`
                 tensor at 0.5 weight, and the `c` tensor at 1.0 weight.
        
    Returns:
        Tensor : The resulting tensor from the linear interpolation.
    '''
    w1 = (weight * 2).clamp(0.0, 1.0)
    w2 = ((weight - 0.5) * 2).clamp(0.0, 1.0)
    t  = (weight >= 0.5).to(weight.device, weight.dtype)
    return lerp(lerp(a, b, w1), lerp(b, c, w2), t)

###############################################################################
# NORMALIZATION
###############################################################################

def normalize_tensor_minmax(
    input:       Tensor,
    value_range: tuple[
        builtins.int | builtins.float, 
        builtins.int | builtins.float
    ] = (0.0, 1.0)
) -> Tensor:
    '''Normalizes the values of a tensor to saturate a given range.

    Args:
        input : The tensor to normalize.
        value_range : The range to normalize the tensor to.
        
    Returns:
        Tensor : The normalized tensor.
    '''
    range_min, range_max = value_range
    input_min, input_max = input.min().item(), input.max().item()
    
    if input_min == input_max:
        return input * 0.0 + range_min + (range_max - range_min) * 0.5
    diff = range_max - range_min
    return ((input-input_min) * (diff / (input_max-input_min)) + range_min)
