from typing import Literal

from nectarml.tensor import Tensor
from nectarml import cuda, cpu

### UTILS ###

MODE_NDIM = {
    'nearest':   None,
    'linear':    3,
    'bilinear':  4,
    'bicubic':   4,
    'trilinear': 5
}

def _compute_output_size(
    input_shape: tuple,
    size: int | tuple | None,
    scale_factor: float | tuple | None
) -> tuple[int, ...]:
    spatial_dims = input_shape[2:]
    n_dims = len(spatial_dims)
    
    if size is not None:
        if isinstance(size, int):
            return (size,) * n_dims
        return tuple(size)
    
    if scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            scale_factor = (scale_factor,) * n_dims
        return tuple(int(s * f) for s, f in zip(spatial_dims, scale_factor))
    
    raise ValueError('Either size or scale_factor must be specified')

### NEAREST NEIGHBOR ###

def upsample_nearest(
    input: Tensor,
    size: int | tuple[int, ...] | None = None,
    scale_factor: float | tuple[float, ...] | None = None,
) -> Tensor:
    output_size = _compute_output_size(input.shape, size, scale_factor)
    if input.device == 'cuda':
        match input.ndim:
            case 3: 
                L_in = input.shape[2]
                L_out = output_size[0]
                out_data = cuda.interpolation.upsample_nearest_1d(input, L_out)
                _backward_fn = lambda x : \
                    cuda.interpolation.upsample_nearest_1d_backward(
                        x, L_in, L_out)
            case 4:
                H_in, W_in = input.shape[2], input.shape[3]
                H_out, W_out = output_size
                out_data = cuda.interpolation.upsample_nearest_2d(
                    input, H_out, W_out)
                _backward_fn = lambda x : \
                    cuda.interpolation.upsample_nearest_2d_backward(
                        x, H_in, W_in, H_out, W_out)
            case 5: 
                D_in = input.shape[2]
                H_in = input.shape[3]
                W_in = input.shape[4]
                D_out, H_out, W_out = output_size
                out_data = cuda.interpolation.upsample_nearest_2d(
                    input, D_out, H_out, W_out)
                _backward_fn = lambda x : \
                    cuda.interpolation.upsample_nearest_3d_backward(
                        x, D_in, H_in, W_in, D_out, H_out, W_out)
            case _: raise ValueError(
                f'upsample_nearest requires input to have 3, 4, or 5 dims.')
    else:
        input_size = input.shape[2:]
        out_data = cpu.interpolation.upsample_nearest(input.data, output_size)
        _backward_fn = lambda x : \
            cpu.interpolation.upsample_nearest_backward(x, input_size)
    
    input_requires_grad = input.requires_grad
    output_shape = (input.shape[0], input.shape[1]) + output_size
    out = Tensor._new(
        out_data, output_shape, input.dtype, input.device,
        input_requires_grad, _children=(input,))
    
    def _backward() -> None:
        if input_requires_grad:
            grad_ptr = _backward_fn(out.grad)
            input.grad += Tensor._new(
                grad_ptr, input.shape, input.dtype, input.device)
    
    out._backward = _backward
    return out

### LINEAR ###

def upsample_linear(
    input: Tensor,
    size: int | None = None,
    scale_factor: float | None = None,
    align_corners: bool = False
) -> Tensor:
    output_size = _compute_output_size(input.shape, size, scale_factor)
    
    if input.device == 'cuda':
        L_in = input.shape[2]
        L_out = output_size[0]
        out_data = cuda.interpolation.upsample_linear(
            input, L_out, align_corners)
        _backward_fn = lambda x : \
            cuda.interpolation.upsample_linear_backward(
                x, L_in, L_out, align_corners)
    else: 
        input_size = input.shape[2:]
        out_data = cpu.interpolation.upsample_linear(
            input.data, output_size, align_corners)
        _backward_fn = lambda x : \
            cpu.interpolation.upsample_linear_backward(
                x, input_size, align_corners)

    input_requires_grad = input.requires_grad
    output_shape = (input.shape[0], input.shape[1]) + output_size
    out = Tensor._new(
        out_data, output_shape, input.dtype, input.device,
        input_requires_grad, _children=(input,))
    
    def _backward() -> None:
        if input_requires_grad:
            grad_ptr = _backward_fn(out.grad)
            input.grad += Tensor._new(
                grad_ptr, input.shape, input.dtype, input.device)
    
    out._backward = _backward
    return out

def upsample_bilinear(
    input: Tensor,
    size: int | tuple[int, int] | None = None,
    scale_factor: float | tuple[float, float] | None = None,
    align_corners: bool = False
) -> Tensor:
    output_size = _compute_output_size(input.shape, size, scale_factor)
    
    if input.device == 'cuda':
        H_in, W_in = input.shape[2], input.shape[3]
        H_out, W_out = output_size
        out_data = cuda.interpolation.upsample_bilinear(
            input, H_out, W_out, align_corners)
        _backward_fn = lambda x : \
            cuda.interpolation.upsample_bilinear_backward(
                x, H_in, W_in, H_out, W_out, align_corners)
    else: 
        input_size = input.shape[2:]
        out_data = cpu.interpolation.upsample_bilinear(
            input.data, output_size, align_corners)
        _backward_fn = lambda x : \
            cpu.interpolation.upsample_bilinear_backward(
                x, input_size, align_corners)

    input_requires_grad = input.requires_grad
    output_shape = (input.shape[0], input.shape[1]) + output_size
    out = Tensor._new(
        out_data, output_shape, input.dtype, input.device,
        input_requires_grad, _children=(input,))
    
    def _backward() -> None:
        if input_requires_grad:
            grad_ptr = _backward_fn(out.grad)
            input.grad += Tensor._new(
                grad_ptr, input.shape, input.dtype, input.device)
    
    out._backward = _backward
    return out

def upsample_trilinear(
    input: Tensor,
    size: int | tuple[int, int, int] | None = None,
    scale_factor: float | tuple[float, float, float] | None = None,
    align_corners: bool = False
) -> Tensor:
    output_size = _compute_output_size(input.shape, size, scale_factor)
    
    if input.device == 'cuda':
        D_in = input.shape[2]
        H_in = input.shape[3]
        W_in = input.shape[4]
        D_out, H_out, W_out = output_size
        out_data = cuda.interpolation.upsample_trilinear(
            input, D_out, H_out, W_out, align_corners)
        _backward_fn = lambda x : \
            cuda.interpolation.upsample_trilinear_backward(
                x, D_in, H_in, W_in, D_out, H_out, W_out, align_corners)
    else: 
        input_size = input.shape[2:]
        out_data = cpu.interpolation.upsample_trilinear(
            input.data, output_size, align_corners)
        _backward_fn = lambda x : \
            cpu.interpolation.upsample_trilinear_backward(
                x, input_size, align_corners)

    input_requires_grad = input.requires_grad
    output_shape = (input.shape[0], input.shape[1]) + output_size
    out = Tensor._new(
        out_data, output_shape, input.dtype, input.device,
        input_requires_grad, _children=(input,))
    
    def _backward() -> None:
        if input_requires_grad:
            grad_ptr = _backward_fn(out.grad)
            input.grad += Tensor._new(
                grad_ptr, input.shape, input.dtype, input.device)
    
    out._backward = _backward
    return out

### CUBIC ###

def upsample_bicubic(
    input: Tensor,
    size: int | tuple[int, int] | None = None,
    scale_factor: float | tuple[float, float] | None = None,
    a: float = -0.75,
    align_corners: bool = False
) -> Tensor:
    output_size = _compute_output_size(input.shape, size, scale_factor)
    
    if input.device == 'cuda':
        H_in, W_in = input.shape[2], input.shape[3]
        H_out, W_out = output_size
        out_data = cuda.interpolation.upsample_bicubic(
            input, H_out, W_out, a, align_corners)
        _backward_fn = lambda x : \
            cuda.interpolation.upsample_bicubic_backward(
                x, H_in, W_in, H_out, W_out, a, align_corners)
    else: 
        input_size = input.shape[2:]
        out_data = cpu.interpolation.upsample_bicubic(
            input.data, output_size, a, align_corners)
        _backward_fn = lambda x : \
            cpu.interpolation.upsample_bicubic_backward(
                x, input_size, a, align_corners)

    input_requires_grad = input.requires_grad
    output_shape = (input.shape[0], input.shape[1]) + output_size
    out = Tensor._new(
        out_data, output_shape, input.dtype, input.device,
        input_requires_grad, _children=(input,))
    
    def _backward() -> None:
        if input_requires_grad:
            grad_ptr = _backward_fn(out.grad)
            input.grad += Tensor._new(
                grad_ptr, input.shape, input.dtype, input.device)
    
    out._backward = _backward
    return out

### WRAPPER ###

def upsample(
    input: Tensor,
    size: int | tuple[int, ...] | None = None,
    scale_factor: float | tuple[float, ...] | None = None,
    mode: Literal[
        'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
    ] = 'nearest',
    a: float = -0.75,
    align_corners: bool = False,
    preserve_aspect_ratio: bool = False
) -> Tensor:
    mode_ndim = MODE_NDIM[mode]
    if mode_ndim is not None:
        assert input.ndim == mode_ndim, \
            f'Upsample mode [{mode}] expects input to have ndim={mode_ndim}.'
    
    spatial = input.shape[2:]
    if isinstance(size, int): size = (size,) * len(spatial)
    if isinstance(scale_factor, float): 
        scale_factor = (scale_factor,) * len(spatial)
    
    if preserve_aspect_ratio:
        if len(spatial) > 1:
            if size is not None:
                ratio = max(size) / max(spatial)
                size = tuple([int(i*ratio) for i in spatial])
            if scale_factor is not None:
                scale_factor = (max(scale_factor),) * len(spatial)
    
    match mode:
        case 'nearest': 
            return upsample_nearest(input, size, scale_factor)
        case 'linear': 
            return upsample_linear(input, size, scale_factor, align_corners)
        case 'bilinear': 
            return upsample_bilinear(input, size, scale_factor, align_corners)
        case 'trilinear': 
            return upsample_trilinear(input, size, scale_factor, align_corners)
        case 'bicubic': 
            return upsample_bicubic(
                input, size, scale_factor, a, align_corners)
        case _: raise ValueError(f'Invalid upsampling mode: {mode}')

