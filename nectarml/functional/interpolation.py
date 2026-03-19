from nectarml.tensor import Tensor
from nectarml import cuda, cpu

### UTILS ###

def _compute_output_size(
    input_shape: tuple,
    size: int | tuple | None,
    scale_factor: float | tuple | None
) -> tuple:
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

### UPSAMPLING ###

def upsample_nearest(
    input: Tensor,
    size: int | tuple[int, int] | tuple[int, int, int] | None = None,
    scale_factor: float | tuple[float, float] | tuple[float, float, float]\
        | None = None,
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
        out_data = cpu.interpolation.upsample_nearest(
            input.data, output_size)
        _backward_fn = lambda x : \
            cpu.interpolation.upsample_nearest_backward(x, input_size)
    
    input_requires_grad = input.requires_grad
    output_shape = (input.shape[0], input.shape[1]) + output_size
    out = Tensor(
        out_data, output_shape, input.dtype, input.device,
        input_requires_grad, _children=(input,))
    
    def _backward() -> None:
        if input_requires_grad:
            grad_ptr = _backward_fn(out.grad)
            input.grad += Tensor(
                grad_ptr, input.shape, input.dtype, input.device)
    
    out._backward = _backward
    return out
