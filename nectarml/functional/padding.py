from typing import Literal

from nectarml.tensor import Tensor
from nectarml import cpu, cuda
from nectarml.functional.common import _eval_core_function

# def pad(
#     input: Tensor, 
#     pad: int | tuple[int, ...],
#     mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
#     value: float = 0.0
# ) -> Tensor:
#     return _eval_core_function(
#         input, lambda x : padding.pad(x, pad, mode, value))
    
def _normalize_padding(
    pad: int | tuple[int, ...], 
    ndim: int
) -> tuple[list[int], list[int]]:
    if isinstance(pad, int): return [pad] * (ndim - 2), [pad] * (ndim - 2)

    pairs = [(pad[i], pad[i+1]) for i in range(0, len(pad), 2)]
    pairs.reverse()
    
    n_spatial = ndim - 2
    while len(pairs) < n_spatial: pairs.insert(0, (0, 0))
    return [p[0] for p in pairs], [p[1] for p in pairs]
  
def _compute_pad_output_shape(
    input_shape: tuple[int, ...],
    pad_before: list[int],
    pad_after: list[int]
) -> tuple[int, ...]:
    spatial_out = tuple(
        s + b + a 
        for s, b, a in zip(input_shape[2:], pad_before, pad_after))
    return input_shape[:2] + spatial_out
    
def pad(
    input: Tensor, 
    pad: int | tuple[int, ...],
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    value: float = 0.0
) -> Tensor:
    if input.device == 'cuda':
        pad_before, pad_after = _normalize_padding(pad, input.ndim)
        out_data = cuda.padding.pad(input, pad, mode, value)
        shape = _compute_pad_output_shape(input.shape, pad_before, pad_after)
    else:
        out_data = cpu.padding.pad(input, pad, mode, value)
        shape = out_data.shape
        
    input_requires_grad = input.requires_grad
    out = Tensor(out_data, shape, input.dtype, input.device,
        input_requires_grad, _children=(input,))

    def _backward() -> None:
        if input_requires_grad:
            slices = (slice(None), slice(None)) + tuple(
                slice(pad_before[d], pad_before[d] + input.shape[d + 2])
                for d in range(len(pad_before)))
            
            input.grad += out.grad[slices]

    out._backward = _backward
    return out

