import builtins
from   typing import Literal

from nectarml      import cpu, cuda, typing
from nectarml.core import Tensor

### UTILITIES ###

def _normalize_padding(
    pad:  builtins.int | tuple[builtins.int, ...], 
    ndim: builtins.int
) -> tuple[list[builtins.int], list[builtins.int]]:
    if isinstance(pad, builtins.int): 
        return [pad] * (ndim - 2), [pad] * (ndim - 2)

    pairs = [(pad[i], pad[i+1]) for i in range(0, len(pad), 2)]
    pairs.reverse()
    
    n_spatial = ndim - 2
    while len(pairs) < n_spatial: pairs.insert(0, (0, 0))
    return [p[0] for p in pairs], [p[1] for p in pairs]
  
def _compute_pad_output_shape(
    input_shape: tuple[builtins.int, ...],
    pad_before:  list[builtins.int],
    pad_after:   list[builtins.int]
) -> tuple[builtins.int, ...]:
    spatial_out = tuple(
        s + b + a 
        for s, b, a in zip(input_shape[2:], pad_before, pad_after))
    return input_shape[:2] + spatial_out
    
### WRAPPER ###
    
def _pad_backward(
    input:         Tensor,
    out:           Tensor,
    requires_grad: builtins.bool,
    pad_before:    list[builtins.int],
    pad_after:     list[builtins.int],
    mode:          Literal['constant', 'reflect', 'replicate', 'circular'],
) -> None:
    if requires_grad:
        if input.device == 'cuda':
            dx_ptr = cuda.tensor.padding.pad_backward(
                out.grad, list(input.shape), pad_before, pad_after, mode)
            dx = Tensor._new(dx_ptr, input.shape, typing.float32, input.device)
            input.grad += dx
            return

        H,  W  = input.shape[2], input.shape[3]
        pt, pb =  pad_before[0],   pad_after[0]
        pl, pr =  pad_before[1],   pad_after[1]
        
        g    = out.grad
        grad = g[:, :, pt:pt+H, pl:pl+W].clone()
        
        if pt > 0:
            grad[:, :, 1:pt+1, :] += \
                g[:, :, 0:pt, pl:pl+W].flip(2)  
            if pl > 0:
                grad[:, :, 1:pt+1, 1:pl+1] += \
                    g[:, :, 0:pt, 0:pl].flip(2, 3)
                    
        if pb > 0:
            grad[:, :, H-pb-1:H-1, :] += \
                g[:, :, pt+H:pt+H+pb, pl:pl+W].flip(2)
            if pr > 0:
                grad[:, :, 1:pt+1, W-pr-1:W-1] += \
                    g[:, :, 0:pt, pl+W:pl+W+pr].flip(2, 3)
        
        if pl > 0:
            grad[:, :, :, 1:pl+1] += \
                g[:, :, pt:pt+H, 0:pl].flip(3)
            if pb > 0:
                grad[:, :, H-pb-1:H-1, 1:pl+1] += \
                    g[:, :, pt+H:pt+H+pb, 0:pl].flip(2, 3)
        
        if pr > 0:
            grad[:, :, :, W-pr-1:W-1] += \
                g[:, :, pt:pt+H, pl+W:pl+W+pr].flip(3)
            if pb > 0:
                grad[:, :, H-pb-1:H-1, W-pr-1:W-1] += \
                    g[:, :, pt+H:pt+H+pb, pl+W:pl+W+pr].flip(2, 3)
                    
        input.grad += grad
    
def pad(
    input: Tensor, 
    pad:   builtins.int | tuple[builtins.int, ...],
    mode:  Literal[
        'constant', 'reflect', 'replicate', 'circular'
    ] = 'constant',
    value: builtins.float = 0.0
) -> Tensor:
    if input.device == 'cuda':
        before, after = _normalize_padding(pad, input.ndim)
        out_data      = cuda.padding.pad(input, before, after, mode, value)
        shape         = _compute_pad_output_shape(input.shape, before, after)
    else:
        out_data = cpu.padding.pad(input, pad, mode, value)
        shape    = out_data.shape
        
    input_requires_grad = input.requires_grad
    out = Tensor._new(out_data, shape, input.dtype, input.device,
        input_requires_grad, _children=(input,))

    def _backward() -> None:
        _pad_backward(input, out, input_requires_grad, before, after, mode)

    out._backward = _backward
    return out

