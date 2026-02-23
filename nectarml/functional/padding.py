from typing import Literal

import numpy as np

from nectarml import Tensor

def pad(
    input: Tensor, 
    pad: tuple, 
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    value: float = 0.0
) -> Tensor:
    match mode:
        case 'constant':  _mode = 'constant'
        case 'reflect':   _mode = 'reflect'
        case 'replicate': _mode = 'edge'
        case 'circular':  _mode = 'wrap'
        case _: raise ValueError(f'Invalid padding mode: {mode}')
    
    pairs = [(pad[i+1], pad[i]) for i in range(0, len(pad), 2)]
    pairs.reverse()

    num_unpadded = input.data.ndim - len(pairs)
    np_pad = [(0, 0)] * num_unpadded + pairs
    
    out = input._build_output_tensor(
        np.pad(
            input.data, np_pad, mode=_mode, 
            **({'constant_values': value} if _mode == 'constant' else {})),
        (input,))
    def _backward():
        if input.requires_grad:
            slices = tuple(
                slice(p[0], s + p[0]) 
                for s, p in zip(input.data.shape, np_pad))
            input.grad += out.grad[slices]
    out._backward = _backward
    return out

