from typing import Literal
from collections.abc import Callable

import numpy as np

def pad(
    input: np.ndarray, 
    pad: tuple, 
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    value: float = 0.0
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    match mode:
        case 'constant':  _mode = 'constant'
        case 'reflect':   _mode = 'reflect'
        case 'replicate': _mode = 'edge'
        case 'circular':  _mode = 'wrap'
        case _: raise ValueError(f'Invalid padding mode: {mode}')
    
    pairs = [(pad[i+1], pad[i]) for i in range(0, len(pad), 2)]
    pairs.reverse()

    np_pad = [(0, 0)] * (input.ndim - len(pairs)) + pairs
    out = np.pad(
        input, np_pad, mode=_mode, 
        **({'constant_values': value} if _mode == 'constant' else {}))
    
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        slices = tuple(
            slice(p[0], s + p[0]) 
            for s, p in zip(input.shape, np_pad))
        return out_grad[slices]
    
    return out, _backward


