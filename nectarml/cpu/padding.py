from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor
    
from typing import Literal

import numpy as np

def pad(
    input: Tensor, 
    pad:   int | tuple[int, ...], 
    mode:  Literal[
        'constant', 'reflect', 'replicate', 'circular'
    ] = 'constant',
    value: float = 0.0
) -> np.ndarray:
    match mode:
        case 'constant':  _mode = 'constant'
        case 'reflect':   _mode = 'reflect'
        case 'replicate': _mode = 'edge'
        case 'circular':  _mode = 'wrap'
        case _: raise ValueError(f'Invalid padding mode: {mode}')
    
    ndim = input.ndim
    if isinstance(pad, int): return [pad] * (ndim - 2), [pad] * (ndim - 2)
    pairs = [(pad[i+1], pad[i]) for i in range(0, len(pad), 2)]
    pairs.reverse()

    np_pad = [(0, 0)] * (input.ndim - len(pairs)) + pairs
    return np.pad(
        input.data, np_pad, mode=_mode, 
        **({'constant_values': value} if _mode == 'constant' else {}))



