import builtins
from typing import Any

import numpy as np

from nectarml.core import Tensor
from nectarml.typing import int32, float32
import nectarml.functional as F

def default_collate(inputs: list[Any]) -> Any:
    first = inputs[0]
    match first:
        case Tensor(): 
            return F.stack(inputs)
        case np.ndarray(): 
            return F.stack([Tensor(i) for i in inputs])
        case builtins.int():
            return F.stack([Tensor([i], dtype=int32) for i in inputs])
        case builtins.float():
            return F.stack([Tensor([i], dtype=float32) for i in inputs])
        case builtins.bytes() | builtins.str(): return inputs
        case None: return inputs
        case builtins.tuple():
            return tuple(default_collate([s[i] for s in inputs]) 
                         for i in range(len(first)))
        case builtins.dict():
            return {key: default_collate([s[key] for s in inputs]) 
                    for key in first}
        case builtins.list():
            return [default_collate([s[i] for s in inputs]) 
                    for i in range(len(first))]
        case _: 
            raise ValueError(
                f'default_collate does not support type [{type(first)}]. '
                f'Please provide a custom collate function.')


