from __future__ import annotations

import builtins
from typing import Any, overload

import numpy as np

class Size(tuple[builtins.int, ...]):
    @property
    def ndim(self: Size) -> builtins.int:
        return len(self)
    
    def numel(self: Size) -> builtins.int:
        prod = 1
        for i in self: prod *= i
        return prod
    
    def reduce(
        self, 
        dim:     builtins.int | tuple[builtins.int, ...] | None, 
        keepdim: builtins.bool
    ) -> Size:
        if dim is None: return Size([])
        s = list(self)
        
        if keepdim:
            if isinstance(dim, (tuple, list)):
                for d in dim: s[d] = 1
            else: s[dim] = 1
        else:
            if isinstance(dim, (tuple, list)):
                for idx, d in enumerate(sorted(dim)):
                    s.pop(d - idx)
            else: s.pop(dim)
        return Size(s)
    
    @overload
    def __getitem__(self: Size, key: builtins.int) -> builtins.int: ...
    @overload
    def __getitem__(self: Size, key: slice) -> Size: ...

    def __getitem__(self: Size, key: Any) -> builtins.int | Size:
        if isinstance(key, (builtins.int, np.integer)):
            return super().__getitem__(key)
        elif isinstance(key, slice):
            x = slice(key.start, key.stop, key.step)
            return Size(super().__getitem__(x))
        raise ValueError(f'Unable to index Size object with type: {type(key)}')
        
    def __add__(self: Size, other: Size | tuple[builtins.int, ...]) -> Size:
        return Size(super().__add__(other))
    
    def __radd__(self: Size, other: Size | tuple[builtins.int, ...]) -> Size:
        return Size(tuple(other) + tuple(self))
    
    def __mul__(self: Size, other: builtins.int) -> Size:
        return Size(super().__mul__(other))
    
    def __rmul__(self: Size, other: builtins.int) -> Size:
        return Size(self * other)
    
    def __str__(self: Size) -> str:  return f'nectarml.Size({list(self)})'
    def __repr__(self: Size) -> str: return self.__str__()





