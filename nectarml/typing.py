from __future__ import annotations

import builtins
from typing import Any, overload

import numpy as np

ArrayLike = np.typing.ArrayLike
DTypeLike = np.typing.DTypeLike

float = np.float32
float16 = np.float16
float32 = np.float32

half = np.half
double = np.double

int = np.int64
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
short = np.short
long = np.long

uint = np.uint
uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64

bool_ = np.bool_

class Size(tuple[int, ...]):
    @property
    def ndim(self: Size) -> int:
        return len(self)
    
    def numel(self: Size) -> int:
        prod = 1
        for i in self: prod *= i
        return prod
    
    def reduce(
        self, 
        dim: int | tuple[int, ...] | None, 
        keepdim: bool
    ) -> Size:
        if dim is None: return Size((1,))
        s = list(self)
        
        if not keepdim: 
            if dim is not None:
                if isinstance(dim, (tuple, list)):
                    for idx, i in enumerate(dim): s.pop(i-idx)
                else: s.pop(dim)
        else: s[dim] = 1
        return Size(s)
    
    @overload
    def __getitem__(self: Size, key: int) -> int: ...
    @overload
    def __getitem__(self: Size, key: slice) -> Size: ...

    def __getitem__(self: Size, key: Any) -> int | Size:
        if isinstance(key, (builtins.int, np.integer)):
            return super().__getitem__(key)
        elif isinstance(key, slice):
            x = slice(key.start, key.stop, key.step)
            return Size(super().__getitem__(x))
        raise ValueError(f'Unable to index Size object with type: {type(key)}')
        
    def __add__(self: Size, other: Size | tuple[int, ...]) -> Size:
        return Size(super().__add__(other))
    
    def __radd__(self: Size, other: Size | tuple[int, ...]) -> Size:
        return Size(self + other)
    
    def __mul__(self: Size, other: int) -> Size:
        return Size(super().__mul__(other))
    
    def __rmul__(self: Size, other: int) -> Size:
        return Size(self * other)
    
    def __str__(self: Size) -> str:
        return f'nectarml.Size({list(self)})'

    def __repr__(self: Size) -> str: return self.__str__()


