from __future__ import annotations

import builtins
from typing import Any, TypeAlias, overload

import numpy as np


'''
DTYPE REWRITE (NOT READY FOR DEPLOYMENT YET!)

from dataclasses import dataclass
from _nectarml import DType

ArrayLike = np.typing.ArrayLike
DTypeLike = np.typing.DTypeLike

@dataclass
class iinfo:
    bits:  builtins.int
    min:   builtins.int
    max:   builtins.int
    dtype: dtype
    
@dataclass
class finfo:
    bits:            builtins.float
    min:             builtins.float
    max:             builtins.float
    eps:             builtins.float
    tiny:            builtins.float
    smallest_normal: builtins.float
    resolution:      builtins.float
    dtype:           dtype

class dtype:  
    _map = {
        typing.float:   DType.Float32,
        typing.float32: DType.Float32,
        typing.float16: DType.Float16,
        typing.half:    DType.Float16,
        typing.int32:   DType.Int32,
        typing.uint8:   DType.UInt8,
        typing.bool_:   DType.Bool,
    }
      
    def __init__(self: dtype, type_cpu: DTypeLike) -> None:
        self.type_cpu          = type_cpu
        self.name              = type_cpu.__name__
        self.is_floating_point = np.issubdtype(type_cpu, np.floating)
        self.is_complex        = np.issubdtype(type_cpu, np.complexfloating)
        self.is_signed         = np.issubdtype(type_cpu, np.signedinteger)
        self.itemsize          = np.dtype(type_cpu).itemsize
        
        self._build_info()
        
    def _build_info(self: dtype) -> None:
        if self.is_floating_point:
            _i = np.finfo(self.type_cpu)
            self.info = finfo(
                _i.bits, _i.min, _i.max, _i.eps, _i.tiny, 
                _i.smallest_normal, _i.resolution, self)
        else:
            _i = np.iinfo(self.type_cpu)
            self.info = iinfo(_i.bits, _i.min, _i.max, self)
                
    def cpu(self: dtype) -> DTypeLike: return self.type_cpu
    def cuda(self: dtype) -> DType: return self._map[self.type_cpu]

    def __repr__(self): return f'nectarml.{self.name}'

float   = dtype(np.float32)
float16 = dtype(np.float16)
float32 = dtype(np.float32)
float64 = dtype(np.float64)

half    = dtype(np.half)
double  = dtype(np.double)

int     = dtype(np.int64)
int8    = dtype(np.int8)
int16   = dtype(np.int16)
int32   = dtype(np.int32)
int64   = dtype(np.int64)
short   = dtype(np.short)
long    = dtype(np.long)

uint    = dtype(np.uint)
uint8   = dtype(np.uint8)
uint16  = dtype(np.uint16)
uint32  = dtype(np.uint32)
uint64  = dtype(np.uint64)

bool_   = dtype(np.bool_)

'''



'''

NEW DEVICE CLASS (NOT READY FOR DEPLOYMENT YET!)

import builtins
from typing import Literal, Any, overload, TypeAlias
from dataclasses import dataclass

@dataclass
class device:
    type:      Literal['cpu', 'cuda']
    device_id: builtins.int | None = None

    @overload
    def __init__(self, type: DeviceLikeType) -> None: ...
    @overload
    def __init__(
        self: device, 
        type: str, 
        device_id: builtins.int | None = None
    ) -> None: ...
    def __init__(
        self:      device, 
        type:      str,
        device_id: builtins.int | None = None
    ) -> None:
        assert type in ['cpu', 'cuda'], f'Device type not valid: {type}'
        if device_id is not None:
            assert isinstance(device_id, builtins.int) and device_id >= 0, \
                'device_id must be an integer value >= 0.'
        else: device_id = 0
                
        super().__setattr__('type', type)
        super().__setattr__('device_id', device_id)
        
    def __get__(
        self:     device, 
        instance: device | None, 
        owner:       Any | None = None
    ) -> device | str:
        if instance is None: return self
        return self.type

'''

### NUMPY DTYPE ALIASING ###

ArrayLike = np.typing.ArrayLike
DTypeLike = np.typing.DTypeLike

float   = np.float32
float16 = np.float16
float32 = np.float32
float64 = np.float64

half    = np.half
double  = np.double

int     = np.int64
int8    = np.int8
int16   = np.int16
int32   = np.int32
int64   = np.int64
short   = np.short
long    = np.long

uint    = np.uint
uint8   = np.uint8
uint16  = np.uint16
uint32  = np.uint32
uint64  = np.uint64

bool_   = np.bool_

### nectarml.Size ###

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


### COMMON TYPE ALIASING ###

# ShapeType: TypeAlias = (
#     builtins.int
#   | list[builtins.int]
#   | tuple[builtins.int, ...]
#   | Size
# )
# DimsType: TypeAlias = (
#     builtins.int
#   | list[builtins.int]
#   | tuple[builtins.int, ...]
# )
# DeviceLikeType: TypeAlias = builtins.str | device
# NumberType:     TypeAlias = builtins.int | builtins.float



