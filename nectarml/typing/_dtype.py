from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml.typing import DTypeLike
    
import builtins
from   dataclasses import dataclass
from   typing      import Self

import numpy as np

import _nectarml

### DATA INFO ###

@dataclass
class iinfo:
    bits:  builtins.int
    min:   builtins.int
    max:   builtins.int
    dtype: dtype
    
    def __repr__(self: finfo) -> str:
        return (
            f'Bits:  {self.bits}\n'
            f'Min:   {self.min}\n'
            f'DType: {self.dtype.__str__()}'
        )
    
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
    
    def __repr__(self: finfo) -> str:
        return (
            f'Bits:            {self.bits}\n'
            f'Min:             {self.min}\n'
            f'Max:             {self.max}\n'
            f'Eps:             {self.eps}\n'
            f'Tiny:            {self.tiny}\n'
            f'Smallest Normal: {self.smallest_normal}\n'
            f'Resolution:      {self.resolution}\n'
            f'DType:           {self.dtype.__str__()}'
        )

### BASE DTYPE CLASS ###

@dataclass(frozen=True)
class DTypeMap:
    mappings = {
        np.float32: _nectarml.DType.Float32,
        np.float16: _nectarml.DType.Float16,
        np.half:    _nectarml.DType.Float16,
        np.int32:   _nectarml.DType.Int32,
        np.uint8:   _nectarml.DType.UInt8,
        np.bool_:   _nectarml.DType.Bool,
    }
    
    def map_dtype(
        self:  DTypeMap,
        dtype: np.typing.DTypeLike
    ) -> _nectarml.DType:
        return self.mappings[dtype]
        
_DTYPE_MAP  = DTypeMap()
_DTYPE_CACHE: dict[np.dtype, dtype] = {}

class dtype:  
    _map = _DTYPE_MAP
    
    def __new__(cls: type[Self], type_cpu: DTypeLike = None) -> Self:
        if type_cpu is None:         return object.__new__(cls)
        if type_cpu in _DTYPE_CACHE: return _DTYPE_CACHE[type_cpu]
        
        obj = object.__new__(cls)
        _DTYPE_CACHE[type_cpu] = obj
        return obj
    
    def __init__(self: dtype, type_cpu: DTypeLike) -> None:
        self._type_cpu         = type_cpu
        self.is_floating_point = np.issubdtype(type_cpu, np.floating)
        self.is_complex        = np.issubdtype(type_cpu, np.complexfloating)
        self.is_signed         = np.issubdtype(type_cpu, np.signedinteger)
        self.itemsize          = np.dtype(type_cpu).itemsize
        
        self.info = self._build_info()
    
    # INIT
    
    def _build_info(self: dtype) -> None:
        if self._type_cpu in [np.bool_]: return None
        
        if self.is_floating_point:
            _i = np.finfo(self._type_cpu)
            return finfo(
                _i.bits, _i.min, _i.max, _i.eps, _i.tiny, 
                _i.smallest_normal, _i.resolution, self)
        else:
            _i = np.iinfo(self._type_cpu)
            return iinfo(_i.bits, _i.min, _i.max, self)
              
    # PROPERTIES
    
    @property
    def numpy(self: dtype) -> np.typing.DTypeLike: 
        return self._type_cpu
    
    @property
    def cpu (self: dtype) -> np.typing.DTypeLike: 
        return self._type_cpu
    
    @property
    def cuda(self: dtype) -> _nectarml.DType:
        return self._map.map_dtype(self.cpu)
    
    @property
    def name(self: dtype) -> str:
        return self._type_cpu.__name__

    # COMPARISON
    
    def __hash__(self): return hash(self._type_cpu)

    def __eq__(self, other):
        if isinstance(other, dtype):
            return self._type_cpu == other._type_cpu
        return NotImplemented
    
    def __ne__(self: dtype, other: dtype) -> builtins.bool:
        if isinstance(other, dtype):
            return self._type_cpu != other._type_cpu
        return NotImplemented

    # INSPECTION
    
    def __str__ (self: dtype) -> str: return f'nectarml.{self.name}'
    def __repr__(self: dtype) -> str: return self.__str__()
    def __reduce__(self): return (dtype, (self._type_cpu,))

### DTYPE INSTANCES ###

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

