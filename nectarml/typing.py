from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import Literal, Any, Self, TypeAlias, overload

import numpy as np

import _nectarml

### DTYPE CLASSES ###

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
        
DTYPE_MAP = DTypeMap()

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

class dtype:  
    _map = DTYPE_MAP
    
    def __init__(self: dtype, type_cpu: DTypeLike) -> None:
        self._type_cpu         = type_cpu
        self.is_floating_point = np.issubdtype(type_cpu, np.floating)
        self.is_complex        = np.issubdtype(type_cpu, np.complexfloating)
        self.is_signed         = np.issubdtype(type_cpu, np.signedinteger)
        self.itemsize          = np.dtype(type_cpu).itemsize
        
        self.info = self._build_info()
    
    ### INIT ###
    
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
              
    ### PROPERTIES ###
    
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

    ### COMPARISON ###
    
    def __hash__(self: dtype) -> builtins.int: return id(self)
    def __eq__(self: dtype, other: dtype) -> builtins.bool:
        if not isinstance(other, dtype):
            raise ValueError(
                f'Unable to compare dtype object to type [{type(other)}]')
        return self.name == other.name
    
    def __ne__(self: dtype, other: dtype) -> builtins.bool:
        return self.name != other.name
    
    ### INSPECTION ###
    
    def __str__ (self: dtype) -> str: return f'nectarml.{self.name}'
    def __repr__(self: dtype) -> str: return self.__str__()

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

### DEVICE CLASS ###

_DEVICE_CACHE: dict[tuple, device] = {}

@dataclass
class device:
    type:      Literal['cpu', 'cuda']
    device_id: builtins.int | None = None

    def __new__(
        cls:       type[Self], 
        type:      DeviceLikeType | builtins.str, 
        device_id: builtins.int   | None = None
    ) -> None:
        if isinstance(type, device):
            type      = type.type
            device_id = device_id if device_id is not None else type.device_id
        if type == 'cuda' and device_id is None: device_id = 0
        if type == 'cpu': device_id = None
        
        key = (type, device_id)
        if key in _DEVICE_CACHE:
            return _DEVICE_CACHE[key]
        
        obj = object.__new__(cls)
        _DEVICE_CACHE[key] = obj
        return obj

    @overload
    def __init__(self, type: DeviceLikeType) -> None: ...
    @overload
    def __init__(
        self:      device, 
        type:      builtins.str, 
        device_id: builtins.int | None = None
    ) -> None: ...
    def __init__(
        self:      device, 
        type:      builtins.str,
        device_id: builtins.int | None = None
    ) -> None:
        if hasattr(self, 'type'): return
        
        if isinstance(type, device):
            device_id = device_id if device_id is not None else type.device_id
            type = type.type
            
        assert type in ['cpu', 'cuda'], f'Device type not valid: {type}'
        if device_id is not None:
            assert isinstance(device_id, builtins.int) and device_id >= 0, \
                'device_id must be an integer value >= 0.'
                
        super().__setattr__('type', type)
        super().__setattr__('device_id', device_id)
    
    def __eq__(self: device, other: DeviceLikeType) -> builtins.bool:
        if isinstance(other, str):
            return self.type == other
        if isinstance(other, device):
            return self.type      == other.type \
               and self.device_id == other.device_id
        return NotImplemented

    def __hash__(self: device) -> builtins.int:
        return hash((self.type, self.device_id))

    def __str__(self: device) -> builtins.str:
        if self.device_id is not None:
            return f'{self.type}:{self.device_id}'
        return self.type

    def __repr__(self: device) -> builtins.str:
        return (
            f'nectarml.device('
            f'type="{self.type}", '
            f'device_id={self.device_id})'
        )

### SIZE CLASS ###

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

DeviceLikeType: TypeAlias = builtins.str | device
DTypeLike:      TypeAlias = dtype | np.typing.DTypeLike
ArrayLike:      TypeAlias = np.typing.ArrayLike | list[Any] | tuple[Any, ...]

ShapeType: TypeAlias = (
    builtins.int
  | list [builtins.int]
  | tuple[builtins.int, ...]
  | Size
)

DimsType: TypeAlias = (
    builtins.int
  | list [builtins.int]
  | tuple[builtins.int, ...]
)

NumberType: TypeAlias = builtins.int | builtins.float 


