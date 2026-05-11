from __future__ import annotations

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
    
    def __repr__(self: finfo) -> builtins.str:
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
    
    def __repr__(self: finfo) -> builtins.str:
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
    '''Helper class to map numpy DTypes to NectarML host DTypes.'''
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
    
    def __new__(cls: type[Self], type_cpu: np.typing.DTypeLike = None) -> Self:
        '''Checks in new DType exists in cache and uses cached instead if so.

        Avoids unnecessary creation of duplicate DTypes while also allowing for
        new mechanisms of DType assignment. This method just checks upon 
        creation of a new `dtype` object whether the `_DTYPE_CACHE` already
        contains a `dtype` object with the corresponding `type_cpu`. If so, it 
        retrieves the cached version and returns a reference to that instead, 
        rather than creating a new `dtype` object. If not, it creates a new 
        `dtype` object for the given `type_cpu`, adds it to the `_DTYPE_CACHE`,
        the returns a reference to the newly created/cached `dtype`.

        Args:
            type_cpu : The np.typing.DTypeLike object to serve as the new 
                       `dtype` object's CPU DType. CUDA DType is mapped 
                       automatically from this.
                    
        Returns:
            dtype : The new (or cached) dtype object.
        '''
        if type_cpu is None:         return object.__new__(cls)
        if type_cpu in _DTYPE_CACHE: return _DTYPE_CACHE[type_cpu]
        
        obj = object.__new__(cls)
        _DTYPE_CACHE[type_cpu] = obj
        return obj
    
    def __init__(self: dtype, type_cpu: np.typing.DTypeLike) -> None:
        '''Initializes a new dtype object.

        Args:
            type_cpu : The np.typing.DTypeLike object to serve as the new 
                       `dtype` object's CPU DType. CUDA DType is mapped 
                       automatically from this.
        '''
        self._type_cpu         = type_cpu
        self.is_floating_point = np.issubdtype(type_cpu, np.floating)
        self.is_complex        = np.issubdtype(type_cpu, np.complexfloating)
        self.is_signed         = np.issubdtype(type_cpu, np.signedinteger)
        self.itemsize          = np.dtype(type_cpu).itemsize
        
        self.info = self._build_info()
    
    # INIT
    
    def _build_info(self: dtype) -> None:
        '''Builds `iinfo` or `finfo` object for new dtype.'''
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
        '''Returns the corresponding CPU/numpy DType for the `dtype` object.

        Returns:
            np.typeing.DTypeLike : The numpy DType for the given `dtype`.
        '''
        return self._type_cpu
    
    @property
    def cpu (self: dtype) -> np.typing.DTypeLike: 
        '''Returns the corresponding CPU/numpy DType for the `dtype` object.

        Returns:
            np.typeing.DTypeLike : The numpy DType for the given `dtype`.
        '''
        return self._type_cpu
    
    @property
    def cuda(self: dtype) -> _nectarml.DType:
        '''Returns the CUDA DType for the given `dtype` object.

        Returns:
            _nectarml.DType : The nectarml.DType which corresponds to the 
                              datatype for the given `dtype` object. 
                              _nectarml.DType is an enum is the host layer. 
                              This is the DType reference expected by C++ ops 
                              and CUDA kernels. See here for more info:
                                - `nectarml/csrc/include/common/dtype.h`
        '''
        return self._map.map_dtype(self.cpu)
    
    @property
    def name(self: dtype) -> builtins.str:
        '''Returns the name of the given DType as a string.

        Example:
            `nectarml.float32.name`: `"float32"`
        
        Returns:
            str : The name of the `dtype` as a string.
        '''
        return self._type_cpu.__name__

    # COMPARISON
    
    def __hash__(self: dtype) -> builtins.int: 
        '''Dtype hashing method.

        This method just returns the hash of the underlying numpy DType object 
        for the given `dtype`. Since each `dtype` can only ever own one unique 
        numpy DType (due to the __new__ behavior of the `dtype` class), each 
        `dtype` just inherits the hash id of said DType to keep things simple.

        Returns:
            int : The hash of the numpy DType object for the given `dtype`.
        '''
        return hash(self._type_cpu)

    def __eq__(self: dtype, other: dtype) -> builtins.bool:
        '''Checks whether two DTypes are the same.

        Checks whether this `dtype` and the other `dtype` have the same 
        underlying numpy DType. Every `dtype` should only ever own one numpy
        DType which is unique to itself (due to the __new__ behavior of the 
        `dtype` class), so doing it this way is not strictly necessary. But 
        this avoids issues should a new `dtype` ever get created with an 
        underlying numpy DType which is the same as another `dtype` object for
        whatever reason.

        Args:
            dtype : The other `dtype` to compare the given `dtype` against.
            
        Returns:
            bool : True if the two `dtype`s are the same, otherwise False.
        '''
        if isinstance(other, dtype):
            return self._type_cpu == other._type_cpu
        return NotImplemented
    
    def __ne__(self: dtype, other: dtype) -> builtins.bool:
        '''Checks whether two DTypes are not the same.

        Checks whether this `dtype` and the other `dtype` do not have the same 
        underlying numpy DType. Every `dtype` should only ever own one numpy
        DType which is unique to itself (due to the __new__ behavior of the 
        `dtype` class), so doing it this way is not strictly necessary. But 
        this avoids issues should a new `dtype` ever get created with an 
        underlying numpy DType which is the same as another `dtype` object for
        whatever reason.

        Args:
            dtype : The other `dtype` to compare the given `dtype` against.
            
        Returns:
            bool : True if the two `dtype`s are not the same, otherwise False.
        '''
        if isinstance(other, dtype):
            return self._type_cpu != other._type_cpu
        return NotImplemented

    # INSPECTION
    
    def __str__   (self: dtype) -> str:  return f'nectarml.{self.name}'
    def __repr__  (self: dtype) -> str:  return self.__str__()
    def __reduce__(self: dtype) -> None: return (dtype, (self._type_cpu,))

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

