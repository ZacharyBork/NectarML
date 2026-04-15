from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import builtins
import ctypes
import ctypes.util
from typing import Any
from dataclasses import dataclass, field

import numpy as np

import _nectarml
from nectarml.cuda.mapping import DTYPE_MAP
from nectarml.typing import DTypeLike

### CUDA SYSTEM UTILS ###

@dataclass
class CUDAInfo:
    available:       builtins.bool
    driver_version:  builtins.str | None = None
    runtime_version: builtins.str | None = None
    device_count:    builtins.int | None = None
    devices:         builtins.list[builtins.str] = field(
                        default_factory=builtins.list)
    
    def __repr__(self: CUDAInfo) -> str:
        return (
            f'CUDAInfo(\n'
            f'    CUDA Available:  {self.available}\n'
            f'    Driver Version:  {self.driver_version}\n'
            f'    Runtime Version: {self.runtime_version}\n\n'
            f'    Device Count:    {self.device_count}\n'
            f'    Devices: [\n'
            f'        {'\n'.join([f'"{i}"' for i in self.devices])}\n'
            f'    ]\n'
            f')'
        )

def get_cuda_info() -> CUDAInfo:
    lib_name = ctypes.util.find_library('cudart')
    if lib_name is None: return None
    
    try: cuda = ctypes.CDLL(lib_name)
    except OSError: return CUDAInfo(available=False)

    runtime_ver = ctypes.c_int(0)
    cuda.cudaRuntimeGetVersion(ctypes.byref(runtime_ver))
    rv = runtime_ver.value
    runtime_str = f'{rv // 1000}.{(rv % 1000) // 10}'

    driver_ver = ctypes.c_int(0)
    cuda.cudaDriverGetVersion(ctypes.byref(driver_ver))
    dv = driver_ver.value
    driver_str = f'{dv // 1000}.{(dv % 1000) // 10}'

    count = ctypes.c_int(0)
    cuda.cudaGetDeviceCount(ctypes.byref(count))
    
    devices = []
    for i in range(count.value):
        prop = ctypes.create_string_buffer(4096)
        cuda.cudaGetDeviceProperties(prop, i)
        devices.append(prop.raw[:256].rstrip(b'\x00').decode('utf-8'))

    return CUDAInfo(
        available=True,
        driver_version=driver_str,
        runtime_version=runtime_str,
        device_count=count.value,
        devices=devices
    )

def is_cuda_available() -> builtins.bool:
    return get_cuda_info().available

### PYTHON-SIDE UTILS ###

def cuda_synchronize() -> None:
    _nectarml.cuda_synchronize()

def map_dtype(dtype: DTypeLike) -> Any:
    return DTYPE_MAP[dtype]

### CUDA-SIDE UTILS ###

def cast_tensor(input: Tensor, new_dtype: DTypeLike) -> builtins.int:
    original_dtype = map_dtype(input.dtype)
    new_dtype      = map_dtype(new_dtype)
    return _nectarml.cast_tensor(
        input._data_ptr, input.size, original_dtype, new_dtype)

def cast_tensor_by_reference(
    device_ptr:     builtins.int, 
    size:           builtins.int,
    original_dtype: DTypeLike,
    new_dtype:      DTypeLike
) -> builtins.int:
    original_dtype = map_dtype(original_dtype)
    new_dtype      = map_dtype(new_dtype)
    return _nectarml.cast_tensor(device_ptr, size, original_dtype, new_dtype)

def to_cuda(input: Tensor) -> builtins.int:
    cast_dtype = map_dtype(input.dtype)
    data = np.atleast_1d(input.data) if input.data.ndim == 0 else input.data
    if not data.flags['C_CONTIGUOUS']: data = np.ascontiguousarray(data)
    return _nectarml.to_cuda(data.ctypes.data, input.size, cast_dtype)

def data_to_cuda(
    data:  np.ndarray, 
    size:  builtins.int, 
    dtype: DTypeLike
) -> builtins.int:
    cast_dtype = map_dtype(dtype)
    ptr = _nectarml.to_cuda(data.ctypes.data, size, cast_dtype)
    return ptr

def to_cpu(input: Tensor, host_dtype: DTypeLike | None = None) -> np.ndarray:
    cast_dtype = host_dtype or input.dtype
    data = _nectarml.to_cpu(
        input._data_ptr, [int(i) for i in input.shape], map_dtype(input.dtype))
    return data.astype(cast_dtype)

def clone(input: Tensor) -> builtins.int:
    return _nectarml.clone(input._data_ptr, input.size, map_dtype(input.dtype))
    
def compute_tensor_min(input: Tensor) -> builtins.float:
    return _nectarml.compute_tensor_min(
        input._data_ptr, input.size, map_dtype(input.dtype))

def compute_tensor_max(input: Tensor) -> builtins.float:
    return _nectarml.compute_tensor_max(
        input._data_ptr, input.size, map_dtype(input.dtype))

def compute_tensor_range(input: Tensor) -> list[builtins.float]:
    return _nectarml.compute_tensor_range(
        input._data_ptr, input.size, map_dtype(input.dtype))
    
### INSPECTION UTILS ###
    
def is_inf(input: Tensor) -> builtins.bool:
    return _nectarml.is_inf(
        input._data_ptr, input.size, map_dtype(input.dtype))
    
def is_finite(input: Tensor) -> builtins.bool:
    return _nectarml.is_finite(
        input._data_ptr, input.size, map_dtype(input.dtype))
    
def is_nan(input: Tensor) -> builtins.bool:
    return _nectarml.is_nan(
        input._data_ptr, input.size, map_dtype(input.dtype))
    
def has_inf(input: Tensor) -> builtins.bool:
    return _nectarml.has_inf(
        input._data_ptr, input.size, map_dtype(input.dtype))
    
def has_nan(input: Tensor) -> builtins.bool:
    return _nectarml.has_nan(
        input._data_ptr, input.size, map_dtype(input.dtype))


