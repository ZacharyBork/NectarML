from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import builtins
import ctypes
import ctypes.util
from dataclasses import dataclass, field

import numpy as np

import _nectarml
from nectarml import typing


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

### CUDA-SIDE UTILS ###

def cast_tensor(input: Tensor, new_dtype: typing.dtype) -> builtins.int:
    return _nectarml.cast_tensor(
        input._data_ptr, input.size, input.dtype.cuda, new_dtype.cuda)

def cast_tensor_by_reference(
    device_ptr:     builtins.int, 
    size:           builtins.int,
    original_dtype: typing.dtype,
    new_dtype:      typing.dtype
) -> builtins.int:
    original_dtype = original_dtype.cuda
    new_dtype      = new_dtype.cuda
    return _nectarml.cast_tensor(device_ptr, size, original_dtype, new_dtype)

def to_cuda(input: Tensor) -> builtins.int:
    data = np.atleast_1d(input.data) if input.data.ndim == 0 else input.data
    if not data.flags['C_CONTIGUOUS']: data = np.ascontiguousarray(data)
    return _nectarml.to_cuda(data.ctypes.data, input.size, input.dtype.cuda)

def data_to_cuda(
    data:  np.ndarray, 
    size:  builtins.int, 
    dtype: typing.dtype
) -> builtins.int:
    return _nectarml.to_cuda(data.ctypes.data, size, dtype.cuda)

def to_cpu(
    input: Tensor, 
    host_dtype: typing.dtype | None = None
) -> np.ndarray:
    cast_dtype = host_dtype or input.dtype
    data = _nectarml.to_cpu(
        input._data_ptr, [int(i) for i in input.shape], input.dtype.cuda)
    return data.astype(cast_dtype.cpu)

def inspect_cuda_data(
    device_ptr: builtins.int, 
    dtype:      typing.dtype,
    shape:      typing.Size,
    precision:  builtins.int = 4
) -> str:
    data = _nectarml.to_cpu(device_ptr, [int(i) for i in shape], dtype.cuda)
    return np.array2string(data, separator=', ', precision=precision)

def clone(input: Tensor) -> builtins.int:
    return _nectarml.clone(input._data_ptr, input.size, input.dtype.cuda)
    
def compute_tensor_min(input: Tensor) -> builtins.float:
    return _nectarml.compute_tensor_min(
        input._data_ptr, input.size, input.dtype.cuda)

def compute_tensor_max(input: Tensor) -> builtins.float:
    return _nectarml.compute_tensor_max(
        input._data_ptr, input.size, input.dtype.cuda)

def compute_tensor_range(input: Tensor) -> list[builtins.float]:
    return _nectarml.compute_tensor_range(
        input._data_ptr, input.size, input.dtype.cuda)
    
### INSPECTION UTILS ###
    
def is_inf(input: Tensor) -> builtins.bool:
    return _nectarml.is_inf(
        input._data_ptr, input.size, input.dtype.cuda)
    
def is_finite(input: Tensor) -> builtins.bool:
    return _nectarml.is_finite(
        input._data_ptr, input.size, input.dtype.cuda)
    
def is_nan(input: Tensor) -> builtins.bool:
    return _nectarml.is_nan(
        input._data_ptr, input.size, input.dtype.cuda)
    
def has_inf(input: Tensor) -> builtins.bool:
    return _nectarml.has_inf(
        input._data_ptr, input.size, input.dtype.cuda)
    
def has_nan(input: Tensor) -> builtins.bool:
    return _nectarml.has_nan(
        input._data_ptr, input.size, input.dtype.cuda)


