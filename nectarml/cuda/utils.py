from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import builtins
import ctypes
import ctypes.util
from   dataclasses import dataclass, field

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
    '''Returns a bool denoting whether CUDA is available on the host system.
    
    Returns:
        bool : True if CUDA is available, otherwise False.
    '''
    return get_cuda_info().available

### PYTHON-SIDE UTILS ###

def cuda_synchronize() -> None:
    '''Forces CUDA device synchronization.
    
    When called, this function disallows any new kernel launches until all
    tasks which are currently be processed have finished, forcing a sync
    for all threads on device.
    '''
    _nectarml.cuda_synchronize()

### CUDA-SIDE UTILS ###

def cast_tensor(input: Tensor, new_dtype: typing.dtype) -> builtins.int:
    '''Casts a CUDA tensor to a new DType.
    
    Args:
        input     : The tensor to cast.
        new_dtype : The DType to cast the tensor's data to.
        
    Returns:
        int : A pointer to the new data's location in device memory.
    '''
    return _nectarml.cast_tensor(
        input._data_ptr, input.size, input.dtype.cuda, new_dtype.cuda)

def cast_ptr(
    device_ptr:     builtins.int, 
    size:           builtins.int,
    original_dtype: typing.dtype,
    new_dtype:      typing.dtype
) -> builtins.int:
    '''Casts tensor data in CUDA memory to a new DType by pointer reference.
    
    Args:
        device_ptr     : The pointer to the address of the tensor data to cast 
                         in device memory.
        size           : The size (in number of elements) of the data to cast.
        original_dtype : The original (current) DType of the data to cast.
        new_dtype      : The DType to cast the data to.
        
    Returns:
        int : A pointer to the new data's location in device memory.
    '''
    original_dtype = original_dtype.cuda
    new_dtype      = new_dtype.cuda
    return _nectarml.cast_tensor(device_ptr, size, original_dtype, new_dtype)

def to_cuda(input: Tensor) -> builtins.int:
    '''Moves a tensor's data from CPU memory to device memory.
    
    Args:
        input : The tensor containing the data to move to device memory.

    Returns:
        int : A pointer to the new data's location in device memory.
    '''
    data = np.atleast_1d(input.data) if input.data.ndim == 0 else input.data
    if not data.flags['C_CONTIGUOUS']: data = np.ascontiguousarray(data)
    return _nectarml.to_cuda(data.ctypes.data, input.size, input.dtype.cuda)

def data_to_cuda(
    data:  np.ndarray, 
    size:  builtins.int, 
    dtype: typing.dtype
) -> builtins.int:
    '''Copies numpy data in host memory directly to CUDA.
    
    Args:
        data  : The numpy array data to copy to device memory.
        size  : The size (in number of elements) of the data to copy.
        dtype : The DType (nectarml.typing.dtype) of the data to copy.
    
    Returns:
        int : A pointer to the data's location in device memory.
    '''
    return _nectarml.to_cuda(data.ctypes.data, size, dtype.cuda)

def to_cpu(
    input:      Tensor, 
    host_dtype: typing.dtype | None = None
) -> np.ndarray:
    '''Copies tensor data from device memory back to host memory.
    
    Args:
        input      : The CUDA tensor who's data should be copied to host.
        host_dtype : The desired return DType of the numpy data, or None to use
                     the input tensor's DType.
    
    Returns:
        np.ndarray : The numpy array data in CPU memory.
    '''
    cast_dtype = host_dtype or input.dtype
    data = _nectarml.to_cpu(
        input._data_ptr, [int(i) for i in input.shape], input.dtype.cuda)
    return data.astype(cast_dtype.cpu)

def ptr_to_cpu(
    device_ptr:  np.ndarray, 
    shape:       typing.Size,
    host_dtype:  typing.dtype
) -> np.ndarray:
    '''Copies tensor data from CUDA memory to CPU memory by pointer reference.
    
    Args:
        device_ptr : The integer pointer to the address of the data in device
                     memory to copy to CPU.
        shape      : The shape (nectarml.typing.Size) of the data to copy.
        host_dtype : The DType of the data to copy.

    Returns:
        np.ndarray : The numpy array data in CPU memory.
    '''
    data = _nectarml.to_cpu(device_ptr, shape, host_dtype.cuda)
    return data.astype(host_dtype.cpu)

def inspect_cuda_data(
    device_ptr: builtins.int, 
    dtype:      typing.dtype,
    shape:      typing.Size,
    precision:  builtins.int = 4
) -> str:
    '''Returns a formatted string containing the data of a CUDA tensor.
    
    Args:
        device_ptr : The integer pointer to the address of the data in device
                     memory to inspect.
        dtype      : The DType of the data to inspect.
        shape      : The shape (nectarml.typing.Size) of the data to inspect.
        precision  : The decimal preision of the tensor data to use when
                     formatting the string.
                     
    Returns:
        string : The formatted string containing the CUDA tensor's data.
    '''
    data = _nectarml.to_cpu(device_ptr, [int(i) for i in shape], dtype.cuda)
    return np.array2string(data, separator=', ', precision=precision)

def clone(input: Tensor) -> builtins.int:
    '''Clones a given tensor in CUDA memory.
    
    Args:
        input : The tensor to clone.
        
    Returns:
        int : The pointer to the new tensor data in CUDA memory.
    '''
    return _nectarml.clone(input._data_ptr, input.size, input.dtype.cuda)

def clone_ptr(
    device_ptr: builtins.int,
    size:       builtins.int,
    dtype:      typing.dtype
) -> builtins.int:
    '''Clones a given tensor in CUDA memory by pointer reference.
    
    Args:
        device_ptr : The integer pointer to the address of the tensor data in
                     CUDA memory to clone.
        size       : The size (in number of elements) of the data to clone.
        dtype      : The DType of the data to clone.
        
    Returns:
        int : The pointer to the new tensor data in CUDA memory.
    '''
    return _nectarml.clone(device_ptr, size, dtype.cuda)

### INSPECTION UTILS ###

def compute_tensor_min(input: Tensor) -> builtins.float | builtins.int:
    '''Computes the minimum value of a tensor in CUDA memory.
    
    Args:
        input : The tensor to compute the min for.
        
    Returns:
        float : The minimum value found in the given tensor.
    '''
    return _nectarml.compute_tensor_min(
        input._data_ptr, input.size, input.dtype.cuda)

def compute_tensor_max(input: Tensor) -> builtins.float | builtins.int:
    '''Computes the maximum value of a tensor in CUDA memory.
    
    Args:
        input : The tensor to compute the max for.
        
    Returns:
        float : The maximum value found in the given tensor.
    '''
    return _nectarml.compute_tensor_max(
        input._data_ptr, input.size, input.dtype.cuda)

def compute_tensor_range(
    input: Tensor
) -> tuple[builtins.float | builtins.int, builtins.float | builtins.int]:
    '''Computes the minimum and maximum value of a tensor in CUDA memory.
    
    Args:
        input : The tensor to compute the min and max for.
        
    Returns:
        tuple : A tuple containing the minimum and maximum values found in the
                given tensor.
    '''
    return tuple(_nectarml.compute_tensor_range(
        input._data_ptr, input.size, input.dtype.cuda))
    
def is_inf(input: Tensor) -> builtins.bool:
    '''Checks whether a given CUDA tensor's values are all infinite.
    
    Args:
        input : The tensor to check.
        
    Returns:
        tuple : True if all values are infinite, otherwise False.
    '''
    return _nectarml.is_inf(
        input._data_ptr, input.size, input.dtype.cuda)
    
def is_finite(input: Tensor) -> builtins.bool:
    '''Checks whether a given CUDA tensor's values are all finite.
    
    Args:
        input : The tensor to check.
        
    Returns:
        tuple : True if all values are finite, otherwise False.
    '''
    return _nectarml.is_finite(
        input._data_ptr, input.size, input.dtype.cuda)
    
def is_nan(input: Tensor) -> builtins.bool:
    '''Checks whether a given CUDA tensor's values are all not a number.
    
    Args:
        input : The tensor to check.
        
    Returns:
        tuple : True if all values are not a number, otherwise False.
    '''
    return _nectarml.is_nan(
        input._data_ptr, input.size, input.dtype.cuda)
    
def has_inf(input: Tensor) -> builtins.bool:
    '''Checks whether any value in a given CUDA tensor is infinite.
    
    Args:
        input : The tensor to check.
        
    Returns:
        tuple : True if any value is infinite, otherwise False.
    '''
    return _nectarml.has_inf(
        input._data_ptr, input.size, input.dtype.cuda)
    
def has_nan(input: Tensor) -> builtins.bool:
    '''Checks whether any value in a given CUDA tensor is not a number.
    
    Args:
        input : The tensor to check.
        
    Returns:
        tuple : True if any value is not a number, otherwise False.
    '''
    return _nectarml.has_nan(
        input._data_ptr, input.size, input.dtype.cuda)


