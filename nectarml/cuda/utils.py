from typing import Any

import numpy as np

import _nectarml
from nectarml.cuda.mapping import DTYPE_MAP
from nectarml.typing import DTypeLike, float16, float32, half

def map_dtype(dtype: DTypeLike) -> Any:
    return DTYPE_MAP[dtype]

def to_cuda(data: np.ndarray, dtype: DTypeLike) -> int:
    c_data = data.astype(dtype)
    return _nectarml.to_cuda(c_data.ctypes.data, c_data.size, map_dtype(dtype))

def to_cpu(
    device_ptr: int, 
    shape: tuple[int, ...], 
    device_dtype: DTypeLike,
    host_dtype: DTypeLike | None = None
) -> np.ndarray:
    cast_dtype = host_dtype or device_dtype
    data = _nectarml.to_cpu(device_ptr, list(shape), map_dtype(device_dtype))
    return data.astype(cast_dtype)

def cast_tensor(
    device_ptr: int, 
    size: int, 
    old_dtype: DTypeLike, 
    new_dtype: DTypeLike
) -> int:
    return _nectarml.cast_tensor(
        device_ptr, size, map_dtype(old_dtype), map_dtype(new_dtype))

