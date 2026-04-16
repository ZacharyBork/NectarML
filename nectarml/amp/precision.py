from typing import Any
from collections.abc import Callable

from nectarml import typing
from nectarml.amp.autocast import autocast_state
from nectarml.cuda import utils, memory

DTYPE_RANK = {typing.float16: 1, typing.float32: 2}

def _extract_ptr(
    arg:          Any, 
    target_dtype: typing.dtype
) -> tuple[int, bool]:
    if not getattr(arg, '_class_type_nectar_tensor', False): return arg, False
    if arg.device != 'cuda': return arg, False
    if arg.dtype == target_dtype: return arg._data_ptr, False
    
    ptr = utils.cast_tensor(arg, target_dtype.cuda)
    return ptr, True

def _free_temporaries(ptrs_and_flags: list[tuple[int, bool]]) -> None:
    for ptr, is_temp in ptrs_and_flags:
        if is_temp and ptr != 0: memory.free_cuda(ptr)

def _run_cast(
    func:         Callable[[Any], Any], 
    target_dtype: typing.dtype, 
    *args:        tuple[Any], 
    **kwargs:     dict[str, Any]
) -> Any:
    from nectarml.tensor import Tensor
    
    def cast_arg(arg: Any) -> Tensor | Any:
        if not getattr(arg, '_class_type_nectar_tensor', False): return arg
        if arg.device != 'cuda' or arg.dtype == target_dtype:    return arg
        ptr = utils.cast_tensor(arg, target_dtype)
        return Tensor._temporary(arg, ptr, target_dtype)
    
    cast_args   = [cast_arg(a) for a in args]
    cast_kwargs = {k: cast_arg(v) for k, v in kwargs.items()}
    result      = func(*cast_args, **cast_kwargs)
    return result

def run_cast_float16(
    func:     Callable[[Any], Any], 
    *args:    tuple[Any], 
    **kwargs: dict[str, Any]
) -> Any:
    state = autocast_state()
    if not state.enabled or state.context != 'cuda': 
        return func(*args, **kwargs)
    return _run_cast(func, typing.float16, *args, **kwargs)

def run_cast_float32(
    func:     Callable[[Any], Any], 
    *args:    tuple[Any], 
    **kwargs: dict[str, Any]
) -> Any:
    state = autocast_state()
    if not state.enabled or state.context != 'cuda': 
        return func(*args, **kwargs)
    return _run_cast(func, typing.float32, *args, **kwargs)

def run_cast_promote(
    func:     Callable[[Any], Any], 
    *args:    tuple[Any], 
    **kwargs: dict[str, Any]
) -> Any:
    state = autocast_state()
    if not state.enabled or state.context != 'cuda': 
        return func(*args, **kwargs)

    dtypes = [
        arg.dtype for arg in list(args) + list(kwargs.values())
        if getattr(arg, '_class_type_nectar_tensor', False)
        and arg.device == 'cuda'
        and arg.dtype in DTYPE_RANK
    ]
    if not dtypes: return func(*args, **kwargs)
    
    target = max(dtypes, key=lambda d: DTYPE_RANK.get(d, 0))
    return _run_cast(func, target, *args, **kwargs)

