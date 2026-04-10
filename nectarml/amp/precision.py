import functools
from typing import Any, ParamSpec, TypeVar
from collections.abc import Callable

from nectarml.typing import DTypeLike, float16, float32
from nectarml.amp.autocast import autocast_state

P = ParamSpec('P')
R = TypeVar('R')

def _cast(
    precision: DTypeLike, 
    *args, 
    **kwargs
) -> tuple[tuple[Any], dict[str, Any]]:
    # NOTE: CPU autocast is no-op currently
    enabled, context = autocast_state()
    if not enabled or context != 'cuda':
        return args, kwargs
        
    args = tuple(
        arg.to(dtype=precision) 
        if getattr(arg, '_class_type_nectar_tensor', False)
        and arg.dtype != precision and arg.device != 'cpu'
        else arg
        for arg in args
    )
    kwargs = {
        k: v.to(dtype=precision)
        if getattr(v, '_class_type_nectar_tensor', False)
        and v.dtype != precision and v.device != 'cpu'
        else v
        for k, v in kwargs.items()
    }
    return args, kwargs

def amp_float16(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        args, kwargs = _cast(float16, *args, **kwargs)
        return func(*args, **kwargs)
    return wrapper
    
def amp_float32(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        args, kwargs = _cast(float32, *args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

    
