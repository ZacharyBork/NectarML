import functools
from typing import Any
from collections.abc import Callable

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float16, float32
from nectarml.amp.autocast import is_autocast_enabled, autocast_context

def _cast(
    precision: DTypeLike, 
    *args, 
    **kwargs
) -> tuple[tuple[Any], dict[str, Any]]:
    if is_autocast_enabled():
        if autocast_context() == 'cuda':
            args = list(args)
            
            for idx, arg in enumerate(args):
                if isinstance(arg, Tensor): 
                    if arg.device == 'cpu': continue
                    args[idx] = arg.to(dtype=precision)

            for key, value in kwargs.items():
                if isinstance(value, Tensor):
                    if value.device == 'cpu': continue
                    kwargs[key] = value.to(dtype=precision)
        
        else: pass # CPU autocast is no-op currently
    return args, kwargs

def amp_float16(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs): 
        args, kwargs = _cast(float16, *args, **kwargs)
        return func(*args, **kwargs)
    return wrapper
    
def amp_float32(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs): 
        args, kwargs = _cast(float32, *args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

    
