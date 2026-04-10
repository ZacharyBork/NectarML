import functools
from typing import Any, ParamSpec, TypeVar
from collections.abc import Callable

from nectarml.typing import DTypeLike, float16, float32
from nectarml.amp.autocast import autocast_state

P = ParamSpec('P')
R = TypeVar('R')

DTYPE_RANK = {
    float32: 2, 
    float16: 1
}

### UTILS ###

def _collect_dtypes(args, kwargs) -> list[DTypeLike]:
    dtypes = []
    for item in list(args) + list(kwargs.values()):
        if getattr(item, '_class_type_nectar_tensor', False):
            dtypes.append(item.dtype)
        elif isinstance(item, (list, tuple)):
            dtypes += [
                t.dtype for t in item
                if getattr(t, '_class_type_nectar_tensor', False)
            ]
    return dtypes

def _cast(
    precision: DTypeLike, 
    *args, 
    **kwargs
) -> tuple[tuple[Any], dict[str, Any]]:
    # NOTE: CPU autocast is no-op currently
    enabled, context = autocast_state()
    if not enabled or context != 'cuda':
        return args, kwargs
    
    def cast_item(item):
        if getattr(item, '_class_type_nectar_tensor', False) \
        and item.dtype != precision and item.device != 'cpu':
            return item.to(dtype=precision)
        elif isinstance(item, (list, tuple)):
            casted = (cast_item(t) for t in item)
            return type(item)(casted)
        return item

    args = tuple(cast_item(arg) for arg in args)
    kwargs = {k: cast_item(v) for k, v in kwargs.items()}
    return args, kwargs

### DECORATORS

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

def amp_promote(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        enabled, context = autocast_state()
        if not enabled or context != 'cuda': return func(*args, **kwargs)

        dtypes = _collect_dtypes(args, kwargs)
        if not dtypes: return func(*args, **kwargs)

        target = max(dtypes, key=lambda d: DTYPE_RANK[d])
        args, kwargs = _cast(target, *args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

    
