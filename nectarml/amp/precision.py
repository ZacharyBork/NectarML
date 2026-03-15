import functools
from typing import Any
from collections.abc import Callable

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float16, float32
from nectarml.amp.autocast import is_autocast_enabled, autocast_context


# ORDER OF OPERATIONS && PER-STEP PRECISION
# -----------------------------------------------------------------------------
# forward pass ----------> float16 ops via autocast
# loss computed ---------> still float16
# loss * scale_factor ---> still float16 but larger magnitude
# backward() ------------> gradients in float16 but in representable range
# scaler.step() ---------> unscale to float32, check for inf/NaN, step or skip
# scaler.update() -------> adjust scale factor up or down
# optimizer.zero_grad() -> reset for next iteration

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
    def wrapper(*args, **kwargs): return _cast(float32, *args, **kwargs)
    return wrapper

    
