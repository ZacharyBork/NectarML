from __future__ import annotations

from typing          import Self, Any
from collections.abc import Callable
from dataclasses     import dataclass

@dataclass
class GradState:
    enabled: bool
    
_GRAD_STATE = GradState(enabled=True)

def is_grad_enabled() -> bool: return _GRAD_STATE.enabled
def set_grad_enabled(enabled: bool) -> None: _GRAD_STATE.enabled = enabled

class no_grad:
    '''Context to disable grad.

    Any tensor created in this context will have `requires_grad` automatically
    set to False, disabling grad allocation during backpropagation.
    '''
    def __enter__(self: no_grad) -> Self:
        self._prev = _GRAD_STATE.enabled
        _GRAD_STATE.enabled = False
        return self
    
    def __exit__(self: no_grad, *args: Any) -> None:
        _GRAD_STATE.enabled = self._prev
        
    def __call__(self, func: Callable) -> Callable:
        '''Allows for function wrapping as a decorator.

        Any function with this decorator applied will be treated as though it
        were run in a no-grad context.
        '''
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)
        return wrapper

