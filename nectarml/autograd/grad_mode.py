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
    def __enter__(self: no_grad) -> Self:
        self._prev = _GRAD_STATE.enabled
        _GRAD_STATE.enabled = False
    
    def __exit__(self: no_grad, *args: Any) -> None:
        _GRAD_STATE.enabled = self._prev
        
    def __call__(self, func: Callable) -> Callable:
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)
        return wrapper

