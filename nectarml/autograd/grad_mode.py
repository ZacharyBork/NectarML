from __future__ import annotations

import threading
from typing import Self
from collections.abc import Callable

_grad_state = threading.local()

def is_grad_enabled() -> bool:
    return getattr(_grad_state, 'enabled', True)

def set_grad_enabled(enabled: bool) -> None:
    _grad_state.enabled = enabled

class no_grad:
    def __enter__(self: no_grad) -> Self:
        self._prev = is_grad_enabled()
        set_grad_enabled(False)
    
    def __exit__(self: no_grad, *args) -> None:
        set_grad_enabled(self._prev)
        
    def __call__(self, func: Callable) -> Callable:
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)
        return wrapper

