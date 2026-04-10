import threading
from typing import Any, Self, Literal

threadLocal = threading.local()
threadLocal.autocast_enabled = False
threadLocal.autocast_context = None

def set_autocast_enabled(enabled: bool) -> None:
    threadLocal.autocast_enabled = enabled
def is_autocast_enabled() -> bool: return threadLocal.autocast_enabled
def autocast_context() -> str | None: return threadLocal.autocast_context
def autocast_state() -> tuple[bool, str | None]:
    return threadLocal.autocast_enabled, threadLocal.autocast_context

class autocast:
    def __init__(
        self, 
        context: Literal['cpu', 'cuda'], 
        enabled: bool = True
    ) -> None:
        self.context = context
        self.enabled = enabled
    
    def __enter__(self) -> Self:
        if self.enabled:
            set_autocast_enabled(True)
            threadLocal.autocast_context = self.context
    
    def __exit__(self, *args: Any) -> None:
        if self.enabled:
            set_autocast_enabled(False)
            threadLocal.autocast_context = None

