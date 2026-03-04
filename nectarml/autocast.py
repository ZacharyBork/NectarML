import threading
from typing import Type, Self, Literal
from types import TracebackType

threadLocal = threading.local()
threadLocal.autocast_enabled = False
threadLocal.autocast_context = None

def set_autocast_enabled(enabled: bool) -> None: 
    threadLocal.autocast_enabled = enabled

def is_autocast_enabled() -> bool: return threadLocal.autocast_enabled

def autocast_context() -> str | None: return threadLocal.autocast_context

class autocast:
    def __init__(self, context: Literal['cpu', 'cuda']) -> None:
        self.context = context
    
    def __enter__(self) -> Self:
        set_autocast_enabled(True)
        threadLocal.autocast_context = self.context
    
    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None
    ) -> None:
        set_autocast_enabled(False)
        threadLocal.autocast_context = None

