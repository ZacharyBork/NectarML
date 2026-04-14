import threading
from dataclasses import dataclass
from typing import Any, Self, Literal

@dataclass
class AutocastState:
    enabled: bool
    context: str | None

tl = threading.local()
tl.autocast = AutocastState(enabled=False, context=None)

def autocast_state() -> AutocastState:    return tl.autocast
def is_autocast_enabled() -> bool:        return tl.autocast.enabled
def autocast_context() -> str | None:     return tl.autocast.context
def set_autocast_enabled(enabled: bool) -> None: tl.autocast.enabled = enabled
def set_autocast_context(context: str)  -> None: tl.autocast.context = context

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
            tl.autocast_state.context = self.context
    
    def __exit__(self, *args: Any) -> None:
        if self.enabled:
            set_autocast_enabled(False)
            tl.autocast_state.context = None

