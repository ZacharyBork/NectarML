from dataclasses import dataclass
from typing import Any, Self, Literal

@dataclass
class AutocastState:
    enabled: bool
    context: str | None

_STATE = AutocastState(enabled=False, context=None)

def autocast_state() -> AutocastState:    return _STATE
def is_autocast_enabled() -> bool:        return _STATE.enabled
def autocast_context() -> str | None:     return _STATE.context
def set_autocast_enabled(enabled: bool) -> None: _STATE.enabled = enabled
def set_autocast_context(context: str)  -> None: _STATE.context = context

class autocast:
    def __init__(
        self, 
        context: Literal['cpu', 'cuda'], 
        enabled: bool = True
    ) -> None:
        '''Enables an autocast context.

        Currently this functionality is pretty limited. The `cpu` context is a 
        no-op, it does nothing currently when enabled. The `cuda` context, when
        enabled, will automatically perform convolution and matrix 
        multiplication on cuda tensors in float16. And that's it right now. It
        will be expanded in the future, however.

        Args:
            context : The context to enable. Options are [`cpu`, `cuda`].
            enabled : Turns contexts into a no-op. Convenience flag to disable
                      context block without having to delete it.
        '''
        self.context = context
        self.enabled = enabled
    
    def __enter__(self) -> Self:
        global _STATE
        if self.enabled:
            _STATE.enabled = True
            _STATE.context = self.context
        return self
    
    def __exit__(self, *args: Any) -> None:
        global _STATE
        if self.enabled:
            _STATE.enabled = False
            _STATE.context = None

