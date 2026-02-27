from typing import Any

from nectarml import Tensor

class Optimizer():
    def __init__(
        self,
        parameters: list[Tensor] | list[dict[str, Any]],
        defaults: dict[str, Any]
    ) -> None:
        pass
    
    def zero_grad(self) -> None:
        pass
    
    def step(self) -> None:
        pass
    
    def state_dict(self) -> None:
        pass
    
    def load_state_dict(self) -> None:
        pass


