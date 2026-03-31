from __future__ import annotations

from nectarml.tensor import Tensor
from nectarml.typing import float32
from nectarml.optim.optimizer import Optimizer
from nectarml.utils import inspection

class GradScaler():
    def __init__(
        self: GradScaler,
        scale:          float = 65536.0,
        backoff_factor: float = 0.5,
        growth_factor:  float = 2.0,
        growth_interval:  int = 2000
    ) -> None:
        self.scale_factor    = scale
        self.backoff_factor  = backoff_factor
        self.growth_factor   = growth_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        self._unscaled       = False
        self._skipped        = False
    
    def scale(self: GradScaler, loss: Tensor) -> Tensor:
        return loss * self.scale_factor
    
    def unscale_(self: GradScaler, optimizer: Optimizer) -> None:
        if self._unscaled: return
        for param in optimizer._get_all_params():
            if param.grad is None: continue
            param.grad = param.grad.to(dtype=float32) / self.scale_factor
        self._unscaled = True
    
    def step(self: GradScaler, optimizer: Optimizer) -> None:
        if not self._unscaled: self.unscale_(optimizer)
        
        found_bad = False
        for param in optimizer._get_all_params():
            if param.grad is None: continue
            if inspection.has_inf(param.grad) or \
               inspection.has_nan(param.grad):
                found_bad = True
                if found_bad: break

        if not found_bad: 
            self._skipped = False
            optimizer.step()
        else: self._skipped = True
        
    def update(self: GradScaler) -> None:
        self._unscaled = False
        if self._skipped:
            self.scale_factor *= self.backoff_factor            
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker == self.growth_interval:
                self.scale_factor *= self.growth_factor
                self._growth_tracker = 0


