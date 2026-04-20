from __future__ import annotations
import builtins

from nectarml                 import typing
from nectarml.tensor          import Tensor
from nectarml.optim.optimizer import Optimizer
from nectarml.utils           import inspection

class GradScaler():
    def __init__(
        self:            GradScaler,
        scale:           builtins.float = 65536.0,
        backoff_factor:  builtins.float = 0.5,
        growth_factor:   builtins.float = 2.0,
        growth_interval: builtins.int   = 2000
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
        for param in optimizer.parameters():
            if param.grad is None: continue
            param.grad = param.grad.to(dtype=typing.float32) \
                       / self.scale_factor
        self._unscaled = True
            
    def step(self: GradScaler, optimizer: Optimizer) -> None:
        if not self._unscaled: self.unscale_(optimizer)
        
        self._skipped = False
        for param in optimizer.parameters():
            if param.grad is None: continue
            if inspection.has_inf(param.grad) \
            or inspection.has_nan(param.grad):
                self._skipped = True
                break
                
        if not self._skipped: optimizer.step()
        
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


