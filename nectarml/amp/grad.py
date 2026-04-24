from __future__ import annotations
import builtins

from nectarml                 import typing
from nectarml.core          import Tensor
from nectarml.optim.optimizer import Optimizer
from nectarml.utils           import inspection
from nectarml.cuda.amp        import unscale_and_check_grad as cuda_unscale_

class GradScaler:
    def __init__(
        self:            GradScaler,
        scale:           builtins.float = 65536.0,
        backoff_factor:  builtins.float = 0.5,
        growth_factor:   builtins.float = 2.0,
        growth_interval: builtins.int   = 2000,
        max_scale:       builtins.float = 2**24
    ) -> None:
        self.scale_factor    = scale
        self.backoff_factor  = backoff_factor
        self.growth_factor   = growth_factor
        self.growth_interval = growth_interval
        self.max_scale       = max_scale
        
        self._growth_tracker = 0
        self._unscaled       = False
        self._found_bad      = False
    
    def scale(self: GradScaler, loss: Tensor) -> Tensor:
        return loss * self.scale_factor

    def unscale_(self: GradScaler, optimizer: Optimizer) -> None:
        if self._unscaled: return
        self._found_bad  = False
        inv_scale        = 1.0 / (float(self.scale_factor) + 1e-8)
        
        for param in optimizer.params():
            if param.grad is None: continue
            if param.device == 'cuda':
                self._found_bad = cuda_unscale_(param.grad, inv_scale)
            else:
                param.grad = param.grad.to(dtype=typing.float32) * inv_scale
                self._found_bad = inspection.has_inf(param.grad) \
                               or inspection.has_nan(param.grad)
        
        self._unscaled = True
            
    def step(self: GradScaler, optimizer: Optimizer) -> None:
        if not self._unscaled: self.unscale_(optimizer)
        if not self._found_bad: optimizer.step()
        
    def update(self: GradScaler) -> None:
        self._unscaled = False
        
        if self._found_bad:
            self.scale_factor *= self.backoff_factor            
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker == self.growth_interval:
                self.scale_factor = min(
                    self.growth_factor * self.scale_factor, self.max_scale)
                self._growth_tracker = 0
        
        self._found_bad = False


