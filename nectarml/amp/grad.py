from __future__ import annotations
import builtins

from nectarml                 import typing
from nectarml.core            import Tensor
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
        '''Initializes a GradScaler instance.

        ### Description

        GradScalers are used to improve stability in half and mixed precision
        training. They achieve this by:

        - Scaling the model's loss each timestep by their current scale factor
          before backward is called on it.
        - Re-normalizing an optimizers parameter values after backpropagation 
          by dividing them by the GradScaler's current scale factor.
        - Checking each parameter after unscaling for infinite or NaN values.

        If bad values are found, the optimizer step is skipped, and the scale 
        factor of the GradScaler is scaled by its backoff factor to make the
        next scale/unscale operation less extreme.

        Otherwise:
        - The optimizer is stepped. 
        - The GradScalers internal growth tracker is checked.
        - If the value of the growth tracker is lower than the 
          `growth_interval`, the growth tracker is increased by 1.
        - If it has reached the `growth_interval`:
            - The GradScaler's scale factor is multiplied by its 
              `growth_factor`.
            - The growth tracker is reset.

        This stops corrupted gradients caused by precision overflows from 
        reaching the optimizer and acts as a sort of adaptive gating mechanism 
        which changes its behaviour in relation to the model's output over 
        time.

        ### Use
        The general use pattern for a GradScaler is as follows:
        ```
        with nectarml.amp.autocast('cuda'):
            # Model forward pass...
        
        optimizer.zero_grad()         # Zero the optimizers gradients
        
        scaler.scale(loss).backward() # Scale the loss, then call backward
        scaler.step(optimizer)        # Step optimizer (if no bad params found)
        scaler.update()               # Update GradScaler
        ```

        Args:
            scale           : The initial scale factor for the GradScaler.
            backoff_factor  : The value to multiply the GradScaler's scale 
                              factor by when updateing the grad scaler on 
                              timesteps where bad gradients are found.
            growth_factor   : The value to multiply the GradScaler's scale 
                              factor by when updateing the grad scaler on 
                              timesteps where no bad gradients are found, and 
                              the GradScalers growth tracker has reached the 
                              `growth_interval`.
            growth_interval : The interval (by number of GradScaler.update() 
                              calls) at which to scale the GradScaler's scale
                              factor by its growth factor.
            max_scale       : The maximum allowable scale factor for the
                              GradScaler. Scale factor will be clamped at this
                              value at the high end.
        '''
        self.scale_factor    = scale
        self.backoff_factor  = backoff_factor
        self.growth_factor   = growth_factor
        self.growth_interval = growth_interval
        self.max_scale       = max_scale
        
        self._growth_tracker = 0
        self._unscaled       = False
        self._found_bad      = False
    
    def scale(self: GradScaler, loss: Tensor) -> Tensor:
        '''Scales a loss tensor by the GradScaler's current scale factor.
        
        Args:
            loss : The loss tensor to scale.
            
        Returns:
            Tensor : The scaled loss tensor.
        '''
        return loss * self.scale_factor

    def unscale_(self: GradScaler, optimizer: Optimizer) -> None:
        '''Unscales an optimizer's gradients and checks for NaN/inf.
        
        Args:
            optimizer : The optimizer to unscale and check.
        '''
        if self._unscaled: return
        self._found_bad  = False
        inv_scale        = 1.0 / (float(self.scale_factor) + 1e-8)
        
        for param in optimizer.params():
            if param.grad is None: continue
            if param.device == 'cuda':
                self._found_bad = cuda_unscale_(param.grad, inv_scale)
            else:
                param.grad = param.grad.to(dtype=typing.float32) * inv_scale
                self._found_bad  = inspection.has_inf(param.grad) \
                                or inspection.has_nan(param.grad)
            if self._found_bad: break

        self._unscaled = True
            
    def step(self: GradScaler, optimizer: Optimizer) -> None:
        '''Conditionally steps optimizer based on gradient validity.

        If unscale has not been called on the gradscale before this, it will 
        be automatically called here to unscale the gradients and check for bad
        values. If bad values are found, the optimizer step will be skipped,
        otherwise the optimizer will be stepped.
        
        Args:
            optimizer : The optimizer to step.
        '''
        self.unscale_(optimizer)
        if not self._found_bad: optimizer.step()
        
    def update(self: GradScaler) -> None:
        '''Updates the GradScaler's growth and tracking variables.

        Should be called after each call to `GradScaler.step()`.      
        '''
        self._unscaled = False
        
        if self._found_bad:
            self._growth_tracker = 0
            self.scale_factor   *= self.backoff_factor
        else:
            self._growth_tracker += 1
            if self._growth_tracker == self.growth_interval:
                self.scale_factor = min(
                    self.growth_factor * self.scale_factor, self.max_scale)
                self._growth_tracker = 0
        
        self._found_bad = False


