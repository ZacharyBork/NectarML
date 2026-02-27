from typing import Any, Literal
from collections.abc import Callable

from nectarml import Tensor
from nectarml.optim.optimizer import Optimizer

class Scheduler():
    def __init__(self, optimizer: Optimizer, last_epoch: int=-1) -> None:
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        
    def get_last_lr(self) -> list[float | Tensor]:
        pass
    
    def get_lr(self) -> list[float | Tensor]:
        pass

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass
    
    def state_dict(self) -> dict[str, Any]:
        pass

    def step(self, epoch: int | None = None) -> None:
        pass

class StepLR(Scheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        step_size: int,
        gamma: float = 0.01,
        last_epoch: int=-1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.step_size = step_size
        self.gamma = gamma

class MultiStepLR(Scheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        milestones: list[int],
        gamma: float = 0.01,
        last_epoch: int=-1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.milestones = milestones
        self.gamma = gamma

class ConstantLR(Scheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        factor: float = 0.3333333333333333,
        total_iters: int = 5,
        last_epoch: int=-1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.factor = factor
        self.total_iters = total_iters

class LinearLR(Scheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        start_factor: float = 0.3333333333333333,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int=-1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        
class ExponentialLR(Scheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        gamma: float,
        last_epoch: int=-1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.gamma = gamma
        
class PolynomialLR(Scheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        total_iters: int = 5,
        power: float = 1.0,
        last_epoch: int=-1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.total_iters = total_iters
        self.power = power
        
class CosineAnnealingLR(Scheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int=-1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.T_max = T_max
        self.eta_min = eta_min
        
class CosineAnnealingWarmRestarts(Scheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int=-1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        
class ReduceLROnPlateau(Scheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        mode: Literal['min', 'max'] = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 0.0001,
        threshold_mode: Literal['rel', 'abs'] = 'rel',
        cooldown: int = 0,
        eps: float = 1e-8,
        last_epoch: int=-1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.eps = eps
        
class CyclicLR(Scheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        step_size_down: int | None = None,
        mode: Literal['triangular', 'triangular2', 'exp_range'] = 'triangular',
        gamma: float | None = None,
        scale_fn: Callable | None = None,
        scale_mode: Literal['cycle', 'iterations'] = 'cycle',
        cycle_momentum: bool = True,
        base_momentum: float | list[float] = 0.8,
        max_momentum: float | list[float] = 0.9,
        last_epoch: int=-1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        
class OneCycleLR(Scheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        max_lr: float | list[float],
        total_steps: int | None = None,
        epochs: int | None = None,
        steps_per_epoch: int | None = None,
        pct_start: float = 0.3,
        anneal_strategy: Literal['cos', 'linear'] = 'cos',
        cycle_momentum: bool = True,
        base_momentum: float | list[float] = 0.85,
        max_momentum: float | list[float] = 0.95,
        div_factor: float = 25.0,
        three_phase: bool = True,
        last_epoch: int=-1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.div_factor = div_factor
        self.three_phase = three_phase
        
        
        
        
        
