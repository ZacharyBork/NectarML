from __future__ import annotations

import math
from typing import Any, Literal
from collections.abc import Callable

from nectarml import Tensor
from nectarml.optim.optimizer import Optimizer

class Scheduler():
    def __init__(
        self: Scheduler, 
        optimizer: Optimizer, 
        last_epoch: int = -1
    ) -> None:
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self._last_lr = self.base_lrs.copy()
        self.step()
        
    def get_lr(self: Scheduler) -> list[float]:
        raise NotImplementedError

    def get_last_lr(self: Scheduler) -> list[float]:
        return self._last_lr

    def step(self: Scheduler, epoch: int | None = None) -> None:
        if epoch is None: self.last_epoch += 1
        else: self.last_epoch = epoch
        
        values = self.get_lr()
        self._last_lr = values
        
        for group, lr in zip(self.optimizer.param_groups, values):
            group['lr'] = lr

    def state_dict(self: Scheduler) -> dict[str, Any]:
        return { k: v for k, v in self.__dict__.items() if k != 'optimizer' }

    def load_state_dict(self: Scheduler, state_dict: dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

class StepLR(Scheduler):
    def __init__(
        self: StepLR, 
        optimizer: Optimizer, 
        step_size: int,
        gamma: float = 0.01,
        last_epoch: int = -1
    ) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: StepLR) -> list[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma 
                for group in self.optimizer.param_groups]

class MultiStepLR(Scheduler):
    def __init__(
        self: MultiStepLR, 
        optimizer: Optimizer, 
        milestones: list[int],
        gamma: float = 0.01,
        last_epoch: int = -1
    ) -> None:
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: MultiStepLR) -> list[float]:
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

class ConstantLR(Scheduler):
    def __init__(
        self: ConstantLR, 
        optimizer: Optimizer, 
        factor: float = 0.3333333333333333,
        total_iters: int = 5,
        last_epoch: int = -1
    ) -> None:
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: ConstantLR) -> list[float]:
        if self.last_epoch == self.total_iters:
            return [base_lr for base_lr in self.base_lrs]
        if self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.factor
                for group in self.optimizer.param_groups]

class LinearLR(Scheduler):
    def __init__(
        self: LinearLR, 
        optimizer: Optimizer, 
        start_factor: float = 0.3333333333333333,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1
    ) -> None:
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: LinearLR) -> list[float]:
        if self.last_epoch >= self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
        
        t = self.last_epoch / self.total_iters
        factor = self.start_factor + (self.end_factor - self.start_factor) * t
        return [base_lr * factor for base_lr in self.base_lrs]
        
class ExponentialLR(Scheduler):
    def __init__(
        self: ExponentialLR, 
        optimizer: Optimizer, 
        gamma: float,
        last_epoch: int = -1
    ) -> None:
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: ExponentialLR) -> list[float]:
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [base_lr * self.gamma ** self.last_epoch 
                for base_lr in self.base_lrs]
        
class PolynomialLR(Scheduler):
    def __init__(
        self: PolynomialLR, 
        optimizer: Optimizer, 
        total_iters: int = 5,
        power: float = 1.0,
        last_epoch: int = -1
    ) -> None:
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: PolynomialLR) -> list[float]:
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]
        decay = (1 - self.last_epoch / self.total_iters) ** self.power
        return [base_lr * decay for base_lr in self.base_lrs]
        
class CosineAnnealingLR(Scheduler):
    def __init__(
        self: CosineAnnealingLR, 
        optimizer: Optimizer, 
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: CosineAnnealingLR) -> list[float]:
        if self.last_epoch == 0: return [base_lr for base_lr in self.base_lrs]
        t = min(self.last_epoch, self.T_max)
        schedule = lambda x: self.eta_min + 0.5 * (x - self.eta_min) \
            * (1 + math.cos(math.pi * t / self.T_max))
        return [schedule(base_lr) for base_lr in self.base_lrs]
        
class CosineAnnealingWarmRestarts(Scheduler):
    def __init__(
        self: CosineAnnealingWarmRestarts, 
        optimizer: Optimizer, 
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ) -> None:
        assert T_0 > 0, 'T_0 must be a positive integer.'
        assert T_mult >= 1, 'T_mult must be >= 1.'
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def _get_T_cur_T_i(self: CosineAnnealingWarmRestarts) -> tuple[int, int]:
        T_i = self.T_0
        T_cur = self.last_epoch
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult
        return T_cur, T_i

    def get_lr(self: CosineAnnealingWarmRestarts) -> list[float]:
        if self.last_epoch == 0:
            return [base_lr for base_lr in self.base_lrs]
        T_cur, T_i = self._get_T_cur_T_i()
        schedule = lambda x: self.eta_min + 0.5 * (x - self.eta_min) \
                 * (1 + math.cos(math.pi * T_cur / T_i))
        return [schedule(base_lr) for base_lr in self.base_lrs]
        
class ReduceLROnPlateau(Scheduler):
    def __init__(
        self: ReduceLROnPlateau, 
        optimizer: Optimizer, 
        mode: Literal['min', 'max'] = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 0.0001,
        threshold_mode: Literal['rel', 'abs'] = 'rel',
        cooldown: int = 0,
        eps: float = 1e-8,
        last_epoch: int = -1
    ) -> None:
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.eps = eps
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: ReduceLROnPlateau) -> list[float]:
        raise NotImplementedError
        
class CyclicLR(Scheduler):
    def __init__(
        self: CyclicLR, 
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
        last_epoch: int = -1
    ) -> None:
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
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: CyclicLR) -> list[float]:
        raise NotImplementedError
        
class OneCycleLR(Scheduler):
    def __init__(
        self: OneCycleLR, 
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
        last_epoch: int = -1
    ) -> None:
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
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: OneCycleLR) -> list[float]:
        raise NotImplementedError
        
        
        
        
        
