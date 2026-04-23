from __future__ import annotations

import math
from typing          import Any, Literal
from collections.abc import Callable

import matplotlib.pyplot as plt

from nectarml.optim.optimizer import Optimizer
from nectarml.constants       import PI

### ABSTRACT PARENT ###

class Scheduler:
    def __init__(
        self:       Scheduler, 
        optimizer:  Optimizer, 
        last_epoch: int = -1
    ) -> None:
        self._optimizer  = optimizer
        self._last_epoch = last_epoch
        self._base_lrs   = [group['lr'] for group in optimizer.param_groups]
        self._last_lrs   = self.base_lrs.copy()
        
        self._prev_single_lr: float = None
        self._curr_single_lr: float = None
        self.step()
        
    ### PROPERTIES ###
    
    @property
    def lr(self: Scheduler) -> float:
        '''Convenience property to get current LR of first optimizer parameter.
        
        Equivelent to Scheduler.get_last_lr()[0]
        '''
        lrs = self.get_last_lr()
        if isinstance(lrs, float): return lrs
        return lrs[0] if lrs else self.optimizer.lr
    
    @property
    def optimizer(self: Scheduler) -> Optimizer:
        '''Returns a reference to the Optimizer managed by the Scheduler.'''
        return self._optimizer
    
    @property
    def last_epoch(self: Scheduler) -> int:
        '''Returns the last_epoch value of the Scheduler.'''
        return self._last_epoch
    
    @property
    def base_lrs(self: Scheduler) -> list[float]:
        '''Returns a list of the base LR values of all scheduled parameters.'''
        return self._base_lrs
        
    ### GETTERS ###
        
    def get_lr(self: Scheduler) -> list[float]:
        raise NotImplementedError

    def get_last_lr(self: Scheduler) -> list[float]:
        return self._last_lrs

    ### SCHEDULE ###

    def step(self: Scheduler, epoch: int | None = None) -> None:
        if epoch is None: self._last_epoch += 1
        else: self._last_epoch = epoch
        
        self._prev_single_lr = self.lr
        self._last_lrs       = self.get_lr()
        self._curr_single_lr = self.lr
        
        for group, lr in zip(self.optimizer.param_groups, self.get_last_lr()):
            group['lr'] = lr

    ### STATE DICT ###

    def state_dict(self: Scheduler) -> dict[str, Any]:
        return { k: v for k, v in self.__dict__.items() if k != 'optimizer' }

    def load_state_dict(self: Scheduler, state_dict: dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
        
    ### UTILITIES ###
        
    def eval_schedule(
        self:           Scheduler,
        start_timestep: int = 0,
        end_timestep:   int = 100
    ) -> list[float]:
        _last_epoch_prev = self.last_epoch
        values           = [self.lr]
        self._last_epoch = start_timestep
        for idx in range(start_timestep, end_timestep):
            self.step(idx+1)
            values.append(self.lr)
        self._last_epoch = _last_epoch_prev
        return values
        
    def plot(
        self:           Scheduler, 
        start_timestep: int = 0,
        end_timestep:   int = 100
    ) -> None:
        values = self.eval_schedule(start_timestep, end_timestep)
        plt.xlabel('Timesteps'); plt.ylabel('Learning Rate')
        plt.plot(values)
        plt.show()
        
    def print(
        self:           Scheduler, 
        start_timestep: int = 0,
        end_timestep:   int = 100
    ) -> None:
        values = self.eval_schedule(start_timestep, end_timestep)
        for idx, value in enumerate(values):
            idx_str  = f'{idx+1+start_timestep}'.rjust(len(str(len(values))))
            step_str = f'(Timestep {idx_str})'
            before   = values[max(0, idx-1)]
            print(f'{step_str} LR: {before:.4f} -> {value:.4f}')

### COMPOSITION ###

class SequentialLR(Scheduler):
    def __init__(
        self:       SequentialLR,
        optimizer:  Optimizer,
        schedulers: list[Scheduler],
        milestones: list[int],
        last_epoch: int | None = None
    ) -> None:
        self.schedulers = schedulers
        self.milestones = milestones
        self._idx       = 0
        self._active    = self.schedulers[self._idx]
        
        for group, lr in zip(optimizer.param_groups, schedulers[0].base_lrs):
            group['lr'] = lr
        
        super().__init__(optimizer, last_epoch or -1)
        self._init_schedule()
                
    def _init_schedule(self: SequentialLR) -> None:
        if self.last_epoch is not None:
            for i, milestone in enumerate(self.milestones):
                if self.last_epoch >= milestone:
                      self._idx = i + 1
                else: break
            self._idx = min(self._idx, len(self.schedulers) - 1)
            
            elapsed = self.last_epoch
            for i in range(self._idx):
                elapsed -= self.milestones[i] \
                        - (self.milestones[i-1] if i > 0 else 0)
            
            self._active = self.schedulers[self._idx]
            if hasattr(self._active, 'last_epoch'):
                self._active._last_epoch = elapsed
    
    def get_lr(self: SequentialLR) -> float:
        if self._idx < len(self.schedulers):
            return self._active.get_lr()[0]
        return self.optimizer.lr
    
    def get_last_lr(self: SequentialLR) -> list[float]:
        if self._idx < len(self.schedulers):
            return self._active.get_last_lr()[0]
        return self.optimizer.lr
    
    def step(self: SequentialLR, epoch: int | None = None) -> None:
        if epoch is None: self._last_epoch += 1
        else: self._last_epoch = epoch
        
        if  self._idx < len(self.milestones) \
        and self.last_epoch > self.milestones[self._idx]:
            self._idx += 1
            self._active = self.schedulers[self._idx]
        
        self._prev_single_lr = self._active.lr
        if self._idx < len(self.schedulers):
            self._active.step()
        self._curr_single_lr = self._active.lr
            
    def clone(self: SequentialLR, new_optimizer: Optimizer) -> SequentialLR:
        return SequentialLR(
            optimizer  = new_optimizer, 
            schedulers = self.schedulers.copy(), 
            milestones = self.milestones.copy(),
            last_epoch = self.last_epoch
        )

### DEFAULT SCHEDULERS ###

class StepLR(Scheduler):
    def __init__(
        self:       StepLR, 
        optimizer:  Optimizer, 
        step_size:  int,
        gamma:      float = 0.01,
        last_epoch: int   = -1
    ) -> None:
        self.step_size = step_size
        self.gamma     = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: StepLR) -> list[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma 
                for group in self.optimizer.param_groups]

class MultiStepLR(Scheduler):
    def __init__(
        self:       MultiStepLR, 
        optimizer:  Optimizer, 
        milestones: list[int],
        gamma:      float = 0.01,
        last_epoch: int   = -1
    ) -> None:
        self.milestones = milestones
        self.gamma      = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: MultiStepLR) -> list[float]:
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

class ConstantLR(Scheduler):
    def __init__(
        self:        ConstantLR, 
        optimizer:   Optimizer, 
        factor:      float = 0.3333333333333333,
        total_iters: int   = 5,
        last_epoch:  int   = -1
    ) -> None:
        self.factor      = factor
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
        self:         LinearLR, 
        optimizer:    Optimizer, 
        start_factor: float = 0.3333333333333333,
        end_factor:   float = 1.0,
        total_iters:  int   = 5,
        last_epoch:   int   = -1
    ) -> None:
        self.start_factor = start_factor
        self.end_factor   = end_factor
        self.total_iters  = total_iters
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: LinearLR) -> list[float]:
        if self.last_epoch >= self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
        
        t = self.last_epoch / self.total_iters
        factor = self.start_factor + (self.end_factor - self.start_factor) * t
        return [base_lr * factor for base_lr in self.base_lrs]
        
class ExponentialLR(Scheduler):
    def __init__(
        self:       ExponentialLR, 
        optimizer:  Optimizer, 
        gamma:      float,
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
        self:        PolynomialLR, 
        optimizer:   Optimizer, 
        total_iters: int   = 5,
        power:       float = 1.0,
        last_epoch:  int   = -1
    ) -> None:
        self.total_iters = total_iters
        self.power       = power
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: PolynomialLR) -> list[float]:
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]
        decay = (1 - self.last_epoch / self.total_iters) ** self.power
        return [base_lr * decay for base_lr in self.base_lrs]
        
class CosineAnnealingLR(Scheduler):
    def __init__(
        self:       CosineAnnealingLR, 
        optimizer:  Optimizer, 
        T_max:      int,
        eta_min:    float = 0.0,
        last_epoch: int   = -1
    ) -> None:
        self.T_max   = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self: CosineAnnealingLR) -> list[float]:
        if self.last_epoch == 0: return [base_lr for base_lr in self.base_lrs]
        t = min(self.last_epoch, self.T_max)
        schedule = lambda x: self.eta_min + 0.5 * (x - self.eta_min) \
            * (1 + math.cos(PI * t / self.T_max))
        return [schedule(base_lr) for base_lr in self.base_lrs]
        
class CosineAnnealingWarmRestarts(Scheduler):
    def __init__(
        self:       CosineAnnealingWarmRestarts, 
        optimizer:  Optimizer, 
        T_0:        int,
        T_mult:     int   = 1,
        eta_min:    float = 0.0,
        last_epoch: int   = -1
    ) -> None:
        assert T_0 > 0, 'T_0 must be a positive integer.'
        assert T_mult >= 1, 'T_mult must be >= 1.'
        self.T_0     = T_0
        self.T_mult  = T_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def _get_T_cur_T_i(self: CosineAnnealingWarmRestarts) -> tuple[int, int]:
        T_i   = self.T_0
        T_cur = self.last_epoch
        while T_cur >= T_i:
            T_cur -= T_i
            T_i   *= self.T_mult
        return T_cur, T_i

    def get_lr(self: CosineAnnealingWarmRestarts) -> list[float]:
        if self.last_epoch == 0:
            return [base_lr for base_lr in self.base_lrs]
        T_cur, T_i = self._get_T_cur_T_i()
        schedule = lambda x: self.eta_min + 0.5 * (x - self.eta_min) \
                 * (1 + math.cos(PI * T_cur / T_i))
        return [schedule(base_lr) for base_lr in self.base_lrs]
        
class ReduceLROnPlateau(Scheduler):
    def __init__(
        self:           ReduceLROnPlateau, 
        optimizer:      Optimizer, 
        mode:           Literal['min', 'max'] = 'min',
        factor:         float = 0.1,
        patience:       int   = 10,
        threshold:      float = 0.0001,
        threshold_mode: Literal['rel', 'abs'] = 'rel',
        cooldown:       int   = 0,
        min_lr:         float = 0.0,
        eps:            float = 1e-8,
        last_epoch:     int   = -1
    ) -> None:
        assert 0.0 < factor < 1.0, \
            'factor must be between 0.0 and 1.0, not inclusive.'
        self.mode           = mode
        self.factor         = factor
        self.patience       = patience
        self.threshold      = threshold
        self.threshold_mode = threshold_mode
        self.cooldown       = cooldown
        self.min_lr         = min_lr
        self.eps            = eps
        
        self.best: int | float = None
        self.num_bad_epochs    = 0
        self.cooldown_counter  = 0
        
        super().__init__(optimizer, last_epoch)
        
    def _check_for_improvement(
        self:   ReduceLROnPlateau,
        metric: int | float
    ) -> bool:
        if self.mode == 'min':
            check = self.best * (1 - self.threshold) \
                    if self.threshold_mode == 'rel' \
                    else self.best - self.threshold
            improved = metric < check
        else:
            check = self.best * (1 + self.threshold) \
                    if self.threshold_mode == 'rel' \
                    else self.best + self.threshold
            improved = metric > check
        return improved
    
    def get_lr(self: ReduceLROnPlateau) -> list[float]:
        return [max(self.min_lr, group['lr'] * self.factor)
            for group in self.optimizer.param_groups]
    
    def step(
        self:   ReduceLROnPlateau, 
        metric: int | float
    ) -> None:
        if self.best is None: 
            self.best = metric
            return
        
        if self.cooldown_counter > 0: 
            self.cooldown_counter -= 1
            return
        
        improved = self._check_for_improvement(metric)
        
        if improved:
            self.num_bad_epochs = 0
            self.best           = metric
        else: self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience:
            self.num_bad_epochs   = 0
            self.cooldown_counter = self.cooldown
            
            self._prev_single_lr = self.lr
            self._last_lrs       = self.get_lr()
            self._curr_single_lr = self.lr
            
            for group, lr in zip(self.optimizer.param_groups, self._last_lrs):
                if group['lr'] - lr > self.eps: group['lr'] = lr
        
class CyclicLR(Scheduler):
    def __init__(
        self:           CyclicLR, 
        optimizer:      Optimizer, 
        base_lr:        float,
        max_lr:         float,
        step_size_up:   int = 2000,
        step_size_down: int | None = None,
        mode:           Literal[
            'triangular', 'triangular2', 'exp_range'
        ] = 'triangular',
        gamma:          float    | None = None,
        scale_fn:       Callable | None = None,
        scale_mode:     Literal['cycle', 'iterations'] = 'cycle',
        cycle_momentum: bool = True,
        base_momentum:  float | list[float] = 0.8,
        max_momentum:   float | list[float] = 0.9,
        last_epoch:     int = -1
    ) -> None:
        self.base_lr        = base_lr
        self.max_lr         = max_lr
        self.step_size_up   = step_size_up
        self.step_size_down = step_size_down if step_size_down is not None \
                         else step_size_up
        self.cycle_momentum = cycle_momentum
        
        if isinstance(base_momentum, list):
            assert len(base_momentum) == len(optimizer.param_groups), \
                f'base_momentum list length {len(base_momentum)} must match ' \
                f'number of param groups {len(optimizer.param_groups)}.'
            self.base_momentum = base_momentum
        else: self.base_momentum = [base_momentum]*len(optimizer.param_groups)

        if isinstance(max_momentum, list):
            assert len(max_momentum) == len(optimizer.param_groups), \
                f'max_momentum list length {len(max_momentum)} must match ' \
                f'number of param groups {len(optimizer.param_groups)}.'
            self.max_momentum = max_momentum
        else: self.max_momentum = [max_momentum]*len(optimizer.param_groups)
        
        self._init_scale_fn(mode, gamma, scale_fn, scale_mode)
        super().__init__(optimizer, last_epoch)
        
    def _init_scale_fn(
        self:       CyclicLR,
        mode:       Literal['triangular', 'triangular2', 'exp_range'],
        gamma:      float    | None,
        scale_fn:   Callable | None,
        scale_mode: Literal['cycle', 'iterations']
    ) -> None:
        if scale_fn is None:
            if mode == 'triangular':
                self.scale_fn   = lambda x: 1.0
                self.scale_mode = 'cycle'
            elif mode == 'triangular2':
                self.scale_fn   = lambda x: 1.0 / (2.0 ** (x - 1))
                self.scale_mode = 'cycle'
            elif mode == 'exp_range':
                assert gamma is not None, \
                    'gamma must be provided when mode="exp_range".'
                self.scale_fn   = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        
    def get_lr(self: CyclicLR) -> list[float]:
        cycle_size   = self.step_size_up + self.step_size_down
        pos_in_cycle = self.last_epoch % cycle_size
        cycle = math.floor(1 + self.last_epoch / cycle_size)
        if pos_in_cycle < self.step_size_up:
            x = pos_in_cycle / self.step_size_up
        else:
            x = (pos_in_cycle - self.step_size_up) / self.step_size_down
            x = 1 - x
                
        arg   = cycle if self.scale_mode == 'cycle' else self.last_epoch
        scale = self.scale_fn(arg)
        lr_values = [base_lr + (self.max_lr - base_lr) * x * scale
                     for base_lr in self.base_lrs]
        
        if self.cycle_momentum:
            zip_m = zip(self.base_momentum, self.max_momentum)
            momentums = [max_m - (max_m - base_m) * x * scale
                         for base_m, max_m in zip_m]
            for group, m in zip(self.optimizer.param_groups, momentums):
                group['momentum'] = m
        
        return lr_values
        
class OneCycleLR(Scheduler):
    def __init__(
        self:             OneCycleLR, 
        optimizer:        Optimizer, 
        max_lr:           float | list[float],
        total_steps:      int | None = None,
        epochs:           int | None = None,
        steps_per_epoch:  int | None = None,
        pct_start:        float = 0.3,
        anneal_strategy:  Literal['cos', 'linear'] = 'cos',
        cycle_momentum:   bool = True,
        base_momentum:    float | list[float] = 0.85,
        max_momentum:     float | list[float] = 0.95,
        div_factor:       float = 25.0,
        final_div_factor: float = 1e4,
        three_phase:      bool  = True,
        last_epoch:       int   = -1
    ) -> None:
        self.max_lr           = max_lr
        self.total_steps      = total_steps
        self.epochs           = epochs
        self.steps_per_epoch  = steps_per_epoch
        self.pct_start        = pct_start
        self.anneal_strategy  = anneal_strategy
        self.cycle_momentum   = cycle_momentum
        self.div_factor       = div_factor
        self.final_div_factor = final_div_factor
        
        if total_steps is not None: self.total_steps = total_steps
        elif epochs is not None and steps_per_epoch is not None:
            self.total_steps = epochs * steps_per_epoch
        else: raise ValueError(
                'Must provide total_steps or both epochs and steps_per_epoch')

        self._compute_phases(
            optimizer, three_phase, pct_start, max_lr, base_momentum, 
            max_momentum, div_factor, final_div_factor)
        super().__init__(optimizer, last_epoch)
        
    def _compute_phases(
        self:             OneCycleLR,
        optimizer:        Optimizer,
        three_phase:      bool,
        pct_start:        float,
        max_lr:           float | list[float],
        base_momentum:    float | list[float],
        max_momentum:     float | list[float],
        div_factor:       float,
        final_div_factor: float
    ) -> None:
        n_groups = len(optimizer.param_groups)
        max_lrs = [max_lr] * n_groups if isinstance(max_lr, (int, float)) \
                  else max_lr
        base_momentums = [base_momentum] * n_groups \
                         if isinstance(base_momentum, (int, float)) \
                         else base_momentum
        max_momentums = [max_momentum] * n_groups \
                        if isinstance(max_momentum, (int, float)) \
                        else max_momentum

        if three_phase:
            self.phases = [
                {
                    'end_step': pct_start * self.total_steps,
                    'start_lrs': [lr / div_factor for lr in max_lrs],
                    'end_lrs': max_lrs,
                    'start_momentums': max_momentums,
                    'end_momentums': base_momentums
                },
                {
                    'end_step': 0.9 * self.total_steps,
                    'start_lrs': max_lrs,
                    'end_lrs': [lr / div_factor for lr in max_lrs],
                    'start_momentums': base_momentums,
                    'end_momentums': max_momentums
                },
                {
                    'end_step': self.total_steps,
                    'start_lrs': [lr / div_factor for lr in max_lrs],
                    'end_lrs': [
                        lr / (div_factor * final_div_factor) for lr in max_lrs
                    ],
                    'start_momentums': max_momentums,
                    'end_momentums': max_momentums
                }
            ]
        else:
            self.phases = [
                {
                    'end_step': pct_start * self.total_steps,
                    'start_lrs': [lr / div_factor for lr in max_lrs],
                    'end_lrs':   max_lrs,
                    'start_momentums': max_momentums,
                    'end_momentums': base_momentums
                },
                {
                    'end_step': self.total_steps,
                    'start_lrs': max_lrs,
                    'end_lrs': [
                        lr / (div_factor * final_div_factor) for lr in max_lrs
                    ],
                    'start_momentums': max_momentums,
                    'end_momentums': max_momentums
                }
            ]
        
    def _anneal(
        self:  OneCycleLR,
        start: float,
        end:   float,
        pct:   float
    ) -> float:
        if self.anneal_strategy == 'cos':
              return end   + (start - end) / 2 * (1 + math.cos(PI * pct))
        else: return start + (end - start) * pct
        
    def get_lr(self: OneCycleLR) -> list[float]:
        step = self.last_epoch
        start_step = 0
        for phase in self.phases:
            if step <= phase['end_step']:
                phase_length = phase['end_step'] - start_step
                pct = (step - start_step) / phase_length \
                    if phase_length > 0 else 1.0
                
                lrs = [self._anneal(s, e, pct) 
                    for s, e in zip(phase['start_lrs'], phase['end_lrs'])]
                
                if self.cycle_momentum:
                    moms = [self._anneal(s, e, pct)
                            for s, e in zip(phase['start_momentums'], 
                                            phase['end_momentums'])]
                    for group, mom in zip(self.optimizer.param_groups, moms):
                        group['momentum'] = mom
                
                return lrs
            start_step = phase['end_step']
        return self.phases[-1]['end_lrs']
                
        
        
        
        
