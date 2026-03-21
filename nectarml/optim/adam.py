from __future__ import annotations

from typing import Any

from nectarml.tensor import Tensor
from nectarml.creation import zeros_like
from nectarml.optim.optimizer import Optimizer

class Adam(Optimizer):
    def __init__(
        self: Adam,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:                    float = 0.003,
        betas:   tuple[float, float] = (0.9, 0.999),
        eps:                   float = 1e-8,
        weight_decay:          float = 0.0,
        decoupled_weight_decay: bool = False,
        amsgrad:                bool = False,
        maximize:               bool = False,
        foreach:                bool = None,  # NOT YET IMPLEMENTED
        capturable:             bool = False, # NOT YET IMPLEMENTED
        fused:                  bool = False  # NOT YET IMPLEMENTED
    ) -> None:
        super().__init__(
            parameters, 
            defaults={
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0.0,
                'decoupled_weight_decay': False,
                'amsgrad': False
            }
        )
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.decoupled_weight_decay = decoupled_weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.foreach = foreach
        self.capturable = capturable
        self.fused = fused
              
    def _build_state(self: Adam, param_index: int, param: Tensor) -> None:
        if param_index not in self.state: self.state[param_index] = {}
        if 'exp_avg' not in self.state[param_index]:
            self.state[param_index]['exp_avg'] = zeros_like(param)
        if 'exp_avg_sq' not in self.state[param_index]:
            self.state[param_index]['exp_avg_sq'] = zeros_like(param)
        if 'step' not in self.state[param_index]: 
            self.state[param_index]['step'] = 0
        if self.amsgrad:
            if 'max_exp_avg_sq' not in self.state[param_index]:
                self.state[param_index]['max_exp_avg_sq'] = zeros_like(param)
        
    def _update(self: Adam) -> None:
        for group in self.param_groups:
            _lr = group.get('lr', self.lr)
                        
            for param in group['params']:
                if param.grad is None: continue
                idx = self._get_parameter_state_index(param)
                self._build_state(idx, param)
                
                grad = param.grad.clone()
                
                if self.maximize: grad = -grad
                if self.weight_decay: 
                    if self.decoupled_weight_decay:
                        param -= _lr * self.weight_decay * param.detach()
                    else: grad = grad + self.weight_decay * param.detach()
                    
                self.state[idx]['step'] += 1
                step = self.state[idx]['step']
                
                exp_avg = self.state[idx]['exp_avg']
                exp_avg = self.beta1 * exp_avg + (1 - self.beta1) * grad
                self.state[idx]['exp_avg'] = exp_avg
                
                exp_avg_sq = self.state[idx]['exp_avg_sq']
                exp_avg_sq = self.beta2 * exp_avg_sq + (1-self.beta2) * grad**2
                self.state[idx]['exp_avg_sq'] = exp_avg_sq
                
                exp_avg_corrected = exp_avg / (1 - self.beta1**step)
                exp_avg_sq_corrected = exp_avg_sq / (1 - self.beta2**step) 
                
                if self.amsgrad:
                    max_exp_avg_sq = self.state[idx]['max_exp_avg_sq']
                    max_exp_avg_sq = max_exp_avg_sq.maximum(
                        exp_avg_sq_corrected)
                    self.state[idx]['max_exp_avg_sq'] = max_exp_avg_sq
                    denom = max_exp_avg_sq.sqrt() + self.eps
                else: denom = exp_avg_sq_corrected.sqrt() + self.eps
                
                param -= _lr * exp_avg_corrected / denom

class AdamW(Optimizer):
    def __init__(
        self: AdamW,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:                  float = 0.003,
        betas: tuple[float, float] = (0.9, 0.999),
        eps:                 float = 1e-8,
        weight_decay:        float = 0.0,
        amsgrad:              bool = False,
        maximize:             bool = False,
        foreach:              bool = None,  # NOT YET IMPLEMENTED
        capturable:           bool = False, # NOT YET IMPLEMENTED
        fused:                bool = False  # NOT YET IMPLEMENTED
    ) -> None:
        super().__init__(
            parameters, 
            defaults={
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0.0,
                'amsgrad': False
            }
        )
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.foreach = foreach
        self.capturable = capturable
        self.fused = fused
        
    def _update(self: AdamW) -> None:
        pass
    
class NAdam(Optimizer):
    def __init__(
        self: NAdam,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:                    float = 0.003,
        betas:   tuple[float, float] = (0.9, 0.999),
        eps:                   float = 1e-8,
        weight_decay:          float = 0.0,
        momentum_decay:        float = 0.004,
        decoupled_weight_decay: bool = False,
        maximize:               bool = False,
        foreach:                bool = None,  # NOT YET IMPLEMENTED
        capturable:             bool = False  # NOT YET IMPLEMENTED
    ) -> None:
        super().__init__(
            parameters, 
            defaults={
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0.0,
                'momentum_decay': 0.004,
                'decoupled_weight_decay': False
            }
        )
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum_decay = momentum_decay
        self.decoupled_weight_decay = decoupled_weight_decay
        self.maximize = maximize
        self.foreach = foreach
        self.capturable = capturable
        
    def _update(self: NAdam) -> None:
        pass
    
class RAdam(Optimizer):
    def __init__(
        self: RAdam,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:                    float = 0.003,
        betas:   tuple[float, float] = (0.9, 0.999),
        eps:                   float = 1e-8,
        weight_decay:          float = 0.0,
        decoupled_weight_decay: bool = False,
        maximize:               bool = False,
        foreach:                bool = None, # NOT YET IMPLEMENTED
        capturable:             bool = False # NOT YET IMPLEMENTED
    ) -> None:
        super().__init__(
            parameters, 
            defaults={
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0.0,
                'decoupled_weight_decay': False
            }
        )
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.decoupled_weight_decay = decoupled_weight_decay
        self.maximize = maximize
        self.foreach = foreach
        self.capturable = capturable
        
    def _update(self: RAdam) -> None:
        pass

class Adamax(Optimizer):
    def __init__(
        self: Adamax,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:                  float = 0.003,
        betas: tuple[float, float] = (0.9, 0.999),
        eps:                 float = 1e-8,
        weight_decay:        float = 0.0,
        maximize:             bool = False,
        foreach:              bool = None, # NOT YET IMPLEMENTED
        capturable:           bool = False # NOT YET IMPLEMENTED
    ) -> None:
        super().__init__(
            parameters, 
            defaults={
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0.0
            }
        )
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.maximize = maximize
        self.foreach = foreach
        self.capturable = capturable
        
    def _update(self: Adamax) -> None:
        pass




