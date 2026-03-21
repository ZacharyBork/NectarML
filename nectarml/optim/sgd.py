from __future__ import annotations

from typing import Any

from nectarml.tensor import Tensor
from nectarml.creation import zeros_like
from nectarml.optim.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(
        self: SGD,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:           float = 0.003,
        momentum:     float = 0.0,
        dampening:    float = 0.0,
        weight_decay: float = 0.0,
        nesterov:      bool = False,
        maximize:      bool = False,
        foreach:       bool = None, # NOT YET IMPLEMENTED
        fused:         bool = None  # NOT YET IMPLEMENTED
    ) -> None:
        super().__init__(
            parameters, 
            defaults={
                'lr': 0.003,
                'momentum': 0.0,
                'dampening': 0.0,
                'weight_decay': 0.0,
                'nesterov': False
            }
        )
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov and momentum > 0.0
        self.maximize = maximize
        self.foreach = foreach
        self.fused = fused
        
    def _build_state(self: SGD, param_index: int, param: Tensor) -> None:
        if self.momentum > 0.0:
            if 'velocity' not in self.state[param_index]:
                self.state[param_index]['velocity'] = zeros_like(param)
                
    def _update(self: SGD) -> None:
        for group in self.param_groups:
            _lr = group.get('lr', self.lr)
                        
            for param in group['params']:
                if param.grad is None: continue
                idx = self._get_parameter_state_index(param)
                self._build_state(idx, param)
                
                grad = param.grad.clone()
                
                if self.maximize: grad = -grad
                if self.weight_decay: 
                    grad = grad + self.weight_decay * param.detach()
                
                if self.momentum > 0.0:
                    if idx not in self.state: self.state[idx] = {}
                    if 'velocity' not in self.state[idx]:
                        self.state[idx]['velocity'] = zeros_like(param)
                    
                    v = self.state[idx]['velocity']
                    v = self.momentum * v + (1 - self.dampening) * grad
                    self.state[idx]['velocity'] = v
                    
                    if self.nesterov: grad = grad + self.momentum * v
                    else: grad = v
                
                param -= _lr * grad
                
            

