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
        lr:            float = 0.003,
        momentum:      float = 0.0,
        dampening:     float = 0.0,
        weight_decay:  float = 0.0,
        nesterov:       bool = False,
        maximize:       bool = False,
        foreach:        bool = None, # NOT YET IMPLEMENTED
        fused:          bool = None  # NOT YET IMPLEMENTED
    ) -> None:
        super().__init__(parameters, defaults=None)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov and momentum > 0.0
        self.maximize = maximize
        self.foreach = foreach
        self.fused = fused
        
        if self.momentum > 0.0:
            for idx, param in enumerate(self._get_all_params()):
                if 'velocity' not in self.state[idx]:
                    self.state[idx]['velocity'] = zeros_like(param)
                
    def _update(self: SGD) -> None:
        for idx, param in enumerate(self._get_all_params()):
            if param.grad is None: continue
            
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
            
            param -= self.lr * grad
                
            

