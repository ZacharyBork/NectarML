from __future__ import annotations

from typing import Any

import _nectarml
from nectarml.core            import Tensor, creation as T
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
        fused:         bool = True
    ) -> None:
        '''
        Reference:
            - Liu et al., "An Improved Analysis of Stochastic Gradient Descent 
            with Momentum.", https://arxiv.org/pdf/2007.07989
        '''
        super().__init__(parameters, defaults={
            'lr':           lr,
            'momentum':     momentum,
            'dampening':    dampening,
            'weight_decay': weight_decay,
            'nesterov':     nesterov
        })
        self.lr           = lr
        self.momentum     = momentum
        self.dampening    = dampening
        self.weight_decay = weight_decay
        self.nesterov     = nesterov and momentum > 0.0
        self.maximize     = maximize
        self.foreach      = foreach
        self.fused        = fused
        
    def _build_state(self: SGD, param_index: int, param: Tensor) -> None:
        if param_index not in self.state: self.state[param_index] = {}
        if self.momentum > 0.0:
            if 'velocity' not in self.state[param_index]:
                self.state[param_index]['velocity'] = T.zeros_like(param)
                
    def _run_update(
        self:  SGD, 
        param: Tensor,
        idx:   int,
        lr:    float
    ) -> None:
        grad = param.grad.detach().clone()
                    
        if self.maximize: grad = -grad
        if self.weight_decay: 
            grad = grad + self.weight_decay * param.detach()
        
        if self.momentum > 0.0:
            v = self.state[idx]['velocity']
            v = self.momentum * v + (1 - self.dampening) * grad
            self.state[idx]['velocity'] = v.detach()
            
            if self.nesterov: grad = grad + self.momentum * v
            else: grad = v
        
        param -= (lr * grad).detach()
                
    def _update(self: SGD) -> None:
        for group in self.param_groups:
            _lr = group.get('lr', self.lr)
                        
            for param in group['params']:
                if param.grad is None: continue
                idx = self._get_parameter_state_index(param)
                self._build_state(idx, param)
                
                if self.fused and param.device == 'cuda':
                    _nectarml.optim.sgd_update(
                        param._data_ptr, param.grad._data_ptr,
                        self.state[idx]['velocity']._data_ptr,
                        _lr, self.momentum, self.dampening, self.weight_decay, 
                        self.nesterov, self.maximize, param.size)
                else: self._run_update(param, idx, _lr)
                
