from __future__ import annotations

from typing import Any

from nectarml.tensor import Tensor
from nectarml.creation import zeros_like
from nectarml.optim.optimizer import Optimizer

class Adagrad(Optimizer):
    def __init__(
        self: Adagrad,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:                        float = 0.003,
        lr_decay:                  float = 0.0,
        weight_decay:              float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps:                       float = 1e-8,
        maximize:                   bool = False,
        foreach:                    bool = None, # NOT YET IMPLEMENTED
        fused:                      bool = False # NOT YET IMPLEMENTED
    ) -> None:
        super().__init__(
            parameters, 
            defaults={
                'lr': 0.003,
                'lr_decay': 0.0,
                'weight_decay': 0.0,
                'initial_accumulator_value': 0.0,
                'eps': 1e-8
            }
        )
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.maximize = maximize
        self.foreach = foreach
        self.fused = fused
        
    def _update(self: Adagrad) -> None:
        pass
    
class Adadelta(Optimizer):
    def __init__(
        self: Adadelta,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:           float = 0.003,
        rho:          float = 0.9,
        eps:          float = 1e-8,
        weight_decay: float = 0.0,
        maximize:      bool = False,
        foreach:       bool = None # NOT YET IMPLEMENTED
    ) -> None:
        super().__init__(
            parameters, 
            defaults={
                'lr': 0.003,
                'rho': 0.9,
                'eps': 1e-8,
                'weight_decay': 0.0
            }
        )
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.maximize = maximize
        self.foreach = foreach
        
    def _update(self: Adadelta) -> None:
        pass
    

