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
        foreach:                bool = None,
        capturable:             bool = False,
        fused:                  bool = False
    ) -> None:
        super().__init__(parameters, defaults=None)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.decoupled_weight_decay = decoupled_weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.foreach = foreach
        self.capturable = capturable
        self.fused = fused
        
    def _update(self: Adam) -> None:
        pass

class AdamW(Optimizer):
    def __init__(
        self: AdamW,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:                    float = 0.003,
        betas:   tuple[float, float] = (0.9, 0.999),
        eps:                   float = 1e-8,
        weight_decay:          float = 0.0,
        amsgrad:                bool = False,
        maximize:               bool = False,
        foreach:                bool = None,
        capturable:             bool = False,
        fused:                  bool = False
    ) -> None:
        super().__init__(parameters, defaults=None)
        self.lr = lr
        self.betas = betas
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
        foreach:                bool = None,
        capturable:             bool = False
    ) -> None:
        super().__init__(parameters, defaults=None)
        self.lr = lr
        self.betas = betas
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
        foreach:                bool = None,
        capturable:             bool = False
    ) -> None:
        super().__init__(parameters, defaults=None)
        self.lr = lr
        self.betas = betas
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
        lr:                    float = 0.003,
        betas:   tuple[float, float] = (0.9, 0.999),
        eps:                   float = 1e-8,
        weight_decay:          float = 0.0,
        maximize:               bool = False,
        foreach:                bool = None,
        capturable:             bool = False
    ) -> None:
        super().__init__(parameters, defaults=None)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.maximize = maximize
        self.foreach = foreach
        self.capturable = capturable
        
    def _update(self: Adamax) -> None:
        pass




