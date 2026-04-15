from __future__ import annotations

import builtins
from typing import Literal

import nectarml.functional as F
from nectarml           import typing
from nectarml.tensor    import Tensor
from nectarml.nn.module import Module

### REGRESSION ###

class L1Loss(Module):
    def __init__(
        self:      L1Loss, 
        dtype:     typing.dtype = typing.float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: L1Loss, x: Tensor, y: Tensor) -> Tensor:
        return F.l1_loss(x, y, self.reduction)
   
MAELoss = L1Loss
    
class L2Loss(Module):
    def __init__(
        self:      L2Loss, 
        dtype:     typing.dtype = typing.float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: L2Loss, x: Tensor, y: Tensor) -> Tensor:
        return F.l2_loss(x, y, self.reduction)

MSELoss = L2Loss
    
class RMSELoss(Module):
    def __init__(
        self:      RMSELoss, 
        dtype:     typing.dtype = typing.float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: RMSELoss, x: Tensor, y: Tensor) -> Tensor:
        return F.rmse_loss(x, y, self.reduction)
    
class HuberLoss(Module):
    def __init__(
        self:      HuberLoss, 
        delta:     builtins.float = 1.0, 
        dtype:     typing.dtype = typing.float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.delta = delta
        self.reduction = reduction

    def forward(self: HuberLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.huber_loss(x, y, self.delta, self.reduction)
    
class LogCoshLoss(Module):
    def __init__(
        self:      LogCoshLoss,  
        dtype:     typing.dtype = typing.float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: LogCoshLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.log_cosh_loss(x, y, self.reduction)
    
### CLASSIFICATION ###
    
class BCELoss(Module):
    def __init__(
        self:      BCELoss,  
        dtype:     typing.dtype = typing.float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: BCELoss, x: Tensor, y: Tensor) -> Tensor:
        return F.bce_loss(x, y, self.reduction)
    
class CrossEntropyLoss(Module):
    def __init__(
        self:      CrossEntropyLoss,  
        dtype:     typing.dtype = typing.float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: CrossEntropyLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy_loss(x, y, self.reduction)
        
class NLLLoss(Module):
    def __init__(
        self:      NLLLoss,  
        dtype:     typing.dtype = typing.float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: NLLLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.nll_loss(x, y, self.reduction)
    
class HingeLoss(Module):
    def __init__(
        self:      HingeLoss,  
        dtype:     typing.dtype = typing.float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: HingeLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.hinge_loss(x, y, self.reduction)
    
class Hinge2Loss(Module):
    def __init__(
        self:      Hinge2Loss,  
        dtype:     typing.dtype = typing.float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: Hinge2Loss, x: Tensor, y: Tensor) -> Tensor:
        return F.hinge2_loss(x, y, self.reduction)
    
### PROBABILISTIC ###

class KLDivergenceLoss(Module):
    def __init__(
        self:      KLDivergenceLoss,  
        dtype:     typing.dtype = typing.float32,
        reduction: Literal['none', 'mean', 'sum'] = 'sum'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: KLDivergenceLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.kl_divergence_loss(x, y, self.reduction)
    
class BCEWithLogitsLoss(Module):
    def __init__(
        self:      BCEWithLogitsLoss,  
        dtype:     typing.dtype = typing.float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: BCEWithLogitsLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.bce_with_logits_loss(x, y, self.reduction)
    
### RANKING ###

class TripletMarginLoss(Module):
    def __init__(
        self:      TripletMarginLoss, 
        margin:    builtins.float = 1.0,
        eps:       builtins.float = 1e-6, 
        dtype:     typing.dtype = typing.float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.margin = margin
        self.eps = eps
        self.reduction = reduction

    def forward(
        self:     TripletMarginLoss, 
        anchor:   Tensor, 
        positive: Tensor, 
        negative: Tensor
    ) -> Tensor:
        return F.triplet_margin_loss(
            anchor, positive, negative, self.margin, self.eps, self.reduction)

