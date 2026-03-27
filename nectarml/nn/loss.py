from __future__ import annotations

from typing import Literal

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.nn.module import Module
from nectarml.typing import DTypeLike, float32

### REGRESSION ###

class L1Loss(Module):
    def __init__(
        self: L1Loss, 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: L1Loss, x: Tensor, y: Tensor) -> Tensor:
        return F.L1Loss(x, y, self.reduction)
   
MAELoss = L1Loss
    
class L2Loss(Module):
    def __init__(
        self: L2Loss, 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: L2Loss, x: Tensor, y: Tensor) -> Tensor:
        return F.L2Loss(x, y, self.reduction)

MSELoss = L2Loss
    
class RMSELoss(Module):
    def __init__(
        self: RMSELoss, 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: RMSELoss, x: Tensor, y: Tensor) -> Tensor:
        return F.RMSELoss(x, y, self.reduction)
    
class HuberLoss(Module):
    def __init__(
        self: HuberLoss, 
        delta: float = 1.0, 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.delta = delta
        self.reduction = reduction

    def forward(self: HuberLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.HuberLoss(x, y, self.delta, self.reduction)
    
class LogCoshLoss(Module):
    def __init__(
        self: LogCoshLoss,  
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: LogCoshLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.LogCoshLoss(x, y, self.reduction)
    
### CLASSIFICATION ###
    
class BCELoss(Module):
    def __init__(
        self: BCELoss,  
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: BCELoss, x: Tensor, y: Tensor) -> Tensor:
        return F.BCELoss(x, y, self.reduction)
    
class CrossEntropyLoss(Module):
    def __init__(
        self: CrossEntropyLoss,  
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: CrossEntropyLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.CrossEntropyLoss(x, y, self.reduction)
        
class NLLLoss(Module):
    def __init__(
        self: NLLLoss,  
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: NLLLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.NLLLoss(x, y, self.reduction)
    
class HingeLoss(Module):
    def __init__(
        self: HingeLoss,  
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: HingeLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.HingeLoss(x, y, self.reduction)
    
class Hinge2Loss(Module):
    def __init__(
        self: Hinge2Loss,  
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: Hinge2Loss, x: Tensor, y: Tensor) -> Tensor:
        return F.Hinge2Loss(x, y, self.reduction)
    
### PROBABILISTIC ###

class KLDivergenceLoss(Module):
    def __init__(
        self: KLDivergenceLoss,  
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'sum'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: KLDivergenceLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.KLDivergenceLoss(x, y, self.reduction)
    
class BCEWithLogitsLoss(Module):
    def __init__(
        self: BCEWithLogitsLoss,  
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.reduction = reduction

    def forward(self: BCEWithLogitsLoss, x: Tensor, y: Tensor) -> Tensor:
        return F.BCEWithLogitsLoss(x, y, self.reduction)
    
### RANKING ###

class TripletMarginLoss(Module):
    def __init__(
        self: TripletMarginLoss, 
        margin: float = 1.0,
        eps: float = 1e-6, 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(dtype)
        self.margin = margin
        self.eps = eps
        self.reduction = reduction

    def forward(
        self: TripletMarginLoss, 
        anchor: Tensor, 
        positive: Tensor, 
        negative: Tensor
    ) -> Tensor:
        return F.TripletMarginLoss(
            anchor, positive, negative, self.margin, self.eps, self.reduction)

