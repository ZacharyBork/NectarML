from typing import Literal

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32
from nectarml.nn import Module
import nectarml.functional as F

### REGRESSION ###

class L1Loss(Module):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(device, dtype)
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.L1Loss(x, y, self.reduction)
   
MAELoss = L1Loss
    
class L2Loss(Module):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(device, dtype)
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.L2Loss(x, y, self.reduction)

MSELoss = L2Loss
    
class RMSELoss(Module):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(device, dtype)
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.RMSELoss(x, y, self.reduction)
    
class HuberLoss(Module):
    def __init__(
        self, 
        delta: float = 1.0,
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(device, dtype)
        self.delta = delta
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.HuberLoss(x, y, self.delta, self.reduction)
    
class LogCoshLoss(Module):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(device, dtype)
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.LogCoshLoss(x, y, self.reduction)
    
### CLASSIFICATION ###
    
class BCELoss(Module):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(device, dtype)
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.BCELoss(x, y, self.reduction)
    
class CrossEntropyLoss(Module):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(device, dtype)
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.CrossEntropyLoss(x, y, self.reduction)
        
class NLLLoss(Module):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(device, dtype)
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.NLLLoss(x, y, self.reduction)
    
class HingeLoss(Module):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(device, dtype)
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.HingeLoss(x, y, self.reduction)
    
class Hinge2Loss(Module):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(device, dtype)
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.Hinge2Loss(x, y, self.reduction)
    
### PROBABILISTIC ###

class KLDivergenceLoss(Module):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'sum'
    ) -> None:
        super().__init__(device, dtype)
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.KLDivergenceLoss(x, y, self.reduction)
    
class BCEWithLogitsLoss(Module):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(device, dtype)
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.BCEWithLogitsLoss(x, y, self.reduction)
    
### RANKING ###

class TripletMarginLoss(Module):
    def __init__(
        self, 
        margin: float = 1.0,
        eps: float = 1e-6,
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(device, dtype)
        self.margin = margin
        self.eps = eps
        self.reduction = reduction

    def forward(
        self, 
        anchor: Tensor, 
        positive: Tensor, 
        negative: Tensor
    ) -> Tensor:
        return F.TripletMarginLoss(
            anchor, positive, negative, self.margin, self.eps, self.reduction)

