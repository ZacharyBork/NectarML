from __future__ import annotations

import nectarml.nn.functional as F
from nectarml.core      import Tensor
from nectarml.nn.module import Module

class Dropout(Module):
    def __init__(
        self:    Dropout,
        p:       float = 0.5,
        inplace: bool  = False
    ) -> None:
        super().__init__()
        self.p       = p
        self.inplace = inplace
        
    def forward(self: Dropout, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training, self.inplace)

class AlphaDropout(Module):
    def __init__(
        self:    AlphaDropout,
        p:       float = 0.5,
        inplace: bool  = False
    ) -> None:
        super().__init__()
        self.p = p
        self.inplace = inplace
        
    def forward(self: AlphaDropout, x: Tensor) -> Tensor:
        return F.alpha_dropout(x, self.p, self.training, self.inplace)
    
class FeatureAlphaDropout(Module):
    def __init__(
        self:    FeatureAlphaDropout,
        p:       float = 0.5,
        inplace: bool  = False
    ) -> None:
        super().__init__()
        self.p       = p
        self.inplace = inplace
        
    def forward(self: FeatureAlphaDropout, x: Tensor) -> Tensor:
        return F.feature_alpha_dropout(x, self.p, self.training, self.inplace)

class Dropout1d(Module):
    def __init__(
        self:    Dropout1d,
        p:       float = 0.5,
        inplace: bool  = False
    ) -> None:
        super().__init__()
        self.p       = p
        self.inplace = inplace
        
    def forward(self: Dropout1d, x: Tensor) -> Tensor:
        return F.dropout1d(x, self.p, self.training, self.inplace)

class Dropout2d(Module):
    def __init__(
        self:    Dropout2d,
        p:       float = 0.5,
        inplace: bool  = False
    ) -> None:
        super().__init__()
        self.p       = p
        self.inplace = inplace
        
    def forward(self: Dropout2d, x: Tensor) -> Tensor:
        return F.dropout2d(x, self.p, self.training, self.inplace)
    
class Dropout3d(Module):
    def __init__(
        self:    Dropout3d,
        p:       float = 0.5,
        inplace: bool  = False
    ) -> None:
        super().__init__()
        self.p       = p
        self.inplace = inplace
        
    def forward(self: Dropout3d, x: Tensor) -> Tensor:
        return F.dropout3d(x, self.p, self.training, self.inplace)


