from __future__ import annotations

from nectarml           import typing
from nectarml.core      import Tensor, creation
from nectarml.nn.module import Module
from nectarml.nn.init   import kaiming_normal_

class Linear(Module):
    def __init__(
        self:          Linear,
        in_features:   int,
        out_features:  int,
        bias:          bool = True,
        dtype: typing.dtype = typing.float32
    ) -> None:
        '''Fully connected neural network layer.
        
        A layer in which every input is connected to every output by a 
        learnable weight parameter.
        
        Args:
            in_features  : The number of input features for the layer.
            out_features : The number of output features for the layer.
            bias         : If True, adds a learnable bias parameter to the 
                           layer.
            dtype        : The DType to use when initializing the layers
                           weight and bias tensors.
        '''
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        
        self.weight = creation.empty(
            (self.out_features, self.in_features), 
            dtype=dtype, requires_grad=True)
        kaiming_normal_(
            weights=self.weight, mode='fan_in', nonlinearity='linear')
        
        if bias:
            self.bias = creation.zeros(
                (out_features,), dtype=dtype, requires_grad=True)
        else: self.bias = None

    def forward(self: Linear, x: Tensor) -> Tensor:
        y = x @ self.weight.transpose(1, 0)
        if self.bias is not None: y = self.bias + y
        return y

class LazyLinear(Linear):
    def __init__(
        self:          LazyLinear, 
        out_features:  int,
        bias:          bool = True,
        dtype: typing.dtype = typing.float32
    ) -> None:
        Module.__init__(self)
        self.out_features = out_features
        self._do_bias     = bias
        self.dtype        = dtype
        self._initialized = False
        
    def has_uninitialized_params(self: LazyLinear) -> bool:
        return not self._initialized
    
    def _initialize(
        self:        LazyLinear, 
        in_features: int,
        device:      typing.DeviceLikeType
    ) -> None:
        super().__init__(
            in_features, self.out_features, self._do_bias, self.dtype)
        self.weight = self.weight.to(device)
        self.bias   = self.bias.to(device) if self.bias is not None else None
        self._initialized = True
        
    def forward(self: LazyLinear, x: Tensor) -> Tensor:
        if self.has_uninitialized_params(): 
            self._initialize(x.shape[-1], x.device)
        return super().forward(x)
        
