from __future__ import annotations

import nectarml.nn.functional as F
from nectarml.core      import Tensor
from nectarml.nn.module import Module

### STANDARD DROPOUT ###

class Dropout(Module):
    def __init__(
        self:    Dropout,
        p:       float = 0.5,
        inplace: bool  = False
    ) -> None:
        '''Randomly disables a portion of neurons.

        Applies a random binary mask to the input tensor to zero values based 
        on the given probability (`p`). This is a common regularization 
        technique to prevent model overfitting.
        
        Args:
            p        : The probability (0-1) of dropout being applied to any 
                       given activation.
            inplace : If True, the dropout function will modify the input 
                      tensor in-place. If False, it will create a new tensor to
                      serve as output. 
                      
                      If the input tensor has `requires_grad`=True, a new
                      tensor will be created regardless of the value of 
                      `inplace`, since in-place modifications to tensor data 
                      break the computation graph.
        '''
        super().__init__()
        self.p       = p
        self.inplace = inplace
        
    def forward(self: Dropout, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training, self.inplace)

class Dropout1d(Module):
    def __init__(
        self:    Dropout1d,
        p:       float = 0.5,
        inplace: bool  = False
    ) -> None:
        '''Randomly disables a channels by a given probability.

        Expects tensor of shape (B, C, L). Applies a random binary mask along
        the input tensors channel dimension to zero values of entire channels 
        based on the given probability (`p`). This is a common regularization 
        technique to prevent model overfitting.
        
        Args:
            p       : The probability (0-1) of dropout being applied to any 
                      given channel of the input tensor.
            inplace : If True, the dropout function will modify the input 
                      tensor in-place. If False, it will create a new tensor to
                      serve as output. 
                    
                      If the input tensor has `requires_grad`=True, a new
                      tensor will be created regardless of the value of 
                      `inplace`, since in-place modifications to tensor data 
                      break the computation graph.
        '''
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
        '''Randomly disables a channels by a given probability.

        Expects tensor of shape (B, C, H, W). Applies a random binary mask 
        along the input tensors channel dimension to zero values of entire
        channels based on the given probability (`p`). This is a common 
        regularization technique to prevent model overfitting.
        
        Args:
            p       : The probability (0-1) of dropout being applied to any
                      given channel of the input tensor.
            inplace : If True, the dropout function will modify the input 
                      tensor in-place. If False, it will create a new tensor 
                      to serve as output. 
                    
                      If the input tensor has `requires_grad`=True, a new
                      tensor will be created regardless of the value of 
                      `inplace`, since in-place modifications to tensor data 
                      break the computation graph.
        '''
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
        '''Randomly disables a channels by a given probability.

        Expects tensor of shape (B, C, D, H, W). Applies a random binary mask 
        along the input tensors channel dimension to zero values of entire 
        channels based on the given probability (`p`). This is a common
        regularization technique to prevent model overfitting.
        
        Args:
            p       : The probability (0-1) of dropout being applied to any 
                      given channel of the input tensor.
            inplace : If True, the dropout function will modify the input 
                      tensor in-place. If False, it will create a new tensor to
                      serve as output. 
                    
                      If the input tensor has `requires_grad`=True, a new
                      tensor will be created regardless of the value of 
                      `inplace`, since in-place modifications to tensor data 
                      break the computation graph.
        '''
        super().__init__()
        self.p       = p
        self.inplace = inplace
        
    def forward(self: Dropout3d, x: Tensor) -> Tensor:
        return F.dropout3d(x, self.p, self.training, self.inplace)

### ALPHA DROPOUT ###

class AlphaDropout(Module):
    def __init__(
        self:    AlphaDropout,
        p:       float = 0.5,
        inplace: bool  = False
    ) -> None:
        '''Randomly scales activations to the negative saturation of SeLU.

        Unlike standard dropout, which zeros values of the dropped activations, 
        Alpha Dropout instead scales the dropped activations by the negative 
        saturation value of the SeLU activation function. This allows it to 
        maintain the zero mean and unit variance of the input tensor.
        
        Args:
            p       : The probability (0-1) of dropout being applied to any 
                      given activation.
            inplace : If True, the dropout function will modify the input 
                      tensor in-place. If False, it will create a new tensor to 
                      serve as output. 
                    
                      If the input tensor has `requires_grad`=True, a new
                      tensor will be created regardless of the value of 
                      `inplace`, since in-place modifications to tensor data 
                      break the computation graph.
        '''
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
        '''Randomly scales input channels to the negative saturation of SeLU.

        Unlike standard dropout, which zeros values of the dropped activations, 
        Feature Alpha Dropout instead scales the dropped activations by the 
        negative saturation value of the SeLU activation function. This allows
        it to maintain the zero mean and unit variance of the input tensor.
        
        Args:
            p       : The probability (0-1) of dropout being applied to any 
                      given channel of the input tensor.
            inplace : If True, the dropout function will modify the input 
                      tensor in-place. If False, it will create a new tensor to 
                      serve as output. 
                    
                      If the input tensor has `requires_grad`=True, a new
                      tensor will be created regardless of the value of
                      `inplace`, since in-place modifications to tensor data 
                      break the computation graph.
        '''
        super().__init__()
        self.p       = p
        self.inplace = inplace
        
    def forward(self: FeatureAlphaDropout, x: Tensor) -> Tensor:
        return F.feature_alpha_dropout(x, self.p, self.training, self.inplace)

