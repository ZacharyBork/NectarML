from __future__ import annotations

import builtins
from   typing          import Literal, Any
from   collections.abc import Callable

import nectarml.nn.functional as F
from nectarml.core      import Tensor
from nectarml.nn.module import Module

## ABSTRACT ###

class LossModule(Module):
    def __init__(
        self:      LossModule, 
        loss_fn:   Callable[[Any], Tensor],
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        super().__init__()
        if save_history and reduction == 'none':
            raise ValueError(
                'Loss history not available when reduction mode is "none".')
        
        self.loss_fn      = loss_fn
        self.reduction    = reduction
        self.save_history = save_history
        self.history      = []
        
    def set_history_enabled(self: LossModule, enabled: bool) -> None:
        self.save_history = enabled
        
    def clear_history(self: LossModule) -> None:
        self.history.clear()
   
    def forward(self: LossModule, *args, **kwargs) -> Tensor:
        loss = self.loss_fn(*args, **kwargs)
        if self.save_history: self.history.append(loss.item())
        return loss

### REGRESSION ###

class L1Loss(LossModule):
    def __init__(
        self:      L1Loss, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        '''L1 (Mean Absolute Error) loss.
        
        Pixel-wise loss. Computes error from the absoulte distance between the
        prediction and the ground truth.
        
        Args:
            reduction    : The reduction method to use for the resulting loss 
                           tensor. Options are ['mean', 'sum', 'none'].
                           
            save_history : If True, the loss values calculated by this module
                           will be stored in an internal list which can be
                           accessed with `LossModule.history`.
        '''
        super().__init__(
            loss_fn      = lambda x, y : F.l1_loss(x, y, reduction),
            reduction    = reduction,
            save_history = save_history
        )

MAELoss = L1Loss
    
class L2Loss(LossModule):
    def __init__(
        self:      L2Loss, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        '''L2 (Mean Squared Error) loss.
        
        Computes loss from the squared distance between the prediction and the
        ground truth. Punishing large prediction errors more harshly than 
        smaller errors.
        
        Args:
            reduction    : The reduction method to use for the resulting loss 
                           tensor. Options are ['mean', 'sum', 'none'].
                           
            save_history : If True, the loss values calculated by this module
                           will be stored in an internal list which can be
                           accessed with `LossModule.history`.
        '''
        super().__init__(
            loss_fn      = lambda x, y : F.l2_loss(x, y, reduction),
            reduction    = reduction,
            save_history = save_history
        )

MSELoss = L2Loss
    
class RMSELoss(LossModule):
    def __init__(
        self:      RMSELoss, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        '''Root mean square error loss.
        
        Args:
            reduction    : The reduction method to use for the resulting loss 
                           tensor. Options are ['mean', 'sum', 'none'].
                           
            save_history : If True, the loss values calculated by this module
                           will be stored in an internal list which can be
                           accessed with `LossModule.history`.
        '''
        super().__init__(
            loss_fn      = lambda x, y : F.rmse_loss(x, y, reduction),
            reduction    = reduction,
            save_history = save_history
        )
    
class HuberLoss(LossModule):
    def __init__(
        self:      HuberLoss, 
        delta:     builtins.float = 1.0, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        '''Huber loss.
        
        Behaves like MSE for smaller errors, and MAE for larger errors, 
        creating a cost function which is less sensitive to extreme outliers in 
        input data, but less likely to average inputs than MSE.
        
        Args:
            delta        : The transition point between the quadratic and linear
                           regions of the loss function.
                           
            reduction    : The reduction method to use for the resulting loss 
                           tensor. Options are ['mean', 'sum', 'none'].
                           
            save_history : If True, the loss values calculated by this module
                           will be stored in an internal list which can be
                           accessed with `LossModule.history`.
        '''
        super().__init__(
            loss_fn      = lambda x, y : F.huber_loss(x, y, delta, reduction),
            reduction    = reduction,
            save_history = save_history
        )

class LogCoshLoss(LossModule):
    def __init__(
        self:      LogCoshLoss,  
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        '''Log hyperbolic cosine loss.
        
        Can serve as an alternative to standard MSE loss. It is less sensitive 
        to outliers than MSE, and behaves similarly to MAE with smaller losses.
        
        Args:
            reduction    : The reduction method to use for the resulting loss 
                           tensor. Options are ['mean', 'sum', 'none'].
                           
            save_history : If True, the loss values calculated by this module
                           will be stored in an internal list which can be
                           accessed with `LossModule.history`.
        '''
        super().__init__(
            loss_fn      = lambda x, y : F.log_cosh_loss(x, y, reduction),
            reduction    = reduction,
            save_history = save_history
        )

### CLASSIFICATION ###
    
class BCELoss(LossModule):
    def __init__(
        self:      BCELoss,  
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        '''Binary cross-entropy loss.
        
        Also known as log loss. Measures the distance between predictions and 
        actual binary labels. Used as a cost function for binary and multi-
        label classification models.
        
        Args:
            reduction    : The reduction method to use for the resulting loss 
                           tensor. Options are ['mean', 'sum', 'none'].
                           
            save_history : If True, the loss values calculated by this module
                           will be stored in an internal list which can be
                           accessed with `LossModule.history`.
        '''
        super().__init__(
            loss_fn      = lambda x, y : F.bce_loss(x, y, reduction),
            reduction    = reduction,
            save_history = save_history
        )
    
class CrossEntropyLoss(LossModule):
    def __init__(
        self:      CrossEntropyLoss,  
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        '''Cross-entropy loss.
        
        Computes distance between model's prediction and actual labels. 
        Standard cost function for classification models.
        
        Args:
            reduction    : The reduction method to use for the resulting loss 
                           tensor. Options are ['mean', 'sum', 'none'].
                           
            save_history : If True, the loss values calculated by this module
                           will be stored in an internal list which can be
                           accessed with `LossModule.history`.
        '''
        super().__init__(
            loss_fn      = lambda x, y : F.cross_entropy_loss(x, y, reduction),
            reduction    = reduction,
            save_history = save_history
        )

class NLLLoss(LossModule):
    def __init__(
        self:      NLLLoss,  
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        '''Negative Log Likelihood loss.
    
        This loss effectively computes how "surprised" the model was when 
        presented with the correct answer.
        
        Args:
            input        : The model prediction output.
            
            target       : The ground truth. Target for model's prediction.
            
            reduction    : The reduction method to use for the resulting loss 
                           tensor. Options are ['mean', 'sum', 'none'].
                           
            save_history : If True, the loss values calculated by this module
                           will be stored in an internal list which can be
                           accessed with `LossModule.history`.
        '''
        super().__init__(
            loss_fn      = lambda x, y : F.nll_loss(x, y, reduction),
            reduction    = reduction,
            save_history = save_history
        )
    
class HingeLoss(LossModule):
    def __init__(
        self:      HingeLoss,  
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        super().__init__(
            loss_fn      = lambda x, y : F.hinge_loss(x, y, reduction),
            reduction    = reduction,
            save_history = save_history
        )
    
class Hinge2Loss(LossModule):
    def __init__(
        self:      Hinge2Loss,  
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        super().__init__(
            loss_fn      = lambda x, y : F.hinge2_loss(x, y, reduction),
            reduction    = reduction,
            save_history = save_history
        )
    
### PROBABILISTIC ###

class KLDivergenceLoss(LossModule):
    def __init__(
        self:      KLDivergenceLoss,  
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        super().__init__(
            loss_fn      = lambda x, y : F.kl_divergence_loss(x, y, reduction),
            reduction    = reduction,
            save_history = save_history
        )
    
class BCEWithLogitsLoss(LossModule):
    def __init__(
        self:      BCEWithLogitsLoss,  
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        super().__init__(
            loss_fn      = \
                lambda x, y : F.bce_with_logits_loss(x, y, reduction),
            reduction    = reduction,
            save_history = save_history
        )
    
### RANKING ###

class TripletMarginLoss(LossModule):
    def __init__(
        self:      TripletMarginLoss, 
        margin:    builtins.float = 1.0,
        eps:       builtins.float = 1e-6, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        save_history: bool = False
    ) -> None:
        loss_fn = lambda anchor, positive, negative : \
            F.triplet_margin_loss(
                anchor, positive, negative, margin, eps, reduction)
            
        super().__init__(
            loss_fn      = loss_fn,
            reduction    = reduction,
            save_history = save_history
        )

