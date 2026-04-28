import builtins
from typing          import Literal, Any
from collections.abc import Iterable

from nectarml.core                   import Tensor
from nectarml.typing                 import float32, int32
from nectarml.nn.functional          import math as tensor_math
from nectarml.nn.functional.indexing import where, gather

# UTILITIES

def _reduce_loss(
    loss:      Tensor, 
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor:
    match reduction:
        case 'none': return loss
        case 'mean': return loss.mean()
        case 'sum':  return loss.sum()
        case _: raise ValueError(f'Invalid reduction mode: {reduction}')
        
def _prep_inputs(*inputs: Any | Iterable[Any]) -> tuple[Any]:
    return tuple([
        x.to(dtype=float32) if isinstance(x, Tensor) 
        else x for x in list(inputs)]
    )
    
# LOSS - REGRESSION

def l1_loss(
    input:     Tensor, 
    target:    Tensor | builtins.float, 
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor:
    '''L1 (Mean Absolute Error) loss.
    
    Pixel-wise loss. Computes error from the absoulte distance between the
    prediction and the ground truth.
    
    Args:
        input     : The model prediction output.
        target    : The ground truth. Target for model's prediction.
        reduction : The reduction method to use for the resulting loss tensor.
                    Options are ['mean', 'sum', 'none'].
        
    Returns:
        Tensor : The computed loss.
    '''
    x, y = _prep_inputs(input, target)
    return _reduce_loss((x - y).abs(), reduction)

def mae_loss(
    input:     Tensor, 
    target:    Tensor, 
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor:
    '''L1 (Mean Absolute Error) loss.
    
    Pixel-wise loss. Computes error from the absoulte distance between the
    prediction and the ground truth.
    
    Args:
        input     : The model prediction output.
        target    : The ground truth. Target for model's prediction.
        reduction : The reduction method to use for the resulting loss tensor.
                    Options are ['mean', 'sum', 'none'].
        
    Returns:
        Tensor : The computed loss.
    '''
    return l1_loss(input, target, reduction)

def l2_loss(
    input:     Tensor, 
    target:    Tensor | builtins.float, 
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor:
    '''L2 (Mean Squared Error) loss.
    
    Computes loss from the squared distance between the prediction and the
    ground truth. Punishing large prediction errors more harshly than smaller
    errors.
    
    Args:
        input     : The model prediction output.
        target    : The ground truth. Target for model's prediction.
        reduction : The reduction method to use for the resulting loss tensor.
                    Options are ['mean', 'sum', 'none'].
        
    Returns:
        Tensor : The computed loss.
    '''
    x, y = _prep_inputs(input, target)
    return _reduce_loss((x - y) ** 2, reduction)

def mse_loss(
    input:     Tensor, 
    target:    Tensor | builtins.float, 
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor:
    '''L2 (Mean Squared Error) loss.
    
    Computes loss from the squared distance between the prediction and the
    ground truth. Punishing large prediction errors more harshly than smaller
    errors.
    
    Args:
        input     : The model prediction output.
        target    : The ground truth. Target for model's prediction.
        reduction : The reduction method to use for the resulting loss tensor.
                    Options are ['mean', 'sum', 'none'].
        
    Returns:
        Tensor : The computed loss.
    ''' 
    return l2_loss(input, target, reduction)

def rmse_loss(
    input:     Tensor, 
    target:    Tensor | builtins.float,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor:
    '''Root mean square error loss.
    
    Args:
        input     : The model prediction output.
        target    : The ground truth. Target for model's prediction.
        reduction : The reduction method to use for the resulting loss tensor.
                    Options are ['mean', 'sum', 'none'].
        
    Returns:
        Tensor : The computed loss.
    '''
    x, y       = input.to(dtype=float32), target.to(dtype=float32)
    loss_value = mse_loss(x, y, reduction='none').sqrt()
    return _reduce_loss(loss_value, reduction)

def huber_loss(
    input:     Tensor, 
    target:    Tensor | builtins.float, 
    delta:     builtins.float = 1.0,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor: 
    '''Huber loss.
    
    Behaves like MSE for smaller errors, and MAE for larger errors, creating
    a cost function which is less sensitive to extreme outliers in input data,
    but less likely to average inputs than MSE.
    
    Args:
        input     : The model prediction output.
        target    : The ground truth. Target for model's prediction.
        delta     : The transition point between the quadratic and linear
                    regions of the loss function.
        reduction : The reduction method to use for the resulting loss tensor.
                    Options are ['mean', 'sum', 'none'].
        
    Returns:
        Tensor : The computed loss.
    '''
    x, y       = _prep_inputs(input, target)
    distance   = x - y
    delta      = delta if delta is not None else 1.0
    quadratic  = 0.5 * (distance ** 2)
    linear     = delta * (distance.abs() - 0.5 * delta)
    loss_value = where((distance).abs() < delta, quadratic, linear)
    return _reduce_loss(loss_value, reduction)

def log_cosh_loss(
    input:     Tensor, 
    target:    Tensor | builtins.float,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor: 
    '''Log hyperbolic cosine loss.
    
    Can serve as an alternative to standard MSE loss. It is less sensitive to
    outliers than MSE, and behaves similarly to MAE with smaller losses.
    
    Args:
        input     : The model prediction output.
        target    : The ground truth. Target for model's prediction.
        reduction : The reduction method to use for the resulting loss tensor.
                    Options are ['mean', 'sum', 'none'].
        
    Returns:
        Tensor : The computed loss.
    '''
    x, y = _prep_inputs(input, target)
    return _reduce_loss((x - y).cosh().log(), reduction)

# LOSS - CLASSIFICATION

def bce_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor: 
    '''Binary cross-entropy loss.
    
    Measures the distance between predictions and actual binary labels. Used as
    a cost function for binary and multi-label classification models.
    
    Args:
        input     : The model prediction output.
        target    : The ground truth. Target for model's prediction.
        reduction : The reduction method to use for the resulting loss tensor.
                    Options are ['mean', 'sum', 'none'].
        
    Returns:
        Tensor : The computed loss.
    '''
    x, y = _prep_inputs(input, target)
    loss = -(y * x.log() + (1 - y) * (1 - x).log())
    return _reduce_loss(loss, reduction)

def cross_entropy_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor: 
    '''Cross-entropy loss.
    
    Computes distance between model's prediction and actual labels. Standard
    cost function for classification models.
    
    Args:
        input     : The model prediction output.
        target    : The ground truth. Target for model's prediction.
        reduction : The reduction method to use for the resulting loss tensor.
                    Options are ['mean', 'sum', 'none'].
        
    Returns:
        Tensor : The computed loss.
    '''
    x = _prep_inputs(input)
    
    x           = x - x.max(dim=1, keepdim=True).values
    log_softmax = x - x.exp().sum(dim=1, keepdim=True).log()

    B = x.shape[0]    
    target_int = target.to(dtype=int32)
    nll        = -log_softmax.gather(dim=1, index=target_int.reshape(B, 1))    
    return _reduce_loss(nll.reshape(B), reduction)

def nll_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor: 
    '''Negative Log Likelihood loss.
    
    This loss effectively computes how "surprised" the model was when presented
    with the correct answer.
    
    Args:
        input     : The model prediction output.
        target    : The ground truth. Target for model's prediction.
        reduction : The reduction method to use for the resulting loss tensor.
                    Options are ['mean', 'sum', 'none'].
        
    Returns:
        Tensor : The computed loss.
    '''
    x, y = _prep_inputs(input, target)
    loss = -(gather(x, dim=1, index=y)).log()
    return _reduce_loss(loss, reduction)

def hinge_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor:
    x, y = _prep_inputs(input, target)
    return _reduce_loss(tensor_math.maximum(1 - y * x, 0.0), reduction)

def hinge2_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor:
    x, y = _prep_inputs(input, target)
    loss = tensor_math.maximum(1 - y * x, 0.0) ** 2
    return _reduce_loss(loss, reduction)

# LOSS - PROBABILISTIC

def kl_divergence_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'sum'
) -> Tensor:
    x, y   = _prep_inputs(input, target)
    safe_y = y.clamp(min_value=1e-8)
    loss   = safe_y * (safe_y.log() - x)
    return _reduce_loss(loss, reduction)

def bce_with_logits_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor:
    x, y = _prep_inputs(input, target)
    loss = tensor_math.maximum(x, 0.0) - x * y + (1.0 + (-abs(x)).exp()).log()
    return _reduce_loss(loss, reduction)

# LOSS - RANKING

def triplet_margin_loss(
    anchor:    Tensor, 
    positive:  Tensor, 
    negative:  Tensor,
    margin:    builtins.float = 1.0,
    eps:       builtins.float = 1e-6,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> Tensor: 
    assert margin > 0.0
        
    a, p, n = _prep_inputs(anchor, positive, negative)
        
    a = anchor.to(dtype=float32)
    p = positive.to(dtype=float32)
    n = negative.to(dtype=float32)
    
    dist       = lambda x, y: (((x - y) ** 2).sum() + eps).sqrt()
    loss_value = tensor_math.maximum(dist(a, p) - dist(a, n) + margin, 0.0)
    return _reduce_loss(loss_value, reduction)


