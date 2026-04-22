import builtins
from typing import Literal

from nectarml.tensor              import Tensor
from nectarml.typing              import float32, int32
from nectarml.functional          import math
from nectarml.functional.indexing import where, gather

# ABSTRACTS

def _reduce_loss(
    loss_value: Tensor, 
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    match reduction:
        case 'none': return loss_value
        case 'mean': return loss_value.mean()
        case 'sum':  return loss_value.sum()
        case _: raise ValueError(f'Invalid reduction mode: {reduction}')

# LOSS - REGRESSION

def l1_loss(
    input:     Tensor, 
    target:    Tensor, 
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    '''L1 (Mean Absolute Error) loss.
    
    Pixel-wise loss. Computes error from the absoulte distance between the
    prediction and the ground truth.
    
    Args:
        input : The model prediction output.
        target : The ground truth. Target for model's prediction.
        
    Returns:
        Tensor : The computed loss.
    '''
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target.to(dtype=float32)
    out         = _reduce_loss((x - y).abs(), reduction)
    return out.to(dtype=input_dtype)

def mae_loss(
    input:     Tensor, 
    target:    Tensor, 
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    '''L1 (Mean Absolute Error) loss.
    
    Pixel-wise loss. Computes error from the absoulte distance between the
    prediction and the ground truth.
    
    Args:
        input : The model prediction output.
        target : The ground truth. Target for model's prediction.
        
    Returns:
        Tensor : The computed loss.
    '''
    return l1_loss(input, target, reduction)

def l2_loss(
    input:     Tensor, 
    target:    Tensor, 
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    '''L2 (Mean Squared Error) loss.
    
    Computes loss from the squared distance between the prediction and the
    ground truth. Punishing large prediction errors more harshly than smaller
    errors.
    
    Args:
        input : The model prediction output.
        target : The ground truth. Target for model's prediction.
        
    Returns:
        Tensor : The computed loss.
    '''
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target.to(dtype=float32)
    loss_value  = (x - y) ** 2
    out         = _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)

def mse_loss(
    input:     Tensor, 
    target:    Tensor, 
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    '''L2 (Mean Squared Error) loss.
    
    Computes loss from the squared distance between the prediction and the
    ground truth. Punishing large prediction errors more harshly than smaller
    errors.
    
    Args:
        input : The model prediction output.
        target : The ground truth. Target for model's prediction.
        
    Returns:
        Tensor : The computed loss.
    ''' 
    return l2_loss(input, target, reduction)

def rmse_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target.to(dtype=float32)
    loss_value  = mse_loss(x, y, reduction='none').sqrt()
    out         = _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)

def huber_loss(
    input:     Tensor, 
    target:    Tensor, 
    delta:     builtins.float = 1.0,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    distance   = input - target
    delta      = delta if delta is not None else 1.0
    quadratic  = 0.5 * (distance ** 2)
    linear     = delta * (distance.abs() - 0.5 * delta)
    loss_value = where((distance).abs() < delta, quadratic, linear)
    return _reduce_loss(loss_value, reduction)

def log_cosh_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target.to(dtype=float32)
    loss_value  = (x - y).cosh().log()
    out         = _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)

# LOSS - CLASSIFICATION

def bce_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target.to(dtype=float32)
    loss_value  = -(y * x.log() + (1 - y) * (1 - x).log())
    out         =  _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)

def cross_entropy_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    input_dtype = input.dtype
    x = input.to(dtype=float32)
    
    x           = x - x.max(dim=1, keepdim=True).values
    log_softmax = x - x.exp().sum(dim=1, keepdim=True).log()

    B = x.shape[0]    
    target_int = target.to(dtype=int32)
    nll        = -log_softmax.gather(dim=1, index=target_int.reshape(B, 1))    
    return _reduce_loss(nll.reshape(B), reduction).to(dtype=input_dtype)

def nll_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    '''Negative Log Likelihood loss.
    
    This loss effectively computes how "surprised" the model was when presented
    with the correct answer.
    
    Args:
        input : The model prediction output.
        target : The ground truth. Target for model's prediction.
        
    Returns:
        Tensor : The computed loss.
    '''
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target
    loss_value  = -(gather(x, dim=1, index=y)).log()
    out         = _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)

def hinge_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    return _reduce_loss(
        math.maximum(1 - target * input), reduction, 0.0)

def hinge2_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    loss_value = math.maximum(1 - target * input, 0.0) ** 2
    return _reduce_loss(loss_value, reduction)

# LOSS - PROBABILISTIC

def kl_divergence_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'sum'
) -> Tensor:
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target.to(dtype=float32)
    safe_y     = y.clamp(min_value=1e-8)
    loss_value = safe_y * (safe_y.log() - x)
    out        = _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)

def bce_with_logits_loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target.to(dtype=float32)
    loss_value  = math.maximum(x, 0.0) - x * y + (1.0 + (-abs(x)).exp()).log()
    out         = _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)

# LOSS - RANKING

def triplet_margin_loss(
    anchor:    Tensor, 
    positive:  Tensor, 
    negative:  Tensor,
    margin:    builtins.float = 1.0,
    eps:       builtins.float = 1e-6,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    assert margin > 0.0
    input_dtype = anchor.dtype
    
    a = anchor.to(dtype=float32)
    p = positive.to(dtype=float32)
    n = negative.to(dtype=float32)
    
    dist       = lambda x, y: (((x - y) ** 2).sum() + eps).sqrt()
    loss_value = math.maximum(dist(a, p) - dist(a, n) + margin, 0.0)
    out        = _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)


