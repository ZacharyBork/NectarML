import builtins
from typing import Literal

from nectarml.tensor import Tensor
from nectarml.typing import float32
from nectarml.functional.reductions import mean, sum
from nectarml.functional.math import sqrt, log, exp, cosh, maximum
from nectarml.functional.indexing import where, gather

# ABSTRACTS

def _reduce_loss(
    loss_value: Tensor, 
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    match reduction:
        case 'none': return loss_value
        case 'mean': return mean(loss_value)
        case 'sum':  return sum(loss_value)
        case _: raise ValueError(f'Invalid reduction mode: {reduction}')

# LOSS - REGRESSION

def L1Loss(
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
    out         = _reduce_loss(abs((x - y)), reduction)
    return out.to(dtype=input_dtype)

def MAELoss(
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
    return L1Loss(input, target, reduction)

def L2Loss(
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

def MSELoss(
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
    return L2Loss(input, target, reduction)

def RMSELoss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target.to(dtype=float32)
    loss_value  = sqrt(MSELoss(x, y, reduction='none'))
    out         = _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)

def HuberLoss(
    input:     Tensor, 
    target:    Tensor, 
    delta:     builtins.float = 1.0,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    distance   = input - target
    quadratic  = 0.5 * (distance ** 2)
    linear     = delta * (abs(distance) - 0.5 * delta)
    loss_value = where((abs(distance)).data < delta, quadratic, linear)
    return _reduce_loss(loss_value, reduction)

def LogCoshLoss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target.to(dtype=float32)
    loss_value  = log(cosh(x - y))
    out         = _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)

# LOSS - CLASSIFICATION

def BCELoss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target.to(dtype=float32)
    loss_value  = -(y * log(x) + (1 - y) * log(1 - x))
    out         =  _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)

def CrossEntropyLoss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target.to(dtype=float32)
    out         = _reduce_loss(-sum(y * log(x)), reduction)
    return out.to(dtype=input_dtype)

def NLLLoss(
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
    loss_value  = -log(gather(x, dim=1, index=y))
    out         = _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)

def HingeLoss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    return _reduce_loss(
        maximum(1 - target * input), reduction, 0.0)

def Hinge2Loss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    loss_value = maximum(1 - target * input, 0.0) ** 2
    return _reduce_loss(loss_value, reduction)

# LOSS - PROBABILISTIC

def KLDivergenceLoss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'sum'
) -> Tensor:
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target.to(dtype=float32)
    loss_value  = y * log(y / x)
    out         = _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)

def BCEWithLogitsLoss(
    input:     Tensor, 
    target:    Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    input_dtype = input.dtype
    x, y        = input.to(dtype=float32), target.to(dtype=float32)
    loss_value  = maximum(x, 0.0) - x * y + log(1.0 + exp(-abs(x)))
    out         = _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)

# LOSS - RANKING

def TripletMarginLoss(
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
    
    dist       = lambda x, y: sqrt(sum((x - y) ** 2) + eps)
    loss_value = maximum(dist(a, p) - dist(a, n) + margin, 0.0)
    out        = _reduce_loss(loss_value, reduction)
    return out.to(dtype=input_dtype)


