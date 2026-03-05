from typing import Literal

from nectarml.tensor import Tensor
from nectarml.creation import zeros_like, ones_like, zeros
from nectarml.functional.reductions import mean, sum
from nectarml.functional.math import sqrt, log, exp, cosh
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
    input: Tensor, 
    target: Tensor, 
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
    return _reduce_loss(abs((input - target)), reduction)

def MAELoss(
    input: Tensor, 
    target: Tensor, 
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
    input: Tensor, 
    target: Tensor, 
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
    return _reduce_loss((input - target) ** 2, reduction)

def MSELoss(
    input: Tensor, 
    target: Tensor, 
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
    input: Tensor, 
    target: Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    loss_value = sqrt(MSELoss(input, target, reduction='none'))
    return _reduce_loss(loss_value, reduction)

def HuberLoss(
    input: Tensor, 
    target: Tensor, 
    delta: float = 1.0,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    distance = input - target
    quadratic = 0.5 * (distance ** 2)
    linear = delta * (abs(distance) - 0.5 * delta)
    loss_value = where((abs(distance)).data < delta, quadratic, linear)
    return _reduce_loss(loss_value, reduction)

def LogCoshLoss(
    input: Tensor, 
    target: Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    return _reduce_loss(log(cosh(input - target)), reduction)

# LOSS - CLASSIFICATION

def BCELoss(
    input: Tensor, 
    target: Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    loss_value = -(target * log(input) + (1 - target) * log(1 - input))
    return _reduce_loss(loss_value, reduction)

def CrossEntropyLoss(
    input: Tensor, 
    target: Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    return _reduce_loss(-sum(target * log(input)), reduction)

def NLLLoss(
    input: Tensor, 
    target: Tensor,
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
    return _reduce_loss(-log(gather(input, dim=1, index=target)), reduction)

def HingeLoss(
    input: Tensor, 
    target: Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    return _reduce_loss(max(zeros_like(input), 1 - target * input), reduction)

def Hinge2Loss(
    input: Tensor, 
    target: Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    loss_value = max(zeros_like(input), 1 - target * input) ** 2
    return _reduce_loss(loss_value, reduction)

# LOSS - PROBABILISTIC

def KLDivergenceLoss(
    input: Tensor, 
    target: Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'sum'
) -> Tensor:
    return _reduce_loss(target * log(target / input), reduction)

def BCEWithLogitsLoss(
    input: Tensor, 
    target: Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor:
    x = input
    _zeros = zeros_like(x)
    _ones = ones_like(x)
    loss_value = max(x, _zeros) - x * target + log(_ones + exp(-abs(x)))
    return _reduce_loss(loss_value, reduction)

# LOSS - RANKING

def TripletMarginLoss(
    anchor: Tensor, 
    positive: Tensor, 
    negative: Tensor,
    margin: float = 1.0,
    eps: float = 1e-6,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> Tensor: 
    assert margin > 0.0
    a, p, n = anchor, positive, negative
    zero = zeros((), dtype=anchor.dtype, device=anchor.device)
    dist = lambda x, y: sqrt(sum((x - y) ** 2) + eps)
    loss_value = max(dist(a, p) - dist(a, n) + margin, zero)
    return mean(loss_value, reduction)

