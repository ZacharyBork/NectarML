from nectarml.core import Tensor
from nectarml.creation import rand

def dropout(
    input:    Tensor,
    p:        float = 0.5,
    training: bool  = True,
    inplace:  bool  = False
) -> Tensor:
    assert 0.0 <= p <= 1.0, 'Probability must be between 0.0 and 1.0.'
    if not training or p == 0.0: return input
    if p == 1.0: return input * 0.0

    mask = (rand(input.shape, device=input.device) > p).to(dtype=input.dtype)
    mask = mask.detach()
    out  = input * mask / (1.0 - p)

    if inplace and not input.requires_grad: return input.copy_(out)
    return out

def alpha_dropout(
    input:  Tensor,
    p:        float = 0.5,
    training: bool  = True,
    inplace:  bool  = False
) -> Tensor:
    assert 0.0 <= p <= 1.0, 'Probability must be between 0.0 and 1.0.'
    if not training or p == 0.0: return input
    if p == 1.0: return input * 0.0

    alpha_prime = -1.7580993408473766
    keep_prob = 1 - p

    a = (keep_prob + alpha_prime**2 * keep_prob * (1-keep_prob))**(-0.5)
    b = -a * (alpha_prime * (1-keep_prob))

    mask = (rand(input.shape, device=input.device) > p).to(dtype=input.dtype)
    mask = mask.detach()
    dropped = mask * input + (1.0 - mask) * alpha_prime
    out = a * dropped + b

    if inplace and not input.requires_grad: return input.copy_(out)
    return out

def feature_alpha_dropout(
    input:    Tensor,
    p:        float = 0.5,
    training: bool  = True,
    inplace:  bool  = False
) -> Tensor:
    assert 0.0 <= p <= 1.0, 'Probability must be between 0.0 and 1.0.'
    assert input.ndim >= 3, 'feature_alpha_dropout expects at least 3D input'
    if not training or p == 0.0: return input
    if p == 1.0: return input * 0.0

    alpha_prime = -1.7580993408473766
    keep_prob = 1 - p

    a = (keep_prob + alpha_prime**2 * keep_prob * (1-keep_prob))**(-0.5)
    b = -a * (alpha_prime * (1-keep_prob))

    mask_shape = (input.shape[0], input.shape[1]) + (1,) * len(input.shape[2:])
    mask = (rand(mask_shape, device=input.device) > p).to(dtype=input.dtype)
    mask = mask.detach().broadcast_to(input.shape)
    dropped = mask * input + (1.0 - mask) * alpha_prime
    out = a * dropped + b
    
    if inplace and not input.requires_grad: return input.copy_(out)
    return out

def dropout1d(
    input:    Tensor,
    p:        float = 0.5,
    training: bool  = True,
    inplace:  bool  = False
) -> Tensor:
    assert 0.0 <= p <= 1.0, 'Probability must be between 0.0 and 1.0.'
    assert input.ndim == 3, 'dropout1d expects 3D input [B, C, L]'
    if not training or p == 0.0: return input
    if p == 1.0: return input * 0.0
    
    mask_shape = (input.shape[0], input.shape[1], 1)
    mask = (rand(mask_shape, device=input.device) > p)
    mask = mask.detach().to(dtype=input.dtype).broadcast_to(input.shape)
    out  = input * mask / (1.0 - p)
    
    if inplace and not input.requires_grad: return input.copy_(out)
    return out

def dropout2d(
    input:    Tensor,
    p:        float = 0.5,
    training: bool  = True,
    inplace:  bool  = False
) -> Tensor:
    assert 0.0 <= p <= 1.0, 'Probability must be between 0.0 and 1.0.'
    assert input.ndim == 4, 'dropout2d expects 4D input [B, C, H, W]'
    if not training or p == 0.0: return input
    if p == 1.0: return input * 0.0
    
    mask_shape = (input.shape[0], input.shape[1], 1, 1)
    mask = (rand(mask_shape, device=input.device) > p)
    mask = mask.detach().to(dtype=input.dtype).broadcast_to(input.shape)
    out  = input * mask / (1.0 - p)
    
    if inplace and not input.requires_grad: return input.copy_(out)
    return out

def dropout3d(
    input:    Tensor,
    p:        float = 0.5,
    training: bool  = True,
    inplace:  bool  = False
) -> Tensor:
    assert 0.0 <= p <= 1.0, 'Probability must be between 0.0 and 1.0.'
    assert input.ndim == 5, 'dropout3d expects 4D input [B, C, D, H, W]'
    if not training or p == 0.0: return input
    if p == 1.0: return input * 0.0
    
    mask_shape = (input.shape[0], input.shape[1], 1, 1, 1)
    mask = (rand(mask_shape, device=input.device) > p)
    mask = mask.detach().to(dtype=input.dtype).broadcast_to(input.shape)
    out  = input * mask / (1.0 - p)
    
    if inplace and not input.requires_grad: return input.copy_(out)
    return out


