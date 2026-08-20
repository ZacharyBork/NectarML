from nectarml.core   import Tensor, creation
from nectarml.typing import ShapeType

### STANDARD DROPOUT ###

def _dropout(
    input:      Tensor,
    mask_shape: ShapeType,
    p:          float,
    inplace:    bool
) -> Tensor:
    '''Abstract dropout function with configurable mask shape.'''
    mask = (creation.rand(input.shape, device=input.device) > p)
    mask = mask.to(dtype=input.dtype).detach()
    if mask_shape != input.shape: 
        mask = mask.broadcast_to(input.shape)
    
    out = input * mask / (1.0 - p)
    if inplace and not input.requires_grad: return input.copy_(out)
    return out

def dropout(
    input:    Tensor,
    p:        float = 0.5,
    training: bool  = True,
    inplace:  bool  = False
) -> Tensor:
    '''Randomly disables a portion of neurons.

    Applies a random binary mask to the input tensor to zero values based on
    the given probability (`p`). This is a common regularization technique to
    prevent model overfitting.
    
    Args:
        input    : The tensor to apply the dropout to.
        p        : The probability (0-1) of dropout being applied to any given
                   activation.
        training : If True, this function will act as a no-op, passing its
                   input through unchanged. You generally only want to enable
                   dropout during training, and disable it during inference and
                   testing of models.
        inplace  : If True, the dropout function will modify the input tensor
                   in-place. If False, it will create a new tensor to serve as
                   output. 
                   
                   If the input tensor has `requires_grad`=True, a new
                   tensor will be created regardless of the value of `inplace`,
                   since in-place modifications to tensor data break the 
                   computation graph.
            
    Returns:
        Tensor : The resulting tensor from the dropout operation.
    '''
    assert 0.0 <= p <= 1.0, 'Probability must be between 0.0 and 1.0.'
    
    if not training or p == 0.0: return input
    if p == 1.0:                 return input * 0.0
    
    return _dropout(input, input.shape, p, inplace)

def dropout1d(
    input:    Tensor,
    p:        float = 0.5,
    training: bool  = True,
    inplace:  bool  = False
) -> Tensor:
    '''Randomly disables a channels by a given probability.

    Expects tensor of shape (B, C, L). Applies a random binary mask along the 
    input tensors channel dimension to zero values of entire channels based on 
    the given probability (`p`). This is a common regularization technique to 
    prevent model overfitting.
    
    Args:
        input    : The tensor to apply the dropout to.
        p        : The probability (0-1) of dropout being applied to any given
                   channel of the input tensor.
        training : If True, this function will act as a no-op, passing its
                   input through unchanged. You generally only want to enable
                   dropout during training, and disable it during inference and
                   testing of models.
        inplace  : If True, the dropout function will modify the input tensor
                   in-place. If False, it will create a new tensor to serve as
                   output. 
                   
                   If the input tensor has `requires_grad`=True, a new
                   tensor will be created regardless of the value of `inplace`,
                   since in-place modifications to tensor data break the 
                   computation graph.
            
    Returns:
        Tensor : The resulting tensor from the dropout operation.
    '''
    assert 0.0 <= p <= 1.0, 'Probability must be between 0.0 and 1.0.'
    assert input.ndim == 3, 'dropout1d expects 3D input [B, C, L]'
    
    if not training or p == 0.0: return input
    if p == 1.0:                 return input * 0.0
    
    mask_shape = (input.shape[0], input.shape[1], 1)
    return _dropout(input, mask_shape, p, inplace)

def dropout2d(
    input:    Tensor,
    p:        float = 0.5,
    training: bool  = True,
    inplace:  bool  = False
) -> Tensor:
    '''Randomly disables a channels by a given probability.

    Expects tensor of shape (B, C, H, W). Applies a random binary mask along 
    the input tensors channel dimension to zero values of entire channels based 
    on the given probability (`p`). This is a common regularization technique 
    to prevent model overfitting.
    
    Args:
        input    : The tensor to apply the dropout to.
        p        : The probability (0-1) of dropout being applied to any given
                   channel of the input tensor.
        training : If True, this function will act as a no-op, passing its
                   input through unchanged. You generally only want to enable
                   dropout during training, and disable it during inference and
                   testing of models.
        inplace  : If True, the dropout function will modify the input tensor
                   in-place. If False, it will create a new tensor to serve as
                   output. 
                   
                   If the input tensor has `requires_grad`=True, a new
                   tensor will be created regardless of the value of `inplace`,
                   since in-place modifications to tensor data break the 
                   computation graph.
            
    Returns:
        Tensor : The resulting tensor from the dropout operation.
    '''
    assert 0.0 <= p <= 1.0, 'Probability must be between 0.0 and 1.0.'
    assert input.ndim == 4, 'dropout2d expects 4D input [B, C, H, W]'
    
    if not training or p == 0.0: return input
    if p == 1.0:                 return input * 0.0
    
    mask_shape = (input.shape[0], input.shape[1], 1, 1)
    return _dropout(input, mask_shape, p, inplace)

def dropout3d(
    input:    Tensor,
    p:        float = 0.5,
    training: bool  = True,
    inplace:  bool  = False
) -> Tensor:
    '''Randomly disables a channels by a given probability.

    Expects tensor of shape (B, C, D, H, W). Applies a random binary mask along 
    the input tensors channel dimension to zero values of entire channels based 
    on the given probability (`p`). This is a common regularization technique 
    to prevent model overfitting.
    
    Args:
        input    : The tensor to apply the dropout to.
        p        : The probability (0-1) of dropout being applied to any given
                   channel of the input tensor.
        training : If True, this function will act as a no-op, passing its
                   input through unchanged. You generally only want to enable
                   dropout during training, and disable it during inference and
                   testing of models.
        inplace  : If True, the dropout function will modify the input tensor
                   in-place. If False, it will create a new tensor to serve as
                   output. 
                   
                   If the input tensor has `requires_grad`=True, a new
                   tensor will be created regardless of the value of `inplace`,
                   since in-place modifications to tensor data break the 
                   computation graph.
            
    Returns:
        Tensor : The resulting tensor from the dropout operation.
    '''
    assert 0.0 <= p <= 1.0, 'Probability must be between 0.0 and 1.0.'
    assert input.ndim == 5, 'dropout3d expects 4D input [B, C, D, H, W]'
    
    if not training or p == 0.0: return input
    if p == 1.0:                 return input * 0.0

    mask_shape = (input.shape[0], input.shape[1], 1, 1, 1)
    return _dropout(input, mask_shape, p, inplace)

### ALPHA DROPOUT ###

def _alpha_dropout(
    input:      Tensor,
    mask_shape: ShapeType,
    p:          float = 0.5,
    inplace:    bool  = False
) -> Tensor:
    '''Abstract alpha dropout function with configurable mask shape.'''
    alpha_prime = -1.7580993408473766
    keep_prob   = 1 - p

    a = (keep_prob + alpha_prime**2 * keep_prob * (1-keep_prob))**(-0.5)
    b = -a * (alpha_prime * (1-keep_prob))

    mask    = (creation.rand(mask_shape, device=input.device) > p)
    mask    = mask.to(dtype=input.dtype).detach().broadcast_to(input.shape)
    dropped = mask * input + (1.0 - mask) * alpha_prime
    out     = a * dropped + b
    
    if inplace and not input.requires_grad: return input.copy_(out)
    return out

def alpha_dropout(
    input:    Tensor,
    p:        float = 0.5,
    training: bool  = True,
    inplace:  bool  = False
) -> Tensor:
    '''Randomly scales a portion of neurons to the negative saturation of SeLU.

    Unlike standard dropout, which zeros values of the dropped activations, 
    Alpha Dropout instead scales the dropped activations by the negative 
    saturation value of the SeLU activation function. This allows it to 
    maintain the zero mean and unit variance of the input tensor.
    
    Args:
        input    : The tensor to apply the dropout to.
        p        : The probability (0-1) of dropout being applied to any given
                   activation.
        training : If True, this function will act as a no-op, passing its
                   input through unchanged. You generally only want to enable
                   dropout during training, and disable it during inference and
                   testing of models.
        inplace  : If True, the dropout function will modify the input tensor
                   in-place. If False, it will create a new tensor to serve as
                   output. 
                   
                   If the input tensor has `requires_grad`=True, a new
                   tensor will be created regardless of the value of `inplace`,
                   since in-place modifications to tensor data break the 
                   computation graph.
            
    Returns:
        Tensor : The resulting tensor from the dropout operation.
    '''
    assert 0.0 <= p <= 1.0, 'Probability must be between 0.0 and 1.0.'
    
    if not training or p == 0.0: return input
    if p == 1.0:                 return input * 0.0
    
    return _alpha_dropout(input, input.shape, p, inplace)

def feature_alpha_dropout(
    input:    Tensor,
    p:        float = 0.5,
    training: bool  = True,
    inplace:  bool  = False
) -> Tensor:
    '''Randomly scales input channels to the negative saturation of SeLU.

    Unlike standard dropout, which zeros values of the dropped activations, 
    Feature Alpha Dropout instead scales the dropped activations by the 
    negative saturation value of the SeLU activation function. This allows it 
    to maintain the zero mean and unit variance of the input tensor.
    
    Args:
        input    : The tensor to apply the dropout to.
        p        : The probability (0-1) of dropout being applied to any given
                   channel of the input tensor.
        training : If True, this function will act as a no-op, passing its
                   input through unchanged. You generally only want to enable
                   dropout during training, and disable it during inference and
                   testing of models.
        inplace  : If True, the dropout function will modify the input tensor
                   in-place. If False, it will create a new tensor to serve as
                   output. 
                   
                   If the input tensor has `requires_grad`=True, a new
                   tensor will be created regardless of the value of `inplace`,
                   since in-place modifications to tensor data break the 
                   computation graph.
            
    Returns:
        Tensor : The resulting tensor from the dropout operation.
    '''
    assert 0.0 <= p <= 1.0, 'Probability must be between 0.0 and 1.0.'
    assert input.ndim >= 3, 'feature_alpha_dropout expects at least 3D input'
    
    if not training or p == 0.0: return input
    if p == 1.0:                 return input * 0.0

    mask_shape = (input.shape[0], input.shape[1]) + (1,) * len(input.shape[2:])
    return _alpha_dropout(input, mask_shape, p, inplace)

    
