from collections.abc import Sequence, Callable
from typing import Literal

import numpy as np

from nectarml import Tensor, DTypeLike, ArrayLike, zeros_like, ones_like, zeros

# ABSTRACTS

def _manipulate_shape(input: Tensor, new_data: np.ndarray) -> Tensor:
    old_shape = input.data.shape
    out = input._build_output_tensor(new_data, (input,))
    def _backward():
        if input.requires_grad:
            input.grad += out.grad.reshape(old_shape)
    out._backward = _backward
    return out

def _wrapper_base(
    input: Tensor,
    func: Callable[[np.ndarray], np.ndarray], 
    grad_func: Callable[[np.ndarray], np.ndarray], 
    children: tuple[Tensor, ...] | None = None,
    reduce_dim: int | None = None
) -> Tensor:
    if children is None: children = (input,)
    out = input._build_output_tensor(func(input.data), children)
    def _backward():
        if input.requires_grad:
            out_grad = out.grad
            if reduce_dim is not None and out_grad.ndim < input.data.ndim:
                out_grad = np.expand_dims(out_grad, axis=reduce_dim)
            out_grad = np.broadcast_to(out_grad, input.data.shape)
            input.grad += grad_func(input.data) * out_grad
    out._backward = _backward
    return out

# SHAPE MANIPULATION

def reshape(input: Tensor, shape: Sequence[int]) -> Tensor:
    return _manipulate_shape(input, input.data.reshape(shape))

def flatten(input: Tensor) -> Tensor:
    return _manipulate_shape(input, input.data.flatten())

def squeeze(input: Tensor, dim: int | tuple[int, ...] | None) -> Tensor: 
    return _manipulate_shape(input, input.data.squeeze(axis=dim))
    
def unsqueeze(input: Tensor, dim: int | tuple[int, ...]) -> Tensor:
    return _manipulate_shape(input, np.expand_dims(input.data, axis=dim))

def transpose(input: Tensor, axes: Sequence[int] | None) -> Tensor:
    out = input._build_output_tensor(input.data.transpose(axes), (input,))
    def _backward():
        if input.requires_grad:
            input.grad += out.grad.transpose(axes)
    out._backward = _backward
    return out

def swapaxes(input: Tensor, axis1: int, axis2: int) -> Tensor: 
    out = input._build_output_tensor(
        input.data.swapaxes(axis1, axis2), (input,))
    def _backward():
        if input.requires_grad:
            input.grad += out.grad.swapaxes(axis1, axis2)
    out._backward = _backward
    return out

def permute(input: Tensor, axes: Sequence[int] | None) -> Tensor:
    out = input._build_output_tensor(
        np.permute_dims(input.data, axes=axes), (input,))
    def _backward():
        if input.requires_grad:
            inverse_axes = np.argsort(axes)
            input.grad += np.permute_dims(out.grad, axes=inverse_axes)
    out._backward = _backward
    return out
    
def expand(input: Tensor, shape: tuple[int, ...]) -> Tensor:
    out = input._build_output_tensor(
        np.broadcast_to(input.data, shape), (input,))
    def _backward():
        if input.requires_grad:
            grad = out.grad
            ndims_added = grad.ndim - input.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, (in_size, out_size) in enumerate(
                zip(input.data.shape, grad.shape)):
                if in_size == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            input.grad += grad.reshape(input.data.shape)
    out._backward = _backward
    return out

def broadcast_to(input: Tensor, shape: tuple[int, ...]) -> Tensor:
    return expand(input, shape)

# REDUCTIONS

def min(
    input: Tensor, 
    dim: int | None = None, 
    keepdims: bool = False
) -> Tensor:
    def _grad(x: np.ndarray):
        max_vals = x.min(axis=dim, keepdims=True)
        return (x == max_vals).astype(x.dtype)
    return _wrapper_base(
        input, lambda x: x.min(axis=dim, keepdims=keepdims), 
        _grad, reduce_dim=dim if not keepdims else None)

def max(
    input: Tensor,
    dim: int | None = None, 
    keepdims: bool = False
) -> Tensor:
    def _grad(x: np.ndarray):
        max_vals = x.max(axis=dim, keepdims=True)
        return (x == max_vals).astype(x.dtype)
    return _wrapper_base(
        input, lambda x: x.max(axis=dim, keepdims=keepdims), 
        _grad, reduce_dim=dim if not keepdims else None)

def argmin(
    input: Tensor,
    dim: int | None = None, 
    keepdims: bool = False
) -> ArrayLike:
    return input.data.argmin(axis=dim, keepdims=keepdims)
    
def argmax(
    input: Tensor,
    dim: int | None = None, 
    keepdims: bool = False
) -> ArrayLike:
    return input.data.argmax(axis=dim, keepdims=keepdims)

def mean(
    input: Tensor,
    dim: int | None = None, 
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
) -> Tensor:
    def _grad(x: np.ndarray):
        n = x.size if dim is None else x.shape[dim]
        return (np.ones_like(x) / n)
    return _wrapper_base(
        input, lambda x: x.mean(axis=dim, dtype=dtype, keepdims=keepdims), 
        _grad, reduce_dim=dim if not keepdims else None)
    
def sum(
    input: Tensor,
    dim: int | None = None, 
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    initial: int | float = 0
) -> Tensor:
    return _wrapper_base(
        input, lambda x: x.sum(
            axis=dim, dtype=dtype, keepdims=keepdims, initial=initial),
        np.ones_like, reduce_dim=dim if not keepdims else None)

def prod(
    input: Tensor,
    dim: int | None = None, 
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    initial: int | float = 1
) -> Tensor:
    out = input._build_output_tensor(
        input.data.prod(
            axis=dim, dtype=dtype, keepdims=keepdims, initial=initial),
        (input,))
    def _backward():
        if input.requires_grad:
            out_grad = out.grad
            if dim is not None and not keepdims:
                out_grad = np.expand_dims(out_grad, axis=dim)
            out_data = out.data
            if not keepdims:
                out_data = np.expand_dims(out_data, axis=dim) \
                    if dim is not None else out_data
            input.grad += np.broadcast_to(
                out_data / input.data * out_grad, input.data.shape)
    out._backward = _backward
    return out

# MATH OPS

def abs(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.abs, np.sign)
    
def exp(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.exp, np.exp)

def log(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.log, lambda x: 1 / x)

def sqrt(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.sqrt, lambda x: 1 / (2 * np.sqrt(x)))

def sin(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.sin, np.cos)

def cos(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.cos, lambda x: -np.sin(x))

def cosh(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.cosh, np.sinh)

def tanh(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.tanh, lambda x: 1 - np.tanh(x) ** 2)

def sigmoid(input: Tensor) -> Tensor:
    return (exp(-input) + 1) ** -1

# TENSOR COMBINATION

def concatenate(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
    out = inputs[0]._build_output_tensor(
        np.concatenate([t.data for t in inputs], axis=dim), tuple(inputs))
    def _backward():
        sizes = [t.data.shape[dim] for t in inputs]
        split_points = np.cumsum(sizes[:-1])
        grads = np.split(out.grad, split_points, axis=dim)
        for tensor, grad in zip(inputs, grads):
            if tensor.requires_grad:
                tensor.grad += grad
    out._backward = _backward
    return out

def cat(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
    return concatenate(inputs, dim)

def stack(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
    out = inputs[0]._build_output_tensor(
        np.stack([t.data for t in inputs], axis=dim), tuple(inputs))
    def _backward():
        grads = np.split(out.grad, len(inputs), axis=dim)
        for tensor, grad in zip(inputs, grads):
            if tensor.requires_grad:
                tensor.grad += np.squeeze(grad, axis=dim)
    out._backward = _backward
    return out

def unstack(input: Tensor, dim: int = 0) -> list[Tensor]:
    _split = np.split(input.data, input.data.shape[dim], axis=dim)
    splits = [np.squeeze(s, axis=dim) for s in _split]
    outputs = [input._build_output_tensor(s, (input,)) for s in splits]
    _backward_called = False
    def _backward():
        nonlocal _backward_called
        if _backward_called: return
        _backward_called = True
        if input.requires_grad:
            input.grad += np.stack([o.grad for o in outputs], axis=dim)
    for out in outputs:
        out._backward = _backward
    return outputs

def unbind(input: Tensor, dim: int = 0) -> list[Tensor]:
    return unstack(input, dim)

def split(
    input: Tensor, 
    sizes: int | Sequence[int], 
    dim: int = 0
) -> list[Tensor]:
    splits = np.split(input.data, sizes, axis=dim)
    outputs = [input._build_output_tensor(s, (input,)) for s in splits]
    _backward_called = False
    def _backward():
        nonlocal _backward_called
        if _backward_called: return
        _backward_called = True
        if input.requires_grad:
            input.grad += np.concatenate([o.grad for o in outputs], axis=dim)
    for out in outputs:
        out._backward = _backward
    return outputs

def chunk(input: Tensor, size: int, dim: int = 0) -> list[Tensor]:
    assert size >= 1
    chunk_size = int(np.ceil(input.shape[dim] / size))
    return split(input, chunk_size, dim)

# INDEXING / SELECTION

def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    out = input._build_output_tensor(
        np.take_along_axis(input.data, index.data.astype(int), axis=dim),
        (input,))
    def _backward():
        if input.requires_grad:
            grad = np.zeros_like(input.data)
            np.add.at(grad, 
                tuple(np.arange(s) if i != dim else index.data.astype(int) 
                for i, s in enumerate(input.data.shape)), out.grad)
            input.grad += grad
    out._backward = _backward
    return out

def scatter(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    out_data = input.data.copy()
    np.put_along_axis(out_data, index.data.astype(int), src.data, axis=dim)
    out = input._build_output_tensor(out_data, (input, src))
    def _backward():
        if src.requires_grad:
            src.grad += np.take_along_axis(
                out.grad, index.data.astype(int), axis=dim)
        if input.requires_grad:
            grad = out.grad.copy()
            np.put_along_axis(grad, index.data.astype(int), 0, axis=dim)
            input.grad += grad
    out._backward = _backward
    return out

def where(condition: np.ndarray, x: Tensor, y: Tensor) -> Tensor:
    # NOTE: Needs Boolean Tensor support!!
    out = x._build_output_tensor(np.where(condition, x.data, y.data), (x, y))
    def _backward():
        if x.requires_grad:
            x.grad += np.where(condition, out.grad, 0)
        if y.requires_grad:
            y.grad += np.where(condition, 0, out.grad)
    out._backward = _backward
    return out

def masked_fill(input: Tensor, mask: np.ndarray, value: float) -> Tensor:
    # NOTE: Needs Boolean Tensor support!!
    out = input._build_output_tensor(
        np.where(mask, value, input.data), (input,))
    def _backward():
        if input.requires_grad:
            input.grad += np.where(mask, 0, out.grad)
    out._backward = _backward
    return out

def index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    out = input._build_output_tensor(
        np.take(input.data, index.data.astype(int), axis=dim), (input,))
    def _backward():
        if input.requires_grad:
            grad = np.zeros_like(input.data)
            np.add.at(grad, 
                tuple(index.data.astype(int) if i == dim else slice(None) 
                for i in range(input.data.ndim)), out.grad)
            input.grad += grad
    out._backward = _backward
    return out

# PADDING

def pad(
    input: Tensor, 
    pad: tuple, 
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    value: float = 0.0
) -> Tensor:
    match mode:
        case 'constant':  _mode = 'constant'
        case 'reflect':   _mode = 'reflect'
        case 'replicate': _mode = 'edge'
        case 'circular':  _mode = 'wrap'
        case _: raise ValueError(f'Invalid padding mode: {mode}')
    
    pairs = [(pad[i+1], pad[i]) for i in range(0, len(pad), 2)]
    pairs.reverse()

    num_unpadded = input.data.ndim - len(pairs)
    np_pad = [(0, 0)] * num_unpadded + pairs
    
    out = input._build_output_tensor(
        np.pad(
            input.data, np_pad, mode=_mode, 
            **({'constant_values': value} if _mode == 'constant' else {})),
        (input,))
    def _backward():
        if input.requires_grad:
            slices = tuple(
                slice(p[0], s + p[0]) 
                for s, p in zip(input.data.shape, np_pad))
            input.grad += out.grad[slices]
    out._backward = _backward
    return out

# LOSS

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




