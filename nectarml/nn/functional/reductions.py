import math
from typing import Literal

import builtins

from nectarml      import typing
from nectarml.core import Tensor
from nectarml.nn.functional.combination import cat, stack

def min(
    input:   Tensor,
    dim:     builtins.int | None = None,
    keepdim: builtins.bool = False
) -> Tensor | typing.return_types.min:
    '''Reduces a tensor to its minimum value.

    Args:
        input   : The tensor to reduce.
        dim     : Optional dimension to perform the reduction along. If not
                  provided, the reduction will return a scalar tensor 
                  containing the smallest value found in the input tensor.
        keepdim : If True, the `dim` that the tensor was reduced along will be
                  kept as a size 1 dimension. For example:
                  ```
                  t = nectarml.rand(shape=(1, 3, 256, 256))
                  no_keep = F.min(t, dim=1, keepdim=False)
                  ```
                  **no_keep.values.shape == (1, 256, 256)**
                  
                  ```
                  t = nectarml.rand(shape=(1, 3, 256, 256))
                  keep = F.min(t, dim=1, keepdim=True)
                  ```
                  **keep.values.shape == (1, 1, 256, 256)**
    
    Returns:
        Tensor | min : If `dim` is None, this function will return a scalar 
                       tensor containing the minimum value found in the input 
                       tensor. 

                       If dim is **not** None, it will instead return a
                       `nectarml.typing.return_types.min`, which contains the
                       resulting tensor from the reduction, and an int32 tensor
                       containing the original indices of the minimum values in
                       the input tensor. These can be accessed like so:
                       ```
                       minimum = tensor.min(dim=1)
                       result  = minimum.values
                       indices = minimum.indices
                       ```
    '''
    return input.min(dim, keepdim)

def amin(
    input:   Tensor,
    dim:     builtins.int | tuple[builtins.int, ...] | None = None,
    keepdim: builtins.bool = False
) -> Tensor:
    return input.amin(dim, keepdim)

def max(
    input:   Tensor, 
    dim:     builtins.int | None = None,
    keepdim: builtins.bool = False
) -> Tensor | typing.return_types.max:
    '''Reduces a tensor to its maximum value.

    Args:
        input   : The tensor to reduce.
        dim     : Optional dimension to perform the reduction along. If not
                  provided, the reduction will return a scalar tensor 
                  containing the largest value found in the input tensor.
        keepdim : If True, the `dim` that the tensor was reduced along will be
                  kept as a size 1 dimension. For example:
                  ```
                  t = nectarml.rand(shape=(1, 3, 256, 256))
                  no_keep = F.max(t, dim=1, keepdim=False)
                  ```
                  **no_keep.values.shape == (1, 256, 256)**
                  
                  ```
                  t = nectarml.rand(shape=(1, 3, 256, 256))
                  keep = F.max(t, dim=1, keepdim=True)
                  ```
                  **keep.values.shape == (1, 1, 256, 256)**
    
    Returns:
        Tensor | max : If `dim` is None, this function will return a scalar 
                       tensor containing the maximum value found in the input 
                       tensor. 

                       If dim is **not** None, it will instead return a
                       `nectarml.typing.return_types.max`, which contains the
                       resulting tensor from the reduction, and an int32 tensor
                       containing the original indices of the maximum values in
                       the input tensor. These can be accessed like so:
                       ```
                       maximum = tensor.max(dim=1)
                       result  = maximum.values
                       indices = maximum.indices
                       ```
    '''
    return input.max(dim, keepdim)

def amax(
    input:   Tensor,
    dim:     builtins.int | tuple[builtins.int, ...] | None = None,
    keepdim: builtins.bool = False
) -> Tensor:
    return input.amax(dim, keepdim)

def argmin(
    input:   Tensor,
    dim:     builtins.int | None = None,
    keepdim: builtins.bool = False
) -> Tensor:
    return input.argmin(dim, keepdim)
    
def argmax(
    input:  Tensor,
    dim:     builtins.int | None = None,
    keepdim: builtins.bool = False
) -> Tensor:
    return input.argmax(dim, keepdim)

def mean(
    input:   Tensor, 
    dim:     builtins.int | tuple[builtins.int, ...] | None = None,
    keepdim: builtins.bool = False
) -> Tensor:
    '''Reduces a tensor to its mean value.

    Args:
        input   : The tensor to reduce.
        dim     : Optional dimension to perform the reduction along. If not
                  provided, the reduction will return a scalar tensor 
                  containing the mean value found in the entire input tensor.
        keepdim : If True, the `dim` that the tensor was reduced along will be
                  kept as a size 1 dimension. For example:
                  ```
                  t = nectarml.rand(shape=(1, 3, 256, 256))
                  no_keep = F.mean(t, dim=1, keepdim=False)
                  ```
                  **no_keep.shape == (1, 256, 256)**
                  
                  ```
                  t = nectarml.rand(shape=(1, 3, 256, 256))
                  keep = F.mean(t, dim=1, keepdim=True)
                  ```
                  **keep.shape == (1, 1, 256, 256)**
    
    Returns:
        Tensor : The resulting tensor from the reduction.
    '''
    return input.mean(dim, keepdim)

def sum(
    input:   Tensor, 
    dim:     builtins.int | tuple[builtins.int, ...] | None = None,
    keepdim: builtins.bool = False,
    initial: builtins.int | builtins.float = 0
) -> Tensor:
    '''Reduces a tensor by summing its elements.

    Args:
        input   : The tensor to reduce.
        dim     : Optional dimension to perform the reduction along. If not
                  provided, the reduction will return a scalar tensor 
                  containing the sum of all elements in the input tensor.
        keepdim : If True, the `dim` that the tensor was reduced along will be
                  kept as a size 1 dimension. For example:
                  ```
                  t = nectarml.rand(shape=(1, 3, 256, 256))
                  no_keep = F.sum(t, dim=1, keepdim=False)
                  ```
                  **no_keep.shape == (1, 256, 256)**
                  
                  ```
                  t = nectarml.rand(shape=(1, 3, 256, 256))
                  keep = F.sum(t, dim=1, keepdim=True)
                  ```
                  **keep.shape == (1, 1, 256, 256)**
    
    Returns:
        Tensor : The resulting tensor from the reduction.
    '''
    return input.sum(dim, keepdim, initial)

def prod(
    input:   Tensor, 
    dim:     builtins.int | tuple[builtins.int, ...] | None = None,
    keepdim: builtins.bool = False,
    initial: builtins.int | builtins.float = 1
) -> Tensor:
    '''Reduces a tensor to the product of its elements.

    Args:
        input   : The tensor to reduce.
        dim     : Optional dimension to perform the reduction along. If not
                  provided, the reduction will return a scalar tensor 
                  containing the product of all elements in the input tensor.
        keepdim : If True, the `dim` that the tensor was reduced along will be
                  kept as a size 1 dimension. For example:
                  ```
                  t = nectarml.rand(shape=(1, 3, 256, 256))
                  no_keep = F.prod(t, dim=1, keepdim=False)
                  ```
                  **no_keep.shape == (1, 256, 256)**
                  
                  ```
                  t = nectarml.rand(shape=(1, 3, 256, 256))
                  keep = F.prod(t, dim=1, keepdim=True)
                  ```
                  **keep.shape == (1, 1, 256, 256)**
    
    Returns:
        Tensor : The resulting tensor from the reduction.
    '''
    return input.prod(dim, keepdim, initial)

def norm(
    input:   Tensor,
    p:       Literal['fro', 'l1', 'inf', '-inf', 'l0', 'lp'] = 'fro',
    dim:     builtins.int | tuple[builtins.int, ...] | None = None,
    keepdim: builtins.bool = False
) -> Tensor:
    '''Reduces a tensor via a normalization function.

    Args:
        input   : The tensor to reduce.
        p       : The normalization function to apply. Options are:
                  - `fro`: L2/Frobenius norm
                  - `l1`: L1 norm
                  - `inf`: L minus infinity norm
                  - `-inf`: L infinity norm
                  - `l0`: L0 norm
                  - `lp`: General Lp norm
        dim     : Optional dimension to perform the reduction along. If not
                  provided, the reduction will return a scalar tensor 
                  containing the product of all elements in the input tensor.
        keepdim : If True, the `dim` that the tensor was reduced along will be
                  kept as a size 1 dimension. For example:
                  ```
                  t = nectarml.rand(shape=(1, 3, 256, 256))
                  no_keep = F.norm(t, dim=1, keepdim=False)
                  ```
                  **no_keep.shape == (1, 256, 256)**
                  
                  ```
                  t = nectarml.rand(shape=(1, 3, 256, 256))
                  keep = F.norm(t, dim=1, keepdim=True)
                  ```
                  **keep.shape == (1, 1, 256, 256)**
    
    Returns:
        Tensor : The resulting tensor from the reduction.
    '''
    return input.norm(p, dim, keepdim)

def quantile(
    input:         Tensor, 
    q:             builtins.float | Tensor,
    dim:           builtins.int | None = None,
    keepdim:       builtins.bool = False,
    interpolation: Literal[
        'linear', 'lower', 'higher', 'nearest', 'midpoint'
    ] = 'linear'
) -> Tensor:
    if isinstance(q, Tensor):
        assert q.ndim <= 1, 'q must be a builtins.float or 1D Tensor.'
        q_vals = q.numpy().tolist()
        scalar_q = False
    else:
        q_vals = [q]
        scalar_q = True
        
    match interpolation:
        case 'lower': 
            def get_idx(idx_float: builtins.float) -> builtins.int:
                return builtins.int(math.floor(idx_float))
        case 'higher': 
            def get_idx(idx_float: builtins.float) -> builtins.int:
                return builtins.int(math.ceil(idx_float))
        case 'nearest': 
            def get_idx(idx_float: builtins.float) -> builtins.int:
                return builtins.int(round(idx_float))
        case 'midpoint' | 'linear': 
            def get_idx(
                idx_float: builtins.float
            ) -> tuple[builtins.int, builtins.int]:
                return (
                    builtins.int(math.floor(idx_float)), 
                    builtins.int(math.ceil(idx_float)))
        case _: raise ValueError(
            f'Interpolation mode not valid: {interpolation}')

    if dim is None:
        data = input.flatten()
        sorted_vals, _ = data.sort()
        n = len(sorted_vals) - 1

        results = []
        for qv in q_vals:
            assert 0.0 <= qv <= 1.0, f'q must be in [0, 1] but got {qv}'
            idx_float = qv * n

            if interpolation == 'lower':
                results.append(sorted_vals[get_idx(idx_float)])
            elif interpolation == 'higher':
                results.append(sorted_vals[get_idx(idx_float)])
            elif interpolation == 'nearest':
                results.append(sorted_vals[get_idx(idx_float)])
            elif interpolation == 'midpoint':
                lo, hi = get_idx(idx_float)
                results.append((sorted_vals[lo] + sorted_vals[hi]) / 2)
            else:
                lo, hi = get_idx(idx_float)
                frac = idx_float - lo
                if lo != hi: 
                    results.append(
                        sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)
                else: results.append(sorted_vals[lo])
                    
        if scalar_q: return results[0]
        return cat(results, dim=0)

    else:
        sorted_vals, _ = input.sort(dim=dim)
        n = input.shape[dim] - 1

        results = []
        for qv in q_vals:
            idx_float = qv * n
            if interpolation == 'lower':
                results.append(sorted_vals.select(dim, get_idx(idx_float)))
            elif interpolation == 'higher':
                results.append(sorted_vals.select(dim, get_idx(idx_float)))
            elif interpolation == 'nearest':
                results.append(sorted_vals.select(dim, get_idx(idx_float)))
            elif interpolation == 'midpoint':
                lo, hi = get_idx(idx_float)
                results.append(
                    (sorted_vals.select(dim, lo) + 
                     sorted_vals.select(dim, hi)) / 2)
            else:
                lo, hi = get_idx(idx_float)
                frac = idx_float - lo
                if lo != hi: 
                    results.append(
                        sorted_vals.select(dim, lo) * (1 - frac) +
                        sorted_vals.select(dim, hi) * frac)
                else: results.append(sorted_vals.select(dim, lo))
                    
        if scalar_q:
            out = results[0]
            if keepdim: out = out.unsqueeze(dim)
            return out

        out = stack(results, dim=0)
        if keepdim: out = out.unsqueeze(dim + 1)
        return out

def cumsum(input: Tensor, dim: builtins.int) -> Tensor:
    return input.cumsum(dim)
