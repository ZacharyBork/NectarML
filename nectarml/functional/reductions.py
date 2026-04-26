import math
from typing import Literal

import builtins

from nectarml      import return_types
from nectarml.core import Tensor
from nectarml.functional.combination import cat, stack

def min(
    input:   Tensor,
    dim:     builtins.int | None = None,
    keepdim: builtins.bool = False
) -> Tensor | return_types.min:
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
) -> Tensor | return_types.max:
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
    return input.mean(dim, keepdim)

def sum(
    input:   Tensor, 
    dim:     builtins.int | tuple[builtins.int, ...] | None = None,
    keepdim: builtins.bool = False,
    initial: builtins.int | builtins.float = 0
) -> Tensor:
    return input.sum(dim, keepdim, initial)

def prod(
    input:   Tensor, 
    dim:     builtins.int | tuple[builtins.int, ...] | None = None,
    keepdim: builtins.bool = False,
    initial: builtins.int | builtins.float = 1
) -> Tensor:
    return input.prod(dim, keepdim, initial)

def norm(
    input:   Tensor,
    p:       Literal['fro', 'l1', 'inf', '-inf', 'l0', 'lp'] = 'fro',
    dim:     builtins.int | tuple[builtins.int, ...] | None = None,
    keepdim: builtins.bool = False
) -> Tensor:
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
