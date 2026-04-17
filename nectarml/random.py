from __future__ import annotations

import builtins
from typing import Any, Literal, Self
from collections.abc import Iterable, Callable

import numpy as np
from numpy.typing import NDArray

from nectarml import typing

### RANDOM WRAPPER ###

class Random:
    def __init__(self: Random, seed: builtins.int | None = None) -> None:
        self._seed = seed
        self._rng  = np.random.default_rng(seed=self.seed)
        
    ### PROPERTIES ###
    
    @property
    def seed(self: Random) -> builtins.int | None:
        return self._seed
    
    @seed.setter
    def seed(self: Random, a: builtins.int) -> None:
        self._seed = a
        self._rng  = np.random.default_rng(seed=a)

    ### METHODS ###

    def random(
        self:  Random, 
        size:  tuple[builtins.int, ...] | typing.Size | None = None, 
        dtype: typing.dtype = typing.float32,
    ) -> float | NDArray[np.float64]:
        return self._rng.random(size=size, dtype=dtype.cpu)
    
    def uniform(
        self: Random,
        low:  float | np.typing.ArrayLike,
        high: float | np.typing.ArrayLike,
        size: tuple[builtins.int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.uniform(low, high, size)
    
    def normal(
        self:  Random,
        loc:   float | np.typing.ArrayLike,
        scale: float | np.typing.ArrayLike,
        size:  tuple[builtins.int, ...] | typing.Size | None = None,
    ) -> float | NDArray[np.float64]:
        return self._rng.normal(loc, scale, size)
    
    def beta(
        self: Random,
        a:    float | np.typing.ArrayLike,
        b:    float | np.typing.ArrayLike,
        size: tuple[builtins.int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.beta(a, b, size)
    
    def binomial(
        self: Random,
        n:    builtins.int | np.typing.ArrayLike,
        p:    float | np.typing.ArrayLike,
        size: tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.int | NDArray[np.int64]:
        return self._rng.binomial(n, p, size)
    
    def negative_binomial(
        self: Random,
        n:    float | np.typing.ArrayLike,
        p:    float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> builtins.int | NDArray[np.int64]:
        return self._rng.negative_binomial(n, p, size)
    
    def bytes(self: Random, length: int) -> builtins.bytes:
        return self._rng.bytes(length)
    
    def chisquare(
        self: Random,
        df:   builtins.float | np.typing.ArrayLike,
        size: tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.chisquare(df, size)
    
    def noncentral_chisquare(
        self: Random,
        df:   builtins.float | np.typing.ArrayLike,
        nonc: builtins.float | np.typing.ArrayLike,
        size: tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.noncentral_chisquare(df, nonc, size)
    
    def dirichlet(
        self:  Random,
        alpha: builtins.float | np.typing.ArrayLike,
        size:  tuple[builtins.int, ...] | typing.Size | None = None
    ) -> NDArray[np.float64]:
        return self._rng.dirichlet(alpha, size)
    
    def exponential(
        self:  Random,
        scale: builtins.float | np.typing.ArrayLike,
        size:  tuple[builtins.int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.exponential(scale, size)
    
    def f(
        self:  Random,
        dfnum: builtins.float | np.typing.ArrayLike,
        dfden: builtins.float | np.typing.ArrayLike,
        size:  tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.f(dfnum, dfden, size)
    
    def noncentral_f(
        self:  Random,
        dfnum: builtins.float | np.typing.ArrayLike,
        dfden: builtins.float | np.typing.ArrayLike,
        nonc:  builtins.float | np.typing.ArrayLike,
        size:  tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.noncentral_f(dfnum, dfden, nonc, size)
    
    def gamma(
        self:  Random,
        shape: builtins.float | np.typing.ArrayLike,
        scale: builtins.float | np.typing.ArrayLike,
        size:  tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.gamma(shape, scale, size)
    
    def geometric(
        self: Random,
        p:    builtins.float | np.typing.ArrayLike,
        size: tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.int | NDArray[np.int64]:
        return self._rng.geometric(p, size)
    
    def gumbel(
        self:  Random,
        loc:   float | np.typing.ArrayLike,
        scale: float | np.typing.ArrayLike,
        size:  tuple[builtins.int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.gumbel(loc, scale, size)
    
    def hypergeometric(
        self:    Random,
        ngood:   builtins.int,
        nbad:    builtins.int,
        nsample: builtins.int,
        size:    tuple[int, ...] | typing.Size | None = None
    ) -> NDArray[np.int64]:
        return self._rng.hypergeometric(ngood, nbad, nsample, size)
    
    def integers(
        self:  Random,
        low:   builtins.int,
        high:  builtins.int,
        size:  tuple[builtins.int, ...] | typing.Size | None = None,
        dtype: typing.dtype = typing.int32,
    ) -> NDArray[Any]:
        return self._rng.integers(low, high, size).astype(dtype.cpu)
    
    def laplace(
        self:  Random,
        loc:   builtins.float | np.typing.ArrayLike,
        scale: builtins.float | np.typing.ArrayLike,
        size:  tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.laplace(loc, scale, size)
    
    def logistic(
        self:  Random,
        loc:   builtins.float | np.typing.ArrayLike,
        scale: builtins.float | np.typing.ArrayLike,
        size:  tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.logistic(loc, scale, size)
    
    def lognormal(
        self:  Random,
        mean:  builtins.float | np.typing.ArrayLike,
        sigma: builtins.float | np.typing.ArrayLike,
        size:  tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.lognormal(mean, sigma, size)
    
    def logseries(
        self: Random,
        p:    builtins.float | np.typing.ArrayLike,
        size: tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.int | NDArray[np.int64]:
        return self._rng.logseries(p, size)
    
    def multinomial(
        self:  Random,
        n:     builtins.int | np.typing.ArrayLike,
        pvals: builtins.float | np.typing.ArrayLike,
        size:  tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.int | NDArray[np.int64]:
        return self._rng.multinomial(n, pvals, size)

    def multivariate_hypergeometric(
        self:    Random,
        colors:  builtins.int | np.typing.ArrayLike,
        nsample: builtins.int,
        size:    tuple[builtins.int, ...] | typing.Size | None = None,
        method:  Literal['marginals', 'count'] = 'marginals'
    ) -> NDArray[np.int64]:
        return self._rng.multivariate_hypergeometric(
            colors, nsample, size, method)
        
    def multivariate_normal(
        self:        Random,
        mean:        builtins.float | np.typing.ArrayLike,
        cov:         builtins.float | np.typing.ArrayLike,
        size:        tuple[builtins.int, ...] | typing.Size | None = None,
        check_valid: Literal['warn', 'raise', 'ignore'] = ...,
        tol:         builtins.float = ...,
        method:      Literal['svd', 'eigh', 'cholesky'] = 'svd'
    ) -> NDArray[np.float64]:
        return self._rng.multivariate_normal(
            mean, cov, size, check_valid, tol, method=method)
        
    def negative_binomial(
        self: Random,
        n:    builtins.float | np.typing.ArrayLike,
        p:    builtins.float | np.typing.ArrayLike,
        size: tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.int | NDArray[np.int64]:
        return self._rng.negative_binomial(n, p, size)
    
    def pareto(
        self: Random,
        a:    builtins.float | np.typing.ArrayLike,
        size: tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.pareto(a, size)
    
    def permutation(
        self: Random,
        x:    builtins.float | np.typing.ArrayLike,
        axis: builtins.int = ...,
        size: tuple[builtins.int, ...] | typing.Size | None = None
    ) -> NDArray[np.int64]:
        return self._rng.permutation(x, axis, size)
    
    def permuted(
        self: Random,
        x:    builtins.float | np.typing.ArrayLike,
        axis: builtins.int = ...,
    ) -> NDArray[Any]:
        return self._rng.permuted(x, axis=axis)
    
    def poisson(
        self: Random,
        lam:  builtins.float | np.typing.ArrayLike,
        size: tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.int | NDArray[np.int64]:
        return self._rng.poisson(lam, size)
    
    def power(
        self: Random,
        a:    builtins.float | np.typing.ArrayLike,
        size: tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.power(a, size)
    
    def rayleigh(
        self:  Random,
        scale: builtins.float | np.typing.ArrayLike,
        size:  tuple[builtins.int, ...] | typing.Size | None = None
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.rayleigh(scale, size)
    
    def standard_t(
        self: Random,
        df:   builtins.float | np.typing.ArrayLike,
        size: tuple[builtins.int, ...] | typing.Size | None = None,
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.standard_t(df, size)
    
    def standard_cauchy(
        self: Random,
        size: tuple[builtins.int, ...] | typing.Size | None = None,
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.standard_cauchy(size)
    
    def standard_exponential(
        self:  Random,
        size:   tuple[builtins.int, ...] | typing.Size | None = None, 
        dtype:  typing.dtype = typing.float32,
        method: Literal['zig', 'inv'] = 'zig'
    ) -> float | NDArray[np.float64]:
        return self._rng.standard_exponential(size, dtype.cpu, method)
    
    def standard_gamma(
        self:  Random,
        shape: builtins.float | np.typing.ArrayLike,
        size:  tuple[builtins.int, ...] | typing.Size | None = None, 
        dtype: typing.dtype = typing.float32,
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.standard_gamma(shape, size, dtype.cpu)
    
    def standard_normal(
        self:  Random,
        size:  tuple[builtins.int, ...] | typing.Size | None = None, 
        dtype: typing.dtype = typing.float32,
    ) -> builtins.float | NDArray[np.float64]:
        return self._rng.standard_normal(size=size, dtype=dtype.cpu)
    
    def choice(
        self:    Random,
        a:       builtins.int | np.typing.ArrayLike,
        size:    tuple[builtins.int, ...] | typing.Size | None = None, 
        replace: builtins.bool = True,
        p:       np.typing.ArrayLike | None = None,
        axis:    builtins.int = 0,
        shuffle: builtins.bool = True
    ) -> Any | NDArray[Any]:
        return self._rng.choice(
            a=a, size=size, replace=replace, p=p, axis=axis, shuffle=shuffle)
     
    def choices(
        self:       Random,
        population: Iterable[Any],
        weights:    Iterable[float] | None = None,
        k:          builtins.int = 1,
        unique:     builtins.bool = True
    ) -> list[Any]:
        population = list(population)
        weights = [1.0]*len(population) if weights is None else weights
        indices = self._rng.choice(
            list(range(len(population))), size=(k,), 
            p=weights, replace=not unique)
        return [population[i] for i in indices]
        
    def shuffle(
        self: Random, 
        x:    Iterable[Any], 
        axis: builtins.int = 0
    ) -> None:
        self._rng.shuffle(x, axis=axis)
        
    def sample(
        self:       Random,
        population: Iterable[Any],
        k:          builtins.int,
        counts:     list[builtins.int] | None = None
    ) -> list[Any]:
        choices = self.choices(population, weights=None, k=k, unique=True)
        if counts is not None:
            output = []
            for idx, choice in enumerate(choices):
                count = counts[idx%len(counts)]
                output.extend([choice]*count)
        else: output = choices
        return output
    
    def randint(
        self: Random, 
        a:    builtins.int, 
        b:    builtins.int
    ) -> builtins.int:
        return self._rng.integers(low=a, high=b+1)
    
    def randfloat(
        self: Random, 
        a:    builtins.float, 
        b:    builtins.float
    ) -> builtins.float:
        return a + (b - a) * self.random() 
    
### INSTANCE && GLOBAL METHODS ###
    
RNG = Random()

def manual_seed(seed: builtins.int) -> None:
    global RNG
    RNG.seed = seed

class fork_rng:
    def __init__(
        self:    fork_rng, 
        seed:    builtins.int | None = None, 
        enabled: builtins.bool = True
    ) -> None:
        self.seed       = seed
        self._prev_seed = None
        self.enabled    = enabled
        
    def __enter__(self: fork_rng, *args: Any) -> Self:
        global RNG
        self._prev_seed = RNG.seed
        RNG = Random(seed=self.seed if self.enabled else self._prev_seed)
    
    def __exit__(self: fork_rng, *args: Any) -> None:
        global RNG
        RNG = Random(seed=self._prev_seed)
        
    def __call__(self: fork_rng, func: Callable) -> Callable:
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.__class__(self.seed, self.enabled):
                return func(*args, **kwargs)
        return wrapper

