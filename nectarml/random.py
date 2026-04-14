import builtins
from typing import Any, Literal, Self
from collections.abc import Iterable, Callable

import numpy as np
from numpy.typing import NDArray

from nectarml.typing import (
    DTypeLike, ArrayLike, float32, float64, int32, int64, Size)

### RANDOM WRAPPER ###

class Random:
    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng  = np.random.default_rng(seed=self.seed)
        
    ### PROPERTIES ###
    
    @property
    def seed(self) -> int | None:
        return self._seed
    
    @seed.setter
    def seed(self, a: int) -> None:
        self._seed = a
        self._rng = np.random.default_rng(seed=a)

    ### METHODS ###

    def random(
        self, 
        size:  tuple[int, ...] | Size | None = None, 
        dtype: DTypeLike = float32
    ) -> float | NDArray[float64]:
        return self._rng.random(size=size, dtype=float32).astype(dtype)
    
    def uniform(
        self,
        low:  float | ArrayLike,
        high: float | ArrayLike,
        size: tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.uniform(low, high, size)
    
    def normal(
        self,
        loc:   float | ArrayLike,
        scale: float | ArrayLike,
        size:  tuple[int, ...] | Size | None = None,
    ) -> float | NDArray[float64]:
        return self._rng.normal(loc, scale, size)
    
    def beta(
        self,
        a:    float | ArrayLike,
        b:    float | ArrayLike,
        size: tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.beta(a, b, size)
    
    def binomial(
        self,
        n:    int | ArrayLike,
        p:    float | ArrayLike,
        size: tuple[int, ...] | Size | None = None
    ) -> int | NDArray[int64]:
        return self._rng.binomial(n, p, size)
    
    def negative_binomial(
        self,
        n:    float | ArrayLike,
        p:    float | ArrayLike,
        size: tuple[int, ...] | Size | None = None
    ) -> int | NDArray[int64]:
        return self._rng.negative_binomial(n, p, size)
    
    def bytes(self, length: int) -> builtins.bytes:
        return self._rng.bytes(length)
    
    def chisquare(
        self,
        df:   float | ArrayLike,
        size: tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.chisquare(df, size)
    
    def noncentral_chisquare(
        self,
        df:   float | ArrayLike,
        nonc: float | ArrayLike,
        size: tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.noncentral_chisquare(df, nonc, size)
    
    def dirichlet(
        self,
        alpha: float | ArrayLike,
        size:  tuple[int, ...] | Size | None = None
    ) -> NDArray[float64]:
        return self._rng.dirichlet(alpha, size)
    
    def exponential(
        self,
        scale: float | ArrayLike,
        size:  tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.exponential(scale, size)
    
    def f(
        self,
        dfnum: float | ArrayLike,
        dfden: float | ArrayLike,
        size:  tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.f(dfnum, dfden, size)
    
    def noncentral_f(
        self,
        dfnum: float | ArrayLike,
        dfden: float | ArrayLike,
        nonc:  float | ArrayLike,
        size:  tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.noncentral_f(dfnum, dfden, nonc, size)
    
    def gamma(
        self,
        shape: float | ArrayLike,
        scale: float | ArrayLike,
        size:  tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.gamma(shape, scale, size)
    
    def geometric(
        self,
        p:    float | ArrayLike,
        size: tuple[int, ...] | Size | None = None
    ) -> int | NDArray[int64]:
        return self._rng.geometric(p, size)
    
    def gumbel(
        self,
        loc:   float | ArrayLike,
        scale: float | ArrayLike,
        size: tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.gumbel(loc, scale, size)
    
    def hypergeometric(
        self,
        ngood:   int,
        nbad:    int,
        nsample: int,
        size:    tuple[int, ...] | Size | None = None
    ) -> NDArray[int64]:
        return self._rng.hypergeometric(ngood, nbad, nsample, size)
    
    def integers(
        self,
        low:   int,
        high:  int,
        size:  tuple[int, ...] | Size | None = None,
        dtype: DTypeLike = int32
    ) -> NDArray[Any]:
        return self._rng.integers(low, high, size, dtype)
    
    def laplace(
        self,
        loc:   float | ArrayLike,
        scale: float | ArrayLike,
        size:  tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.laplace(loc, scale, size)
    
    def logistic(
        self,
        loc:   float | ArrayLike,
        scale: float | ArrayLike,
        size:  tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.logistic(loc, scale, size)
    
    def lognormal(
        self,
        mean:  float | ArrayLike,
        sigma: float | ArrayLike,
        size:  tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.lognormal(mean, sigma, size)
    
    def logseries(
        self,
        p:    float | ArrayLike,
        size: tuple[int, ...] | Size | None = None
    ) -> int | NDArray[int64]:
        return self._rng.logseries(p, size)
    
    def multinomial(
        self,
        n:     int | ArrayLike,
        pvals: float | ArrayLike,
        size:  tuple[int, ...] | Size | None = None
    ) -> int | NDArray[int64]:
        return self._rng.multinomial(n, pvals, size)

    def multivariate_hypergeometric(
        self,
        colors:  int | ArrayLike,
        nsample: int,
        size:    tuple[int, ...] | Size | None = None,
        method:  Literal['marginals', 'count'] = 'marginals'
    ) -> NDArray[int64]:
        return self._rng.multivariate_hypergeometric(
            colors, nsample, size, method)
        
    def multivariate_normal(
        self,
        mean:        float | ArrayLike,
        cov:         float | ArrayLike,
        size:        tuple[int, ...] | Size | None = None,
        check_valid: Literal['warn', 'raise', 'ignore'] = ...,
        tol: float = ...,
        method: Literal['svd', 'eigh', 'cholesky'] = 'svd'
    ) -> NDArray[float64]:
        return self._rng.multivariate_normal(
            mean, cov, size, check_valid, tol, method=method)
        
    def negative_binomial(
        self,
        n:    float | ArrayLike,
        p:    float | ArrayLike,
        size: tuple[int, ...] | Size | None = None
    ) -> int | NDArray[int64]:
        return self._rng.negative_binomial(n, p, size)
    
    def pareto(
        self,
        a:    float | ArrayLike,
        size: tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.pareto(a, size)
    
    def permutation(
        self,
        x:    float | ArrayLike,
        axis: int = ...,
        size: tuple[int, ...] | Size | None = None
    ) -> NDArray[int64]:
        return self._rng.permutation(x, axis, size)
    
    def permuted(
        self,
        x:    float | ArrayLike,
        axis: int = ...,
    ) -> NDArray[Any]:
        return self._rng.permuted(x, axis=axis)
    
    def poisson(
        self,
        lam:  float | ArrayLike,
        size: tuple[int, ...] | Size | None = None
    ) -> int | NDArray[int64]:
        return self._rng.poisson(lam, size)
    
    def power(
        self,
        a:    float | ArrayLike,
        size: tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.power(a, size)
    
    def rayleigh(
        self,
        scale: float | ArrayLike,
        size:  tuple[int, ...] | Size | None = None
    ) -> float | NDArray[float64]:
        return self._rng.rayleigh(scale, size)
    
    def standard_t(
        self,
        df:   float | ArrayLike,
        size: tuple[int, ...] | Size | None = None,
    ) -> float | NDArray[float64]:
        return self._rng.standard_t(df, size)
    
    def standard_cauchy(
        self,
        size: tuple[int, ...] | Size | None = None,
    ) -> float | NDArray[float64]:
        return self._rng.standard_cauchy(size)
    
    def standard_exponential(
        self,
        size:   tuple[int, ...] | Size | None = None, 
        dtype:  DTypeLike = float32,
        method: Literal['zig', 'inv'] = 'zig'
    ) -> float | NDArray[float64]:
        return self._rng.standard_exponential(size, dtype, method)
    
    def standard_gamma(
        self,
        shape: float | ArrayLike,
        size:  tuple[int, ...] | Size | None = None, 
        dtype: DTypeLike = float32
    ) -> float | NDArray[float64]:
        return self._rng.standard_gamma(shape, size, dtype)
    
    def standard_normal(
        self,
        size:  tuple[int, ...] | Size | None = None, 
        dtype: DTypeLike = float32
    ) -> float | NDArray[float64]:
        return self._rng.standard_normal(size=size, dtype=dtype)
    
    def choice(
        self,
        a:       int | ArrayLike,
        size:    tuple[int, ...] | Size | None = None, 
        replace: bool = True,
        p:       ArrayLike | None = None,
        axis:    int = 0,
        shuffle: bool = True
    ) -> Any | NDArray[Any]:
        return self._rng.choice(
            a=a, size=size, replace=replace, p=p, axis=axis, shuffle=shuffle)
     
    def choices(
        self,
        population: Iterable[Any],
        weights:    Iterable[float] | None = None,
        k:          int = 1,
        unique:     bool = True
    ) -> list[Any]:
        population = list(population)
        weights = [1.0]*len(population) if weights is None else weights
        indices = self._rng.choice(
            list(range(len(population))), size=(k,), 
            p=weights, replace=not unique)
        return [population[i] for i in indices]
        
    def shuffle(self, x: Iterable[Any], axis: int = 0) -> None:
        self._rng.shuffle(x, axis=axis)
        
    def sample(
        self,
        population: Iterable[Any],
        k:          int,
        counts:     list[int] | None = None
    ) -> list[Any]:
        choices = self.choices(population, weights=None, k=k, unique=True)
        if counts is not None:
            output = []
            for idx, choice in enumerate(choices):
                count = counts[idx%len(counts)]
                output.extend([choice]*count)
        else: output = choices
        return output
    
    def randint(self, a: int, b: int) -> int:
        return self._rng.integers(low=a, high=b+1)
    
    def randfloat(self, a: float, b: float) -> float:
        return a + (b - a) * self.random() 
    
### INSTANCE && GLOBAL METHODS ###
    
RNG = Random() # np.random.default_rng()

def manual_seed(seed: int) -> None:
    global RNG
    RNG.seed = seed

class fork_rng:
    def __init__(self, seed: int | None = None, enabled: bool = True) -> None:
        self.seed = seed
        self._prev_seed = None
        self.enabled = enabled
        
    def __enter__(self, *args) -> Self:
        global RNG
        self._prev_seed = RNG.seed
        RNG = Random(seed=self.seed if self.enabled else self._prev_seed)
    
    def __exit__(self, *args) -> None:
        global RNG
        RNG = Random(seed=self._prev_seed)
        
    def __call__(self, func: Callable) -> Callable:
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.__class__(self.seed, self.enabled):
                return func(*args, **kwargs)
        return wrapper

