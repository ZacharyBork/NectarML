import builtins
from typing import Any, Literal, Self
from collections.abc import Iterable, Callable

import numpy as np
from numpy.typing import NDArray

from nectarml import typing

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
        size:  tuple[int, ...] | typing.Size | None = None, 
        dtype: typing.dtype = typing.float32,
    ) -> float | NDArray[np.float64]:
        return self._rng.random(size=size, dtype=np.float32).astype(dtype.cpu)
    
    def uniform(
        self,
        low:  float | np.typing.ArrayLike,
        high: float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.uniform(low, high, size)
    
    def normal(
        self,
        loc:   float | np.typing.ArrayLike,
        scale: float | np.typing.ArrayLike,
        size:  tuple[int, ...] | typing.Size | None = None,
    ) -> float | NDArray[np.float64]:
        return self._rng.normal(loc, scale, size)
    
    def beta(
        self,
        a:    float | np.typing.ArrayLike,
        b:    float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.beta(a, b, size)
    
    def binomial(
        self,
        n:    int | np.typing.ArrayLike,
        p:    float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> int | NDArray[np.int64]:
        return self._rng.binomial(n, p, size)
    
    def negative_binomial(
        self,
        n:    float | np.typing.ArrayLike,
        p:    float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> int | NDArray[np.int64]:
        return self._rng.negative_binomial(n, p, size)
    
    def bytes(self, length: int) -> builtins.bytes:
        return self._rng.bytes(length)
    
    def chisquare(
        self,
        df:   float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.chisquare(df, size)
    
    def noncentral_chisquare(
        self,
        df:   float | np.typing.ArrayLike,
        nonc: float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.noncentral_chisquare(df, nonc, size)
    
    def dirichlet(
        self,
        alpha: float | np.typing.ArrayLike,
        size:  tuple[int, ...] | typing.Size | None = None
    ) -> NDArray[np.float64]:
        return self._rng.dirichlet(alpha, size)
    
    def exponential(
        self,
        scale: float | np.typing.ArrayLike,
        size:  tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.exponential(scale, size)
    
    def f(
        self,
        dfnum: float | np.typing.ArrayLike,
        dfden: float | np.typing.ArrayLike,
        size:  tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.f(dfnum, dfden, size)
    
    def noncentral_f(
        self,
        dfnum: float | np.typing.ArrayLike,
        dfden: float | np.typing.ArrayLike,
        nonc:  float | np.typing.ArrayLike,
        size:  tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.noncentral_f(dfnum, dfden, nonc, size)
    
    def gamma(
        self,
        shape: float | np.typing.ArrayLike,
        scale: float | np.typing.ArrayLike,
        size:  tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.gamma(shape, scale, size)
    
    def geometric(
        self,
        p:    float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> int | NDArray[np.int64]:
        return self._rng.geometric(p, size)
    
    def gumbel(
        self,
        loc:   float | np.typing.ArrayLike,
        scale: float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.gumbel(loc, scale, size)
    
    def hypergeometric(
        self,
        ngood:   int,
        nbad:    int,
        nsample: int,
        size:    tuple[int, ...] | typing.Size | None = None
    ) -> NDArray[np.int64]:
        return self._rng.hypergeometric(ngood, nbad, nsample, size)
    
    def integers(
        self,
        low:   int,
        high:  int,
        size:  tuple[int, ...] | typing.Size | None = None,
        dtype: typing.dtype = typing.float32,
    ) -> NDArray[Any]:
        return self._rng.integers(low, high, size, dtype.cpu)
    
    def laplace(
        self,
        loc:   float | np.typing.ArrayLike,
        scale: float | np.typing.ArrayLike,
        size:  tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.laplace(loc, scale, size)
    
    def logistic(
        self,
        loc:   float | np.typing.ArrayLike,
        scale: float | np.typing.ArrayLike,
        size:  tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.logistic(loc, scale, size)
    
    def lognormal(
        self,
        mean:  float | np.typing.ArrayLike,
        sigma: float | np.typing.ArrayLike,
        size:  tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.lognormal(mean, sigma, size)
    
    def logseries(
        self,
        p:    float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> int | NDArray[np.int64]:
        return self._rng.logseries(p, size)
    
    def multinomial(
        self,
        n:     int | np.typing.ArrayLike,
        pvals: float | np.typing.ArrayLike,
        size:  tuple[int, ...] | typing.Size | None = None
    ) -> int | NDArray[np.int64]:
        return self._rng.multinomial(n, pvals, size)

    def multivariate_hypergeometric(
        self,
        colors:  int | np.typing.ArrayLike,
        nsample: int,
        size:    tuple[int, ...] | typing.Size | None = None,
        method:  Literal['marginals', 'count'] = 'marginals'
    ) -> NDArray[np.int64]:
        return self._rng.multivariate_hypergeometric(
            colors, nsample, size, method)
        
    def multivariate_normal(
        self,
        mean:        float | np.typing.ArrayLike,
        cov:         float | np.typing.ArrayLike,
        size:        tuple[int, ...] | typing.Size | None = None,
        check_valid: Literal['warn', 'raise', 'ignore'] = ...,
        tol: float = ...,
        method: Literal['svd', 'eigh', 'cholesky'] = 'svd'
    ) -> NDArray[np.float64]:
        return self._rng.multivariate_normal(
            mean, cov, size, check_valid, tol, method=method)
        
    def negative_binomial(
        self,
        n:    float | np.typing.ArrayLike,
        p:    float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> int | NDArray[np.int64]:
        return self._rng.negative_binomial(n, p, size)
    
    def pareto(
        self,
        a:    float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.pareto(a, size)
    
    def permutation(
        self,
        x:    float | np.typing.ArrayLike,
        axis: int = ...,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> NDArray[np.int64]:
        return self._rng.permutation(x, axis, size)
    
    def permuted(
        self,
        x:    float | np.typing.ArrayLike,
        axis: int = ...,
    ) -> NDArray[Any]:
        return self._rng.permuted(x, axis=axis)
    
    def poisson(
        self,
        lam:  float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> int | NDArray[np.int64]:
        return self._rng.poisson(lam, size)
    
    def power(
        self,
        a:    float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.power(a, size)
    
    def rayleigh(
        self,
        scale: float | np.typing.ArrayLike,
        size:  tuple[int, ...] | typing.Size | None = None
    ) -> float | NDArray[np.float64]:
        return self._rng.rayleigh(scale, size)
    
    def standard_t(
        self,
        df:   float | np.typing.ArrayLike,
        size: tuple[int, ...] | typing.Size | None = None,
    ) -> float | NDArray[np.float64]:
        return self._rng.standard_t(df, size)
    
    def standard_cauchy(
        self,
        size: tuple[int, ...] | typing.Size | None = None,
    ) -> float | NDArray[np.float64]:
        return self._rng.standard_cauchy(size)
    
    def standard_exponential(
        self,
        size:   tuple[int, ...] | typing.Size | None = None, 
        dtype:  typing.dtype = typing.float32,
        method: Literal['zig', 'inv'] = 'zig'
    ) -> float | NDArray[np.float64]:
        return self._rng.standard_exponential(size, dtype.cpu, method)
    
    def standard_gamma(
        self,
        shape: float | np.typing.ArrayLike,
        size:  tuple[int, ...] | typing.Size | None = None, 
        dtype: typing.dtype = typing.float32,
    ) -> float | NDArray[np.float64]:
        return self._rng.standard_gamma(shape, size, dtype.cpu)
    
    def standard_normal(
        self,
        size:  tuple[int, ...] | typing.Size | None = None, 
        dtype: typing.dtype = typing.float32,
    ) -> float | NDArray[np.float64]:
        return self._rng.standard_normal(size=size, dtype=dtype.cpu)
    
    def choice(
        self,
        a:       int | np.typing.ArrayLike,
        size:    tuple[int, ...] | typing.Size | None = None, 
        replace: bool = True,
        p:       np.typing.ArrayLike | None = None,
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

