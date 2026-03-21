import atexit
import _nectarml

atexit.register(_nectarml.destroy_cublas_handle)

from . import amp, autograd, cpu, cuda, nn, optim, functional, utils
from .tensor import Tensor
from .autograd import no_grad
from .creation import (
    clone, zeros_like, ones_like, rand_like, full_like, empty_like, tril,
    zeros, ones, rand, randn, full, empty, eye, arange, linspace)
from .typing import (
    float, float16, float32, half, double, int, int8, int16, int32, int64, 
    short, long, uint, uint8, uint16, uint32, uint64, ArrayLike, DTypeLike)
from .utils import (
    save, load, is_inf, is_finite, is_nan, has_inf, has_nan)
