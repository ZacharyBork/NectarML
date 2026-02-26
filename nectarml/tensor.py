from __future__ import annotations

from typing import Any, Literal
from collections.abc import Sequence, Callable

import numpy as np

from nectarml.typing import ArrayLike, DTypeLike, float32
import nectarml._core as _core

class Tensor():
    def __init__(
        self,
        data: ArrayLike,
        dtype: DTypeLike = float32,
        device: Literal['cpu', 'cuda'] = 'cpu',
        requires_grad: bool = False,
        _children = ()
    ) -> None:
        self.data = np.array(data, dtype=dtype)
        self.dtype = dtype
        self.device = device
                
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        
        self._backward = lambda : None
        self._prev: set[Tensor] = set(_children)

    # GRADIENTS

    def backward(self) -> None:
        assert self.ndim == 0 or self.data.size == 1, \
            'backward() can only be called on scalar tensors.'
        
        visited: set[int] = set()
        graph: list[Tensor] = []
        
        def build_graph(node: Tensor):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build_graph(child)
                graph.append(node)
        
        build_graph(self)
        graph.reverse()
        self.grad = np.ones_like(self.data)
        for node in graph: node._backward()
    
    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
    
    # PROPERTIES
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape
    
    @shape.setter
    def shape(self, value: tuple[int, ...]) -> None:
        self.data = self.data.reshape(value)
        
    @property
    def ndim(self) -> int:
        return self.data.ndim
        
    @property
    def device(self) -> str:
        return self._device
    
    @device.setter
    def device(self, value: str) -> None:
        self._device = value
        
    @property
    def dtype(self) -> DTypeLike:
        return self._dtype
    
    @dtype.setter
    def dtype(self, value: DTypeLike) -> None:
        self._dtype = value
        self.data = self.data.astype(dtype=value, copy=False)
        
    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
        
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value
        if value and self.grad is None:
            self.grad = np.zeros_like(self.data)
        elif not value: self.grad = None
    
    # UTILS
    
    def _validate_other(self, other: Tensor) -> None:
        assert isinstance(other, Tensor)
        
    def _numerical_to_tensor(self, other: int | float) -> Tensor:
        return Tensor(np.full_like(self.data, other))
    
    def _handle_tensor_or_numerical(
        self, 
        other: Tensor | int | float
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        if isinstance(other, (float, int)):
            other = self._numerical_to_tensor(other)
            children = (self,)
        else: 
            self._validate_other(other)
            children = (self, other)
        return other, children
    
    def _build_output_tensor(
        self, 
        data: np.ndarray, 
        children: tuple[Tensor, ...]
    ) -> Tensor:
        _requires_grad = False
        for child in children:
            if child.requires_grad: _requires_grad = True
        return Tensor(
            data=data, device=self.device, dtype=self.dtype, 
            requires_grad=_requires_grad, _children=children)
    
    def _eval_core_function(
        self,
        func: Callable[
            [np.ndarray], 
            tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]]
    ) -> Tensor:
        out_data, _backward = func(self.data)
        out = self._build_output_tensor(out_data, (self,))
        def _backward_hook():
            if self.requires_grad:
                self.grad += _backward(out.grad)
        out._backward = _backward_hook
        return out
    
    # REDUCTIONS
    
    def min(
        self, 
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.reductions.min(x, dim=dim, keepdims=keepdims))
    
    def max(
        self, 
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.reductions.max(x, dim=dim, keepdims=keepdims))
    
    def argmin(
        self, 
        dim: int | None = None, 
        keepdims: bool = False
    ) -> ArrayLike:
        return _core.reductions.argmin(self.data, dim=dim, keepdims=keepdims)
        
    def argmax(
        self, 
        dim: int | None = None, 
        keepdims: bool = False
    ) -> ArrayLike:
        return _core.reductions.argmax(self.data, dim=dim, keepdims=keepdims)
    
    def mean(
        self, 
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.reductions.mean(x, dim=dim, keepdims=keepdims))
        
    def sum(
        self, 
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        initial: int | float = 0
    ) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.reductions.sum(x, dim, keepdims, initial))
    
    def prod(
        self, 
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        initial: int | float = 1
    ) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.reductions.prod(x, dim, keepdims, initial))
        
    # MATH OPS
    
    def abs(self) -> Tensor: return self._eval_core_function(_core.math.abs)
    
    def exp(self) -> Tensor: return self._eval_core_function(_core.math.exp)
    
    def log(self) -> Tensor: return self._eval_core_function(_core.math.log)
    
    def sqrt(self) -> Tensor: return self._eval_core_function(_core.math.sqrt)
    
    def sin(self) -> Tensor: return self._eval_core_function(_core.math.sin)
    
    def cos(self) -> Tensor: return self._eval_core_function(_core.math.cos)
    
    def tanh(self) -> Tensor: return self._eval_core_function(_core.math.tanh)
    
    def sigmoid(self) -> Tensor: return ((-self).exp() + 1) ** -1
    
    # RESHAPING
    
    def reshape(self, shape: tuple[int, ...]) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.shapes.reshape(x, shape))
        
    def flatten(self) -> Tensor:
        return self._eval_core_function(_core.shapes.flatten)
    
    def squeeze(self, dim: int | tuple[int, ...] | None) -> Tensor: 
        return self._eval_core_function(
            lambda x : _core.shapes.squeeze(x, dim))
    
    def unsqueeze(self, dim: int | tuple[int, ...]) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.shapes.unsqueeze(x, dim))
        
    def transpose(self, dims: Sequence[int] | None) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.shapes.transpose(x, dims))

    def swapdims(self, dim1: int, dim2: int) -> Tensor: 
        return self._eval_core_function(
            lambda x : _core.shapes.swapdims(x, dim1, dim2))
        
    def permute(self, dims: Sequence[int] | None) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.shapes.permute(x, dims))

    def expand(self, shape: tuple[int, ...]) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.shapes.expand(x, shape))

    def broadcast_to(self, shape: tuple[int, ...]) -> Tensor:
        return self.expand(shape)
        
    # GETTERS / SETTERS
        
    def __getitem__(self, key: Any) -> int | float:
        return self.data[key]
    
    def __setitem__(self, key: Any, value: int | float) -> None:
        self.data[key] = value
        
    def __str__(self) -> str: return self.data.__str__()
    
    def __repr__(self) -> str:
        return (
            f'shape: {self.shape}\n'
            f'data: {self.data}\n'
            f'grad: {self.grad}\n'
            f'requires_grad: {self.requires_grad}\n'
            f'_prev: {self._prev}')
    
    def __len__(self) -> int: return self.data.__len__()
    
    def __hash__(self) -> int: return id(self)
    
    # COMPARISON
    
    def __eq__(self, other: Tensor) -> np.ndarray:
        return self.data == other.data
    
    def __lt__(self, other: Tensor) -> np.ndarray: 
        return self.data < other.data
    
    def __gt__(self, other: Tensor) -> np.ndarray:
        return self.data > other.data
    
    # MATH OPS
    
    def __add__(self, other: Tensor | int | float) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        out = self._build_output_tensor(self.data + other.data, children)
        
        def _backward():
            if self.requires_grad: self.grad += out.grad
            if other.requires_grad: other.grad += out.grad
            
        out._backward = _backward
        return out
    
    def __radd__(self, other: Tensor | int | float) -> Tensor:
        return self + other

    def __sub__(self, other: Tensor | int | float) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        out = self._build_output_tensor(self.data - other.data, children)
        
        def _backward():
            if self.requires_grad: self.grad += out.grad
            if other.requires_grad: other.grad += -out.grad
            
        out._backward = _backward
        return out
    
    def __rsub__(self, other: Tensor | int | float) -> Tensor:
        return (-self) + other
    
    def __neg__(self) -> Tensor:
        out = self._build_output_tensor(-self.data, (self,))
        def _backward():
            if self.requires_grad: self.grad += -out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Tensor | int | float) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        out = self._build_output_tensor(self.data * other.data, children)
        
        def _backward():
            if self.requires_grad: self.grad += other.data * out.grad
            if other.requires_grad: other.grad += self.data * out.grad
            
        out._backward = _backward
        return out
    
    def __rmul__(self, other: Tensor | int | float) -> Tensor:
        return self * other
    
    def __matmul__(self, other: Tensor) -> Tensor:
        assert isinstance(other, Tensor)
        assert other.data.shape == self.data.shape
        if self.ndim == 1 or other.ndim == 1:
            raise NotImplementedError('matmul not supported for 1D tensors.')
        
        out = self._build_output_tensor(
            np.matmul(self.data, other.data), (self, other))
        
        def _backward():
            if self.requires_grad:
                self.grad += np.matmul(
                    out.grad, np.swapaxes(other.data, -1, -2))
            if other.requires_grad:
                other.grad += np.matmul(
                    np.swapaxes(self.data, -1, -2), out.grad)
            
        out._backward = _backward
        return out
    
    def __rmatmul__(self, other: Tensor) -> Tensor: return other @ self
    
    def __pow__(self, exponent: float | int) -> Tensor: 
        out = self._build_output_tensor(self.data ** exponent, (self,))
        
        def _backward():
            if self.requires_grad:
                self.grad += exponent * (self.data ** (exponent-1)) * out.grad
            
        out._backward = _backward
        return out
    
    def __rpow__(self, exponent: float | int) -> Tensor: 
        raise NotImplementedError
    
    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        return self * other ** -1
    
    def __rtruediv__(self, other: Tensor | float | int) -> Tensor:
        return (self ** -1) * other
    
    def __abs__(self) -> Tensor:
        return self._eval_core_function(_core.math.abs)
    
    
    
