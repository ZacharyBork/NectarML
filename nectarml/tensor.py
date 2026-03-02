from __future__ import annotations

from typing import Any, Literal
from collections.abc import Sequence, Callable

import numpy as np

import _nectarml
from nectarml import typing
import nectarml._core as _core

class Tensor():
    def __init__(
        self,
        data: Any,
        shape: tuple[int, ...] | None = None,
        dtype: typing.DTypeLike = typing.float32,
        device: Literal['cpu', 'cuda'] = 'cpu',
        requires_grad: bool = False,
        _children = ()
    ) -> None:
        self._init_tensor(data, shape, dtype, device)       
        
        self.grad: np.ndarray | None = None
        self._grad_ptr: int | None = None
        self.requires_grad = requires_grad
        
        self._backward = lambda : None
        self._prev: set[Tensor] = set(_children)
        
    def _init_tensor(
        self, 
        data: Any,
        shape: tuple[int, ...] | None = None,
        dtype: typing.DTypeLike = typing.float32,
        device: Literal['cpu', 'cuda'] = 'cpu'
    ) -> None:
        self.device = device
        if self.device == 'cpu':
            self.data = np.array(data, dtype=dtype)
            self.shape = shape or self.data.shape
            self._dtype = dtype
        elif self.device == 'cuda': 
            if shape is None:
                raise ValueError(
                    'Unable to initialize CUDA Tensor without explicit shape.')
            self._data_ptr = data
            self.shape = shape
            self._dtype = dtype 
        else: raise ValueError(f'Invalid device type: {self.device}')

    # GRADIENTS
    
    def _deallocate_grad(self) -> None:
        if self.device == 'cuda':
            if self._grad_ptr is not None:
                self._grad_ptr = _nectarml.free_cuda(self._grad_ptr)
        else: self.grad = None

    def _allocate_grad(self) -> None:
        if self.device == 'cuda':
            self._deallocate_grad()
            self._grad_ptr = _nectarml.alloc_cuda_full(
                self.shape, self.dtype, 0.0)
        else: self.grad = np.zeros_like(self.data)
        
    def backward(self) -> None:
        assert self.ndim == 0 or self.size == 1, \
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
        self._allocate_grad()
        for node in graph: node._backward()
    
    def zero_grad(self) -> None:
        if self.requires_grad:
            self._allocate_grad()
    
    # PROPERTIES
      
    @property
    def dtype(self) -> int:
        if self.device == 'cpu': return self._dtype
        else: return _core.cuda.DTYPE_MAP[self._dtype]
    
    @property
    def ndim(self) -> int:
        return len(self.shape)
    
    @property
    def size(self) -> int:
        return np.prod(self.shape)
        
    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
        
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value
        if value:
            if self.device == 'cuda' and self._grad_ptr is None \
            or self.device == 'cpu' and self.grad is None:
                self._allocate_grad()
        else: self._deallocate_grad()
    
    # DEVICE / DTYPE        
        
    def to(
        self,
        device: Literal['cpu', 'cuda'],
        dtype: typing.DTypeLike | None = None
    ) -> Tensor: 
        if device == self.device:
            if dtype is None or dtype == self._dtype:
                return self
        
        dtype = dtype or self._dtype
        if device == 'cuda':
            shape = self.shape
            if self.device == 'cpu':
                data = _nectarml.to_cuda(
                    self.data.astype(dtype), self.size, self.dtype)
            else:
                dest_dtype = _core.cuda.DTYPE_MAP[dtype]
                data = _nectarml.cast_tensor(
                    self._data_ptr, self.size, self.dtype, dest_dtype)
       
        elif device == 'cpu':
            shape = None
            if self.device == 'cuda':
                data = _nectarml.to_cpu(self._data_ptr, self.shape, self.dtype)
                data = data.astype(dtype)
            else: data = self.data
            
        else: raise ValueError(f'Invalid device type: {device}')
        
        new = Tensor(data=data, shape=shape, dtype=dtype, device=device, 
            requires_grad=self.requires_grad)
        new.grad = self.grad
        new._prev = self._prev
        new._backward = self._backward
        return new
        
    def cuda(self) -> Tensor: return self.to(device='cuda')

    def cpu(self) -> Tensor: return self.to(device='cpu')
    
    # UTILS
    
    def _validate_other(self, other: Tensor) -> None:
        assert isinstance(other, Tensor)
        
    def _numerical_to_tensor(self, other: int | float) -> Tensor:
        new = Tensor(np.full_like(self.data, other), dtype=self.dtype)
        return new.to(self.device)
    
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
        new = Tensor(data=data, device=self.device, dtype=self.dtype, 
            requires_grad=_requires_grad, _children=children)
        return new.to(self.device)
    
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
    ) -> typing.ArrayLike:
        return _core.reductions.argmin(self.data, dim=dim, keepdims=keepdims)
        
    def argmax(
        self, 
        dim: int | None = None, 
        keepdims: bool = False
    ) -> typing.ArrayLike:
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
    
    # GARBAGE COLLECTION
    
    def __del__(self) -> None:
        if self.device == 'cuda' and self._data_ptr is not None:
            _nectarml.free_cuda(self._data_ptr)
        
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
        out_data, _backward = _core.math.add(self.data, other.data)
        out = self._build_output_tensor(out_data, children)
        def _backward_hook():
            grad = _backward(out.grad)
            if self.requires_grad: self.grad += grad
            if other.requires_grad: other.grad += grad
        out._backward = _backward_hook
        return out
    
    def __radd__(self, other: Tensor | int | float) -> Tensor:
        return self + other

    def __sub__(self, other: Tensor | int | float) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        out_data, _backward = _core.math.subtract(self.data, other.data)
        out = self._build_output_tensor(out_data, children)
        def _backward_hook():
            grad = _backward(out.grad)
            if self.requires_grad: self.grad += grad
            if other.requires_grad: other.grad += grad
        out._backward = _backward_hook
        return out
    
    def __rsub__(self, other: Tensor | int | float) -> Tensor:
        return (-self) + other
    
    def __neg__(self) -> Tensor:
        out_data, _backward = _core.math.negate(self.data)
        out = self._build_output_tensor(out_data, (self,))
        def _backward_hook():
            if self.requires_grad: self.grad += _backward(out.grad)
        out._backward = _backward_hook
        return out

    def __mul__(self, other: Tensor | int | float) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        out_data, _backward = _core.math.multiply(self.data, other.data)
        out = self._build_output_tensor(out_data, children)
        
        def _backward_hook():
            a_grad, b_grad = _backward(out.grad)
            if self.requires_grad: self.grad += a_grad
            if other.requires_grad: other.grad += b_grad
            
        out._backward = _backward_hook
        return out
    
    def __rmul__(self, other: Tensor | int | float) -> Tensor:
        return self * other
    
    def __matmul__(self, other: Tensor) -> Tensor:
        assert isinstance(other, Tensor)
        assert other.data.shape == self.data.shape
        if self.ndim == 1 or other.ndim == 1:
            raise NotImplementedError('matmul not supported for 1D tensors.')
        
        out_data, _backward = _core.math.matmul(self.data, other.data)
        out = self._build_output_tensor(out_data, (self, other))
        
        def _backward_hook():
            a_grad, b_grad = _backward(out.grad)
            if self.requires_grad: self.grad += a_grad
            if other.requires_grad: other.grad += b_grad
            
        out._backward = _backward_hook
        return out
    
    def __rmatmul__(self, other: Tensor) -> Tensor: return other @ self
    
    def __pow__(self, exponent: float | int) -> Tensor: 
        return self._eval_core_function(
            lambda x : _core.math.pow(x, exponent))
    
    def __rpow__(self, exponent: float | int) -> Tensor: 
        raise NotImplementedError
    
    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        return self * other ** -1
    
    def __rtruediv__(self, other: Tensor | float | int) -> Tensor:
        return (self ** -1) * other
    
    def __abs__(self) -> Tensor:
        return self._eval_core_function(_core.math.abs)
    
    
    
