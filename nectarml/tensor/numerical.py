from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bool import BoolTensor
    
import builtins
from typing import Literal

import numpy as np

from nectarml import typing, cpu, cuda
from nectarml.tensor.tensor import tensor
from nectarml.cuda.memory import CudaBuffer
from nectarml.amp.precision import amp_float16, amp_float32
    
class Tensor(tensor):    
    def __init__(
        self: Tensor,
        data: typing.ArrayLike,
        shape: typing.Size | tuple[builtins.int, ...] | None = None,
        dtype: typing.DTypeLike = typing.float32,
        device: Literal['cpu', 'cuda'] = 'cpu',
        requires_grad: bool = False,
        _children = ()
    ) -> None:
        '''Initializes a new Tensor object.
        
        If data is an ArrayLike object and the Tensor's device is "cpu", this
        method will fill the Tensor's data with the data from the ArrayLike
        object. In this case, if shape is provided, it will be used to set the
        Tensor's shape. If shape is not provided, the Tensor's shape will
        instead be taken from the shape of the input data.
        
        If the Tensor's device is "cuda" and data is a unintptr to tensor data
        in CUDA memory, this method will create a CudaBuffer object from the
        given pointer. In this case, shape is required, and will be used
        directly to set the Tensor's shape.
        
        If the Tensor's device is "cuda" and data is an ArrayLike object, the
        given data will be passed to CUDA and a CudaBuffer object will be
        created for the resulting pointer. In this case, shape is not required,
        and if not provided, will be assumed from the shape of the given data.
        Shape may be provided, however, and if it is, will override the shape
        of the input data when setting the Tensor's shape.
        
        Args:
            data : Either an ArrayLike object of tensor data, or a uintptr to
                a tensor in CUDA memory.
            shape : Optional if initializing with ArrayLike data. Used to
                define the shape of the Tensor.
            dtype : A DtypeLike defining the data type for the new Tensor.
            device : The device for the new Tensor, "cpu" or "cuda".
            requires_grad : A boolean defining whether the new Tensor should
                require grad or not. If True, a grad Tensor will be created
                and assigned to the new Tensor, and the new Tensor will be
                included in the computational graph to participate in gradient
                backpropagation. If False, the Tensor will not be included in
                the computational graph, and will not contribute to the 
                network's gradients.
            _children : The _prev Tensors for the newly created Tensor. If the
                new Tensor is included in the computational graph, the 
                gradients from the new Tensor will flow back to the _children
                Tensors during backpropagation. Used for autograd operations,
                generally not set manually.
        '''
        if not isinstance(data, np.ndarray): data = np.array(data)
        data = data.astype(dtype)
        
        shape = shape if isinstance(shape, typing.Size) else \
                typing.Size(shape or data.shape)
        super().__init__(
            data=Tensor._build_data(data, shape, dtype, device), 
            shape=shape, dtype=dtype, device=device, 
            requires_grad=requires_grad, _children=_children)

    ### INIT ###
        
    @classmethod
    def _build_data(
        cls,
        data:   np.ndarray,
        shape:  typing.Size,
        dtype:  typing.DTypeLike,
        device: Literal['cpu', 'cuda'] 
    ) -> np.ndarray | CudaBuffer:
        '''Initializes a Tensor from given data and optional shape.
        
        Args:
            data : Either an ArrayLike object of tensor data, or a uintptr to
                a tensor in CUDA memory.
            shape : Optional if initializing with ArrayLike data. Used to
                define the shape of the Tensor.
                
        Raises:
            ValueError : If data is uintptr to CUDA tensor data and shape is
                not provided.
            ValueError : If Tensor's device type is not valid (i.e. not "cpu"
                or "cuda").
        '''
        match device:
            case 'cpu': ref = data
            case 'cuda': 
                ref = CudaBuffer(
                    cuda.data_to_cuda(data, shape.numel(), dtype), dtype)
            case _: raise ValueError(f'Invalid device type: {device}')
        return ref

    ### UTILS ###

    def detach(self: Tensor) -> Tensor:
        '''Returns a copy of the tensor detached from the computation graph.
        
        NOTE: The newly created tensor will share the same same underlying
        storage. As such, modifying the resulting detached tensor in-place will 
        also modify the original tensor.
        
        Returns:
            tensor : A detached copy of the tensor this method is called on.
        '''
        if self.device == 'cuda':
            return self._from_buffer(
                self._buffer, self.shape, self.dtype, self.requires_grad)
        else: 
            return self._from_data(
                self.data.view(), self.shape, self.dtype, self.requires_grad)
    
    def detach_(self: Tensor) -> None:
        '''In-place detach. Detaches given tensor from the computation graph. 
        
        WARNING: This will corrupt the gradients of any tensors which depend on 
        the tensor this is called from!
        
        Detaches the tensor this method is called on from the computation graph
        by disabling requires_grad and clearing all autograd data.
        '''
        self.requires_grad = False
        self._backward = None
        self._prev.clear()
    
    def requires_grad_(self: Tensor, value: bool) -> Tensor:
        '''In-place setter for tensor.requires_grad.
        
        If value=True and the given tensor does not already have a grad tensor,
        this method will allocate a new grad tensor. If value=False and the
        tensor does have a grad tensor, the grad tensor will be deallocated and
        the memory will be freed (or buffer decremented, in the case of CUDA
        tensors).
        
        NOTE: Calling this function with value=True inside of a no_grad context
        will bypass the context and set requires_grad=True.
        
        Args:
            value : The new value for requires_grad.
            
        Returns:
            tensor : A reference to the tensor that this method was called on.
                Useful for chaining ops.
        '''
        self.requires_grad = value
        if value and self.grad is None: self._allocate_grad()
        elif not value: self._deallocate_grad()
        return self
    
    def fill_(
        self: Tensor, 
        fill_value: builtins.float | builtins.int
    ) -> Tensor:
        '''In-place fill method. Fills given tensor data with fill_value.
        
        Args:
            fill_value : The (float|int) value to fill the tensor's data with.
            
        Returns:
            tensor : A reference to the tensor that this method was called on.
                Useful for chaining ops.
        '''
        if self.device == 'cuda':
            new_ptr = cuda.memory.alloc_cuda_full(
                self.size, self.dtype, fill_value)
            old_buffer = self._buffer
            self._buffer = CudaBuffer(new_ptr, self.dtype)
            old_buffer.decrement()
        else: self.data.fill(fill_value)
        return self
    
    def zero_(self: Tensor) -> Tensor:
        '''In-place zero-fill method. Fills given tensor's data with zeros.
        
        Returns:
            tensor : A reference to the tensor that this method was called on.
                Useful for chaining ops.
        '''
        return self.fill_(0.0)
    
    def _broadcast_grad(
        self,
        grad: Tensor,
        original_shape: tuple[builtins.int, ...] | typing.Size, 
    ) -> Tensor:
        if grad.shape == original_shape: return grad

        ndim_diff = grad.ndim - len(original_shape)
        axes = list(range(ndim_diff))
        shapes = zip(original_shape, grad.shape[ndim_diff:])
        for i, (s, g) in enumerate(shapes):
            if s != g: axes.append(i + ndim_diff)
        
        if axes: grad = grad.sum(dim=tuple(axes), keepdim=False)
        if grad.shape != original_shape: grad = grad.reshape(original_shape)
        return grad

    def _build_output_tensor(
        self: Tensor, 
        data: np.ndarray | builtins.int, 
        children: tuple[Tensor, ...]
    ) -> Tensor:
        '''Helper method to build output tensor's from tensor ops.
        
        Args:
            data : Either ArrayLike data for new tensor, or uintptr to new
                tensor's data in CUDA memory.
            children : The child tensors to assign to the newly created tensor
                for backpropagation.
        '''
        _requires_grad = False
        for child in children:
            if child.requires_grad: _requires_grad = True
        return Tensor._new(data, self.shape, self.dtype, self.device, 
            _requires_grad, _children=children)
        
    ### GRADIENTS ###
    
    def _deallocate_grad(self: Tensor) -> None:
        '''Deallocated grad by setting tensor.grad to None.'''
        self.grad = None

    def _allocate_grad(self: Tensor, fill_value: builtins.float = 0.0) -> None:
        '''Allocates gradient tensor for given tensor.
        
        Args:
            fill_value : The value to fill the new grad tensor with.
        '''
        self._deallocate_grad()
        self.grad = Tensor._new(
            np.full(self.shape, fill_value, typing.float32), 
            self.shape, typing.float32, self.device, requires_grad=False) 
        
    def backward(self: Tensor) -> None:
        '''Gradient backpropagation method.
        
        Builds a topo graph from all children of the tensor it was called on
        recursively. Then allocates a new gradient tensor for the given tensor,
        fills the gradient tensor with ones, and then walks the graph, calling
        each subsequent child's _backward() method.
        '''
        assert self.ndim == 0 or self.size == 1, \
            'backward() can only be called on scalar tensors.'
        
        visited: set[builtins.int] = set()
        graph: list[tensor] = []
        
        def build_graph(node: tensor):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build_graph(child)
                graph.append(node)
        
        build_graph(self)
        graph.reverse()
        
        for node in graph:
            if node.requires_grad and node is not self:
                node._allocate_grad(fill_value=0.0)
                    
        self._allocate_grad(fill_value=1.0)
        for node in graph: node._backward()
        
        for node in graph:
            if node._prev:
                node._prev.clear()
                node._backward = lambda : None
    
    def zero_grad(self: Tensor) -> None:
        '''Zeros values in the grad tensor of the tensor it is called on.'''
        if self.requires_grad and self.grad is not None:
            self.grad.zero_()
    
    ### COMPARISON ###
    
    def __eq__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> BoolTensor:
        '''tensor elementwise equality operator.
        
        Args:
            other : The tensor, int, or float value to compare the given tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new tensor with the same shape, device,
                and Dtype as the tensor this operator is called on.
                
        Returns:
            tensor : A boolean tensor denoting whether each element is equal
                to the corresponding value in the "other" tensor. Note: This
                return tensor can be turned into a binary mask like so:
                
                    x = tensor(data, dtype=typing.float32)
                    y = tensor(data, dtype=typing.float32)
                    mask = (x == y).to(x.device, x.dtype)
        '''        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
        else: out_shape = self.shape
        
        if self.device == 'cuda': 
            data = cuda.math.equal(self, other, out_shape)
        else: data = cpu.math.equal(self, other)
        return self._to_bool(data, out_shape)
        
    def __lt__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> BoolTensor:
        '''tensor elementwise less than operator.
        
        Args:
            other : The tensor, int, or float value to compare the given tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new tensor with the same shape, device,
                and Dtype as the tensor this operator is called on.
                
        Returns:
            tensor : A boolean tensor denoting whether each element is less 
                than to the corresponding value in the "other" tensor. Note: 
                This return tensor can be turned into a binary mask like so:
                
                    x = tensor(data, dtype=typing.float32)
                    y = tensor(data, dtype=typing.float32)
                    mask = (x < y).to(x.device, x.dtype)
        '''        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
        else: out_shape = self.shape
        
        if self.device == 'cuda':
            data = cuda.math.less_than(self, other, out_shape)
        else: data = cpu.math.less_than(self, other)
        return self._to_bool(data, out_shape)
    
    def __le__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> BoolTensor:
        '''tensor elementwise less than or equal operator.
        
        Args:
            other : The tensor, int, or float value to compare the given tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new tensor with the same shape, device,
                and Dtype as the tensor this operator is called on.
                
        Returns:
            tensor : A boolean tensor denoting whether each element is less 
                than or equal to the corresponding value in the "other" tensor. 
                Note: This return tensor can be turned into a binary mask 
                like so:
                
                    x = tensor(data, dtype=typing.float32)
                    y = tensor(data, dtype=typing.float32)
                    mask = (x <= y).to(x.device, x.dtype)
        '''
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
        else: out_shape = self.shape

        if self.device == 'cuda':
            data = cuda.math.less_than_or_equal(self, other, out_shape)
        else: data = cpu.math.less_than_or_equal(self, other)
        return self._to_bool(data, out_shape)
    
    def __gt__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> BoolTensor:
        '''tensor elementwise greater than operator.
        
        Args:
            other : The tensor, int, or float value to compare the given tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new tensor with the same shape, device,
                and Dtype as the tensor this operator is called on.
                
        Returns:
            tensor : A boolean tensor denoting whether each element is greater 
                than to the corresponding value in the "other" tensor. Note: 
                This return tensor can be turned into a binary mask like so:
                
                    x = tensor(data, dtype=typing.float32)
                    y = tensor(data, dtype=typing.float32)
                    mask = (x > y).to(x.device, x.dtype)
        '''        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
        else: out_shape = self.shape
        
        if self.device == 'cuda':
            data = cuda.math.greater_than(self, other, out_shape)
        else: data = cpu.math.greater_than(self, other)
        return self._to_bool(data, out_shape)
    
    def __ge__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> BoolTensor:
        '''tensor elementwise greater than or equal operator.
        
        Args:
            other : The tensor, int, or float value to compare the given tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new tensor with the same shape, device,
                and Dtype as the tensor this operator is called on.
                
        Returns:
            tensor : A boolean tensor denoting whether each element is greater 
                than or equal to the corresponding value in the "other" tensor. 
                Note: This return tensor can be turned into a binary mask 
                like so:
                
                    x = tensor(data, dtype=typing.float32)
                    y = tensor(data, dtype=typing.float32)
                    mask = (x >= y).to(x.device, x.dtype)
        '''        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
        else: out_shape = self.shape
        
        if self.device == 'cuda':
            data = cuda.math.greater_than_or_equal(self, other, out_shape)
        else: data = cpu.math.greater_than_or_equal(self, other)
        return self._to_bool(data, out_shape)
    
    def __hash__(self: tensor) -> builtins.int: 
        '''Hash override. Returns memory address of tensor.
        
        Returns:
            int : Address in system memory of the tensor.
        '''
        return id(self)
    
    ### ROUNDING ###
    
    def floor(self) -> Tensor:
        '''Takes the floor of the given tensor's data, returns as new tensor.
        
        Returns:
            tensor : The resulting tensor from the floor operation.
        '''
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.floor(self)
        else: out_data = np.floor(self.data).astype(self.dtype)
        out = Tensor._new(out_data, self.shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad
                
        out._backward = _backward
        return out
    
    def ceil(self) -> Tensor:
        '''Takes the ceil of the given tensor's data, returns as new tensor.
        
        Returns:
            tensor : The resulting tensor from the ceil operation.
        '''
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.ceil(self)
        else: out_data = np.ceil(self.data).astype(self.dtype)
        out = Tensor._new(out_data, self.shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad
                
        out._backward = _backward
        return out
    
    def round(self, precision: builtins.int = 0) -> Tensor:
        '''Rounds the data of a given tensor and returns as new tensor.
        
        NOTE: Currently, the precision argument only works on CPU tensors. CUDA
        tensors will be rounded to the nearest whole integer (though they will
        keep the DType of the original tensor).
        
        Args:
            precision : The number of decimal places to round the tensor's data
                to (only works on CPU tensors currently).
        
        Returns:
            tensor : The resulting tensor from the rounding operation.
        '''
        
        # CUDA ROUND NEEDS PRECISION!
        
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.round(self)
        else: out_data = np.round(self.data, decimals=precision)
        out = Tensor._new(out_data, self.shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad
                
        out._backward = _backward
        return out
    
    ### MATH DUNDERS ###
    
    def __iadd__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Adds other (tensor|int|float) to given tensor's data in-place.
        
        Args:
            other : The other tensor or scalar value to add to the tensor.
            
        Returns:
            tensor : A reference to the tensor being added to.
        '''
        if isinstance(other, Tensor):
            assert other.shape == self.shape, \
                f'grad shape mismatch: self.shape={self.shape} ' \
                f'other.shape={other.shape}'

        if self.device == 'cuda': 
            new_ptr = cuda.math.add(self, other, self.shape)
            self._buffer.decrement()
            self._buffer = CudaBuffer(new_ptr, self.dtype)
        else: self.data = cpu.math.add(self, other)
        return self
    
    def __add__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Adds (tensor|int|float) to tensor's data and returns new tensor.
        
        Args:
            other : The tensor or scalar value to add to the given tensor.
            
        Returns:
            tensor : Resulting tensor from addition operation.
        '''
        self_requires_grad = self.requires_grad
        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
            children = (self, other)
            other_requires_grad = other.requires_grad
        else:
            out_shape = self.shape
            children = (self,)
            other_requires_grad = False
            
        if self.device == 'cuda': 
            out_data = cuda.math.add(self, other, out_shape)
        else: out_data = cpu.math.add(self, other)
        out = Tensor._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=children)
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += self._broadcast_grad(out.grad, self.shape)
            if other_requires_grad:
                other.grad += self._broadcast_grad(out.grad, other.shape)
                
        out._backward = _backward
        return out
    
    def __radd__(
        self: Tensor, 
        other: tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Adds (tensor|int|float) to tensor's data and returns new tensor.
        
        Args:
            other : The tensor or scalar value to add to the given tensor.
            
        Returns:
            tensor : Resulting tensor from addition operation.
        '''
        return self + other

    def __isub__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Subtracts other (tensor|scalar) from given tensor's data in-place.
        
        Args:
            other : The other tensor or scalar value to subtract from the 
                tensor.
            
        Returns:
            tensor : A reference to the tensor being subtracted from.
        '''
        if isinstance(other, Tensor):
            assert other.shape == self.shape, \
                f'grad shape mismatch: self.shape={self.shape} ' \
                f'other.shape={other.shape}'
        
        if self.device == 'cuda': 
            new_ptr = cuda.math.subtract(self, other, self.shape)
            self._buffer.decrement()
            self._buffer = CudaBuffer(new_ptr, self.dtype)    
        else: self.data = cpu.math.subtract(self, other)
        return self

    def __sub__(self, other: Tensor | builtins.int | builtins.float) -> Tensor:
        '''Subtracts (tensor|scalar) from tensor's data and returns new tensor.
        
        Args:
            other : The tensor or scalar value to add to the given tensor.
            
        Returns:
            tensor : Resulting tensor from addition operation.
        '''        
        self_requires_grad = self.requires_grad
        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
            children = (self, other)
            other_requires_grad = other.requires_grad
        else:
            out_shape = self.shape
            children = (self,)
            other_requires_grad = False
        
        if self.device == 'cuda': 
            out_data = cuda.math.subtract(self, other, out_shape)   
        else: out_data = cpu.math.subtract(self, other)
        out = Tensor._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=children)
        
        def _backward() -> None:
            if self_requires_grad: 
                self.grad += self._broadcast_grad(out.grad, self.shape)
            if other_requires_grad: 
                other.grad += -self._broadcast_grad(out.grad, other.shape)

        out._backward = _backward
        return out
    
    def __rsub__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Subtracts (tensor|scalar) from tensor's data and returns new tensor.
        
        Args:
            other : The tensor or scalar value to add to the given tensor.
            
        Returns:
            tensor : Resulting tensor from addition operation.
        '''
        return (-self) + other
    
    def __neg__(self) -> Tensor:
        '''Negates the data of a given tensor.
            
        Returns:
            tensor : A new tensor with the data from the negation operation.
        '''
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':out_data = cuda.math.negate(self)
        else: out_data = cpu.math.negate(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += -out.grad

        out._backward = _backward
        return out

    def __imul__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Multiplies other (tensor|scalar) with given tensor's data in-place.
        
        Args:
            other : The other tensor or scalar value to multiply the tensor by.
            
        Returns:
            tensor : A reference to the tensor being multiplied.
        '''
        if isinstance(other, Tensor):
            assert other.shape == self.shape, \
                f'grad shape mismatch: self.shape={self.shape} ' \
                f'other.shape={other.shape}'
        
        if self.device == 'cuda': 
            new_ptr = cuda.math.multiply(self, other, self.shape)
            self._buffer.decrement()
            self._buffer = CudaBuffer(new_ptr, self.dtype)
        else: self.data = cpu.math.multiply(self, other)
        return self

    def __mul__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Multiplies tensor by (tensor|scalar) and returns new tensor.
        
        Args:
            other : The tensor or scalar value to multiply the given tensor by.
            
        Returns:
            tensor : Resulting tensor from multiplication operation.
        '''
        self_requires_grad = self.requires_grad
        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
            children = (self, other)
            other_requires_grad = other.requires_grad
        else:
            out_shape = self.shape
            children = (self,)
            other_requires_grad = False
        
        if self.device == 'cuda': 
            out_data = cuda.math.multiply(self, other, out_shape)
        else: out_data = cpu.math.multiply(self, other)
        out = Tensor._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=children)
        
        def _backward() -> None:
            if self_requires_grad: 
                grad = other * out.grad
                self.grad += self._broadcast_grad(grad, self.shape)
            if other_requires_grad: 
                grad = self * out.grad
                other.grad += self._broadcast_grad(grad, other.shape)
        
        out._backward = _backward
        return out
    
    def __rmul__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Multiplies tensor by (tensor|scalar) and returns new tensor.
        
        Args:
            other : The tensor or scalar value to multiply the given tensor by.
            
        Returns:
            tensor : Resulting tensor from multiplication operation.
        '''
        return self * other
    
    @amp_float16
    def __matmul__(self: Tensor, other: Tensor) -> Tensor:
        '''Performs a matrix multiplication between the data of two tensors.
        
        Args:
            other : The other tensor for the matrix multiplication.
            
        Returns:
            tensor : Resulting tensor from matrix multiplication operation.
        '''
        self._validate_other(other)
        if self.ndim == 1 or other.ndim == 1:
            raise NotImplementedError('matmul not supported for 1D tensors.')
        
        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda': out_data = cuda.matmul.matmul(self, other)
        else: out_data = cpu.math.matmul(self, other)
        out = self._build_output_tensor(out_data, (self, other))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad @ other.transpose(-2, -1)
            if other_requires_grad:
                other.grad += self.transpose(-2, -1) @ out.grad
        
        out._backward = _backward
        return out
    
    def __rmatmul__(self: Tensor, other: Tensor) -> Tensor: 
        '''Performs a matrix multiplication between the data of two tensors.
        
        Args:
            other : The other tensor for the matrix multiplication.
            
        Returns:
            tensor : Resulting tensor from matrix multiplication operation.
        '''
        return other @ self
    
    @amp_float32
    def __pow__(
        self: Tensor, 
        exponent: builtins.float | builtins.int
    ) -> Tensor:
        '''Raises a tensor by the given exponent and returns as new tensor.
        
        Args:
            exponent : The exponent to raise the tensor's data by.
            
        Returns:
            tensor : Resulting tensor from power operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.pow(self, exponent)
        else: out_data = cpu.math.pow(self, exponent)
        out = self._build_output_tensor(out_data, (self,))
        
        def _backward() -> None:
            if self_requires_grad: 
                self.grad += exponent * (self**(exponent-1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __rpow__(
        self: Tensor, 
        exponent: builtins.float | builtins.int
    ) -> Tensor: 
        raise NotImplementedError
    
    def __truediv__(
        self: Tensor, 
        other: Tensor | builtins.float | builtins.int
    ) -> Tensor:
        '''Divides a tensor by a (tensor/scalar) and returns as new tensor.
        
        Args:
            other : The (tensor/scalar) to act as the divisor.
            
        Returns:
            tensor : Resulting tensor from division operation.
        ''' 
        return self * other ** -1
    
    def __rtruediv__(
        self: Tensor, 
        other: Tensor | builtins.float | builtins.int
    ) -> Tensor:
        '''Divides a tensor by a (tensor/scalar) and returns as new tensor.
        
        Args:
            other : The (tensor/scalar) to act as the divisor.
            
        Returns:
            tensor : Resulting tensor from division operation.
        ''' 
        return (self ** -1) * other
    
    def __abs__(self: Tensor) -> Tensor:
        '''Takes the absolute of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from absolute operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.abs(self)
        else: out_data = cpu.math.abs(self)
        out = self._build_output_tensor(out_data, (self,))
        
        def _backward() -> None:
            if self_requires_grad: 
                self.grad += self.sign() * out.grad
        
        out._backward = _backward
        return out
    
    ### CLAMP ###
    
    def minimum(
        self: Tensor, 
        other: Tensor | builtins.float | builtins.int
    ) -> Tensor:
        '''Return the minimum of a tensor and a (tensor/scalar) as new tensor.
        
        NOTE: It "other" is a tensor, this method will perform and elementwise
        mimimum operation, comparing each element of the given tensor against 
        the corresponding element from the "other" tensor and returning the
        smaller value.
        
        Args:
            other : The (tensor/scalar) to compare against.
        
        Returns:
            tensor : Resulting tensor from the minimum operation.
        '''         
        self_requires_grad = self.requires_grad
        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
            children = (self, other)
            other_requires_grad = other.requires_grad
        else:
            out_shape = self.shape
            children = (self,)
            other_requires_grad = False
        
        if self.device == 'cuda': 
            out_data = cuda.math.minimum(self, other, out_shape)
        else: out_data = cpu.math.minimum(self, other)
        out = Tensor._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=children)
        
        def _backward() -> None:
            if self_requires_grad:
                grad = (self < other).to(self.device, self.dtype) * out.grad
                self.grad += self._broadcast_grad(grad, self.shape) 
            if other_requires_grad:
                grad = (other < self).to(other.device, other.dtype) * out.grad
                other.grad += self._broadcast_grad(grad, other.shape) 
        
        out._backward = _backward
        return out
    
    def maximum(
        self: Tensor, 
        other: tensor | builtins.float | builtins.int
    ) -> Tensor:
        '''Return the maximum of a tensor and a (tensor/scalar) as new tensor.
        
        NOTE: It "other" is a tensor, this method will perform and elementwise
        maximum operation, comparing each element of the given tensor against 
        the corresponding element from the "other" tensor and returning the
        larger value.
        
        Args:
            other : The (tensor/scalar) to compare against.
        
        Returns:
            tensor : Resulting tensor from the maximum operation.
        '''         
        self_requires_grad = self.requires_grad
        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
            children = (self, other)
            other_requires_grad = other.requires_grad
        else:
            out_shape = self.shape
            children = (self,)
            other_requires_grad = False
        
        if self.device == 'cuda': 
            out_data = cuda.math.maximum(self, other, out_shape)
        else: out_data = cpu.math.maximum(self, other)
        out = Tensor._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=children)
        
        def _backward() -> None:
            if self_requires_grad:
                grad = (self >= other).to(self.device, self.dtype) * out.grad
                self.grad += self._broadcast_grad(grad, self.shape) 
            if other_requires_grad:
                grad = (other <= self).to(other.device, other.dtype) * out.grad
                other.grad += self._broadcast_grad(grad, other.shape)
        
        out._backward = _backward
        return out
    
    def clamp(
        self: Tensor, 
        min_value: builtins.float | None = None, 
        max_value: builtins.float | None = None
    ) -> Tensor:
        '''Clamps tensor's values between min and max, returns as new tensor.
        
        Args:
            min_value : The minimum allowable value for the clamping operation,
                or None. If this is None, the minimum representable value for
                the tensor's datatype will be used instead.
            max_value : The minimum allowable value for the clamping operation,
                or None. If this is None, the maximum representable value for
                the tensor's datatype will be used instead.
        
        Returns:
            tensor : Resulting tensor from the clamp operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': 
            out_data = cuda.math.clamp(self, min_value, max_value)
        else: out_data = cpu.math.clamp(self, min_value, max_value)
        out = self._build_output_tensor(out_data, (self,))
                    
        def _backward() -> None:
            if self_requires_grad:
                mask = (self >= min_value).to(self.device, self.dtype) \
                     * (self <= max_value).to(self.device, self.dtype)
                self.grad += mask * out.grad
        
        out._backward = _backward
        return out
        
    ### ABS ###

    def abs(self: Tensor) -> Tensor: 
        '''Takes the absolute of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the absolute operation.
        ''' 
        return self.__abs__()
        
    ### EXP ###
            
    @amp_float32
    def exp(self: Tensor) -> Tensor:
        '''Takes the exponent of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the exponent operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.exp(self)
        else: out_data = cpu.math.exp(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += out * out.grad

        out._backward = _backward
        return out
      
    ### LOG ###
            
    @amp_float32
    def log(self: Tensor) -> Tensor:
        '''Takes the logarithm of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the log operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.log(self)
        else: out_data = cpu.math.log(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += (1 / self) * out.grad

        out._backward = _backward
        return out
    
    @amp_float32
    def log2(self: Tensor) -> Tensor:
        '''Takes the log^2 of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the log2 operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.log2(self)
        else: out_data = cpu.math.log2(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += out.grad / (self * np.log(2))

        out._backward = _backward
        return out
    
    @amp_float32
    def log10(self: Tensor) -> Tensor:
        '''Takes the log^10 of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the log10 operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.log10(self)
        else: out_data = cpu.math.log10(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += out.grad / (self * np.log(10))

        out._backward = _backward
        return out
          
    ### SQRT ###
            
    def sqrt(self: Tensor) -> Tensor:
        '''Takes the square root of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the square root operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':out_data = cuda.math.sqrt(self)
        else: out_data = cpu.math.sqrt(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += (1 / (2 * out)) * out.grad

        out._backward = _backward
        return out
    
    @amp_float32
    def rsqrt(self: Tensor) -> Tensor:
        '''Takes the reciprocal sqrt of a tensor's data, returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the reciprocal square root 
                operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':out_data = cuda.math.rsqrt(self)
        else: out_data = cpu.math.rsqrt(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += -0.5 * out**3 * out.grad

        out._backward = _backward
        return out
    
    ### SIN / COS ###
            
    def sin(self: Tensor) -> Tensor:
        '''Returns the sine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the sine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.sin(self)
        else: out_data = cpu.math.sin(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            # NOTE: This builds unneeded graph nodes. Eventually, this should
            # be replaced with something akin to torch.no_grad()
            if self_requires_grad: self.grad += self.cos() * out.grad

        out._backward = _backward
        return out
    
    @amp_float32
    def asin(self: Tensor) -> Tensor:
        '''Returns the arc sine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the arcsine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.asin(self)
        else: out_data = cpu.math.asin(self)
        
        def _backward() -> None:
            if self_requires_grad:
                denom = (1 - self ** 2).sqrt().clamp(min_value=1e-7)
                self.grad += out.grad / denom
        
        out = self._build_output_tensor(out_data, (self,))
        out._backward = _backward
        return out
    
    @amp_float32
    def sinh(self: Tensor) -> Tensor:
        '''Returns the hyperbolic sine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the hyperbolic sine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.sinh(self)
        else: out_data = cpu.math.sinh(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                self.grad += self.cosh() * out.grad
        
        out._backward = _backward
        return out
    
    def asinh(self: Tensor) -> Tensor:
        '''Returns the hyperbolic arc sine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the hyperbolic arcsine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.asinh(self)
        else: out_data = cpu.math.asinh(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad / (self**2 + 1).sqrt()
        
        out._backward = _backward
        return out
        
    def cos(self: Tensor) -> Tensor: 
        '''Returns the cosine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the cosine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.cos(self)
        else: out_data = cpu.math.cos(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            # NOTE: This builds unneeded graph nodes. Eventually, this should
            # be replaced with something akin to torch.no_grad()
            if self_requires_grad: self.grad += -self.sin() * out.grad

        out._backward = _backward
        return out
    
    @amp_float32
    def acos(self: Tensor) -> Tensor:
        '''Returns the arc cosine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the arc cosine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.acos(self)
        else: out_data = cpu.math.acos(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                grad = (1 - self**2).sqrt().clamp(min_value=1e-7)
                self.grad += -out.grad / grad
        
        out._backward = _backward
        return out
    
    @amp_float32
    def cosh(self: Tensor) -> Tensor:
        '''Returns the hyperbolic cosine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the hyperbolic cosine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.cosh(self)
        else: out_data = cpu.math.cosh(self)
        out = self._build_output_tensor(out_data, (self,))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += self.sinh() * out.grad
        
        out._backward = _backward
        return out
    
    def acosh(self: Tensor) -> Tensor:
        '''Returns the hyperbolic arc cosine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the hyperbolic arc cosine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.acosh(self)
        else: out_data = cpu.math.acosh(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                grad = (self**2 - 1).sqrt().clamp(min_value=1e-7)
                self.grad += out.grad / grad
        
        out._backward = _backward
        return out
    
    ### TAN / ATAN ###
    
    @amp_float32
    def tan(self: Tensor) -> Tensor:
        '''Returns the tangent of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the tangent operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.tan(self)
        else: out_data = cpu.math.tan(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                self.grad += (1 + out**2) * out.grad
        
        out._backward = _backward
        return out
        
    def tanh(self: Tensor) -> Tensor:
        '''Returns the hyperbolic tangent of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the hyperbolic tangent operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.tanh(self)
        else: out_data = cpu.math.tanh(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward():
            if self_requires_grad: 
                self.grad += (1 - out**2) * out.grad

        out._backward = _backward
        return out
    
    def atan(self: Tensor) -> Tensor:
        '''Returns the arc tangent of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the arc tangent operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.atan(self)
        else: out_data = cpu.math.atan(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward():
            if self_requires_grad: 
                self.grad += out.grad / (1 + self**2)

        out._backward = _backward
        return out
    
    def atanh(self: Tensor) -> Tensor:
        '''Returns the hyperbolic arc tangent of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the hyperbolic arc tangent
                operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.atanh(self)
        else: out_data = cpu.math.atanh(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward():
            if self_requires_grad: 
                self.grad += out.grad / (1 - self**2).clamp(min_value=1e-7)

        out._backward = _backward
        return out
    
    def atan2(self: Tensor, other: Tensor) -> Tensor:
        '''Returns the arc tangent^2 of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the arc tangent^2 operation.
        ''' 
        self._validate_other(other)

        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        out_shape = self._broadcast_shape(self.shape, other.shape)
        if self.device == 'cuda': 
            out_data = cuda.math.atan2(other, self, out_shape)
        else: out_data = cpu.math.atan2(other, self)
        out = Tensor._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=(self, other))
        
        def _backward() -> None:
            denom = (self**2 + other**2).clamp(min_value=1e-7)
            if self_requires_grad:
                grad = out.grad * other / denom
                self.grad += self._broadcast_grad(grad, self.shape)
            if other_requires_grad:
                grad = out.grad * (-self) / denom
                other.grad += self._broadcast_grad(grad, other.shape)
                
        out._backward = _backward
        return out
    
    ### SIGN ###
    
    def sign(self: Tensor) -> Tensor:
        '''Returns the sign of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the sign operation.
        ''' 
                
        if self.device == 'cuda': out_data = cuda.math.sign(self)
        else: out_data = cpu.math.sign(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None: pass
        out._backward = _backward
        return out
    
    def copysign(self: Tensor, other: Tensor) -> Tensor:
        self._validate_other(other)

        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        out_shape = self._broadcast_shape(self.shape, other.shape)
        
        if self.device == 'cuda': 
            out_data = cuda.math.copysign(self, other, out_shape)
        else: out_data = cpu.math.copysign(self, other)
        out = Tensor._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=(self, other))
        
        def _backward() -> None:
            if self_requires_grad: 
                self.grad += self._broadcast_grad(out.grad, self.shape)
            if other_requires_grad: 
                other.grad += self._broadcast_grad(out.grad, other.shape)
                
        out._backward = _backward
        return out
    
    ### REDUCTIONS ###
    
    def min(
        self: Tensor,
        dim: builtins.int | None = None,
        keepdim: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        self_requires_grad = self.requires_grad

        if self.device == 'cuda':
            data = cuda.reductions.min(self, dim)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data = cpu.reductions.min(self, dim, keepdim)
            output_shape = typing.Size(data.shape)

        out = Tensor._new(data, output_shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))

        def _backward() -> None:
            if self_requires_grad:
                min_vals = out if keepdim else \
                        out.unsqueeze(dim) if dim is not None else \
                        out.reshape([1] * self.ndim)
                mask = (self == min_vals.expand(self.shape)).to(
                    self.device, self.dtype)
                grad = out.grad if keepdim else \
                    out.grad.unsqueeze(dim) if dim is not None else \
                    out.grad.reshape([1] * self.ndim)
                self.grad += mask * grad.expand(self.shape)

        out._backward = _backward

        if dim is not None:
            _, indices = self.sort(dim=dim)
            idx = indices.select(dim, 0)
            if keepdim: idx = idx.unsqueeze(dim)
            return out, idx
        return out

    def amin(
        self: Tensor,
        dim: builtins.int | tuple[builtins.int, ...] | None = None,
        keepdim: bool = False
    ) -> Tensor:
        self_requires_grad = self.requires_grad

        if isinstance(dim, (tuple, list)):
            ndim = self.ndim
            dims = sorted(set(d % ndim for d in dim), reverse=True)
            
            result = self
            for d in dims: result = result.amin(d, keepdim=True)
            
            if not keepdim:
                for d in dims: result = result.squeeze(d)
            return result

        if self.device == 'cuda':
            data = cuda.reductions.min(self, dim)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data = cpu.reductions.min(self, dim, keepdim)
            output_shape = typing.Size(data.shape)

        out = Tensor._new(data, output_shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))

        def _backward() -> None:
            if self_requires_grad:
                min_vals = out if keepdim else \
                    out.unsqueeze(dim) if dim is not None else \
                    out.reshape([1] * self.ndim)
                mask = (self == min_vals.expand(self.shape)).to(
                    self.device, self.dtype)
                grad = out.grad if keepdim else \
                    out.grad.unsqueeze(dim) if dim is not None else \
                    out.grad.reshape([1] * self.ndim)
                self.grad += mask * grad.expand(self.shape)

        out._backward = _backward
        return out

    def max(
        self: Tensor,
        dim: builtins.int | None = None,
        keepdim: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        self_requires_grad = self.requires_grad

        if self.device == 'cuda':
            data = cuda.reductions.max(self, dim)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data = cpu.reductions.max(self, dim, keepdim)
            output_shape = typing.Size(data.shape)

        out = Tensor._new(data, output_shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))

        def _backward() -> None:
            if self_requires_grad:
                max_vals = out if keepdim else \
                        out.unsqueeze(dim) if dim is not None else \
                        out.reshape([1] * self.ndim)
                mask = (self == max_vals.expand(self.shape)).to(
                    self.device, self.dtype)
                grad = out.grad if keepdim else \
                    out.grad.unsqueeze(dim) if dim is not None else \
                    out.grad.reshape([1] * self.ndim)
                self.grad += mask * grad.expand(self.shape)

        out._backward = _backward

        if dim is not None:
            _, indices = self.sort(dim=dim, descending=True)
            idx = indices.select(dim, 0)
            if keepdim: idx = idx.unsqueeze(dim)
            return out, idx
        return out

    def amax(
        self: Tensor,
        dim: builtins.int | tuple[builtins.int, ...] | None = None,
        keepdim: bool = False
    ) -> Tensor:
        self_requires_grad = self.requires_grad

        if isinstance(dim, (tuple, list)):
            ndim = self.ndim
            dims = sorted(set(d % ndim for d in dim), reverse=True)
            
            result = self
            for d in dims: result = result.amax(d, keepdim=True)
            
            if not keepdim:
                for d in dims: result = result.squeeze(d)
            return result

        if self.device == 'cuda':
            data = cuda.reductions.max(self, dim)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data = cpu.reductions.max(self, dim, keepdim)
            output_shape = typing.Size(data.shape)

        out = Tensor._new(data, output_shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))

        def _backward() -> None:
            if self_requires_grad:
                max_vals = out if keepdim else \
                        out.unsqueeze(dim) if dim is not None else \
                        out.reshape([1] * self.ndim)
                mask = (self == max_vals.expand(self.shape)).to(
                    self.device, self.dtype)
                grad = out.grad if keepdim else \
                    out.grad.unsqueeze(dim) if dim is not None else \
                    out.grad.reshape([1] * self.ndim)
                self.grad += mask * grad.expand(self.shape)

        out._backward = _backward
        return out
    
    def argmin(
        self: Tensor,
        dim: builtins.int | None = None,
        keepdim: bool = False
    ) -> Tensor:        
        data = np.argmin(self.cpu().numpy(), axis=dim)
        if keepdim and dim is not None:
            data = np.expand_dims(data, axis=dim)
        out = Tensor._new(data, data.shape(), typing.int32, self.device)
            
        def _backward() -> None:
            raise RuntimeError(
                'argmin is not differentiable. Use amin() if you need '
                'gradients through a min operation.')

        out._backward = _backward
        return out

    def argmax(
        self: Tensor,
        dim: builtins.int | None = None,
        keepdim: bool = False
    ) -> Tensor:        
        data = np.argmax(self.cpu().numpy(), axis=dim)
        if keepdim and dim is not None:
            data = np.expand_dims(data, axis=dim)
        out = Tensor._new(data, data.shape(), typing.int32, self.device)
        
        def _backward() -> None:
            raise RuntimeError(
                'argmargmaxin is not differentiable. Use amax() if you need '
                'gradients through a max operation.')

        out._backward = _backward
        return out
    
    def mean(
        self: Tensor, 
        dim: builtins.int | tuple[builtins.int, ...] | None = None,
        keepdim: bool = False,
    ) -> Tensor:
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            if isinstance(dim, (tuple, list)):
                result = self
                for d in sorted(dim, reverse=True):
                    result = result.mean(d, keepdim=True)
                if not keepdim:
                    result.shape = result.shape.reduce(dim, keepdim)
                return result
            
            data = cuda.reductions.mean(self, dim)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data = cpu.reductions.mean(self, dim, keepdim)
            output_shape = typing.Size(data.shape)

        out = Tensor._new(data, output_shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                n = self.size if dim is None else self.shape[dim]
                
                grad = out.grad if keepdim else \
                    out.grad.unsqueeze(dim) if dim is not None else \
                    out.grad.reshape([1] * self.ndim)
                
                self.grad += grad.expand(self.shape) / n
        
        out._backward = _backward
        return out
    
    @amp_float32
    def sum(
        self: Tensor, 
        dim: builtins.int | tuple[builtins.int, ...] | None = None,
        keepdim: bool = False,
        initial: builtins.int | builtins.float = 0.0
    ) -> Tensor:
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            if isinstance(dim, (tuple, list)):
                result = self
                for d in sorted(dim, reverse=True):
                    result = result.sum(d, keepdim=True)
                if not keepdim:
                    result.shape = result.shape.reduce(dim, keepdim)
                return result
            
            data = cuda.reductions.sum(self, dim, initial)
        else: data = cpu.reductions.sum(self, dim, keepdim, initial)

        output_shape = self.shape.reduce(dim, keepdim)
        out = Tensor._new(data, output_shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                grad = out.grad if keepdim else out.grad.unsqueeze(dim) \
                    if dim is not None else out.grad.reshape([1] * self.ndim)
                self.grad += grad.expand(self.shape)
                
        out._backward = _backward
        return out
    
    @amp_float32
    def cumsum(self: Tensor, dim: builtins.int) -> Tensor:
        self_requires_grad = self.requires_grad

        dim = dim if dim >= 0 else self.ndim + dim

        if self.device == 'cuda':
            out_data = cuda.reductions.cumsum(self, dim)
        else: out_data = np.cumsum(self.data, axis=dim)

        out = Tensor._new(out_data, self.shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))

        def _backward() -> None:
            if self_requires_grad:
                grad = out.grad
                flipped = grad.flip(dim)
                flipped_csum = flipped.cumsum(dim)
                self.grad += flipped_csum.flip(dim)

        out._backward = _backward
        return out
        
    @amp_float32
    def prod(
        self: Tensor, 
        dim: builtins.int | tuple[builtins.int, ...] | None = None,
        keepdim: bool = False,
        initial: builtins.int | builtins.float = 1.0
    ) -> Tensor:
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            if isinstance(dim, (tuple, list)):
                result = self
                for d in sorted(dim, reverse=True):
                    result = result.prod(d, keepdim=True)
                if not keepdim:
                    result.shape = result.shape.reduce(dim, keepdim)
                return result
            
            data = cuda.reductions.prod(self, dim, initial)
        else: data = cpu.reductions.prod(self, dim, keepdim, initial)
        
        output_shape = self.shape.reduce(dim, keepdim)
        out = Tensor._new(data, output_shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
            
        def _backward() -> None:
            if self_requires_grad:
                out_expanded = out if keepdim else out.unsqueeze(dim) \
                    if dim is not None else out.reshape([1] * self.ndim)
                grad_expanded = out.grad if keepdim else \
                    out.grad.unsqueeze(dim) if dim is not None else \
                    out.grad.reshape([1] * self.ndim)
                
                out_full = out_expanded.expand(self.shape)
                grad_full = grad_expanded.expand(self.shape)

                sign = (self >= 0.0).to(self.device, self.dtype) \
                     - (self <  0.0).to(self.device, self.dtype)
                safe_self = self.abs().clamp(min_value=1e-7) * sign
                self.grad += (out_full / safe_self) * grad_full
        
        out._backward = _backward
        return out

    def std(
        self: Tensor, 
        dim: builtins.int | tuple[builtins.int, ...] | None = None, 
        keepdim: bool = False, 
        correction: builtins.int = 1
    ) -> Tensor:
        mean = self.mean(dim=dim, keepdim=True)
        variance = ((self - mean) ** 2).mean(dim=dim, keepdim=keepdim)
        if correction == 1:
            n = self.size if dim is None else self.shape[dim]
            variance = variance * n / (n - 1)
        return variance.sqrt()
    
    @amp_float32
    def norm(
        self: Tensor,
        p: Literal['fro', 'l1', 'inf', '-inf', 'l0', 'lp'] = 'fro',
        dim: builtins.int | tuple[builtins.int, ...] | None = None,
        keepdim: bool = False
    ) -> Tensor:        
        match p:
            case 'fro': # L2/Frobenius norm
                return (self ** 2).sum(dim=dim, keepdim=keepdim).sqrt()
            case 'l1': # L1 norm
                return self.abs().sum(dim=dim, keepdim=keepdim)
            case 'inf': # L-(-inf) norm
                return self.abs().amax(dim=dim, keepdim=keepdim)
            case '-inf': # L-inf norm
                return self.abs().amin(dim=dim, keepdim=keepdim)
            case 'l0':  # L0 norm
                out = (self != 0).to(self.device, self.dtype)
                return out.sum(dim=dim, keepdim=keepdim)
            case _: # general Lp norm
                return (self.abs()**p).sum(dim=dim, keepdim=keepdim) ** (1.0/p)

    ### SORTING ###
    
    def sort(
        self: Tensor,
        dim: builtins.int = -1,
        descending: bool = False
    ) -> tuple[Tensor, Tensor]:
        self_requires_grad = self.requires_grad
        dim = dim if dim >= 0 else self.ndim + dim
        
        if self.device == 'cuda':
            out_data, indices = cuda.sorting.sort(self, dim, descending)
        else: out_data, indices = cpu.sorting.sort(self, dim, descending)
        
        indices = Tensor._new(indices, self.shape, typing.int32, self.device)
        values  = Tensor._new(out_data, self.shape, self.dtype, self.device,
            requires_grad=self_requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                grad_input = Tensor._new(
                    np.zeros(self.grad.shape, self.grad.dtype),
                    self.grad.shape, self.grad.dtype, self.grad.device)
                grad_input = grad_input.scatter(dim, indices, values.grad)
                self.grad += grad_input
        
        values._backward = _backward
        return values, indices
    
    def argsort(
        self: Tensor,
        dim: builtins.int = -1,
        descending: bool = False
    ) -> Tensor:
        _, indices = self.sort(dim=dim, descending=descending)
        return indices
        
    ### INDEXING ###
    
    def scatter_add(
        self:   Tensor, 
        dim:    builtins.int, 
        index:  Tensor, 
        source: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        assert self.shape == index.shape, \
            f'Shape of index tensor must match shape of input tensor.'
        
        if not isinstance(source, tensor):
            source = Tensor._new(
                np.full(index.shape, fill_value=source, dtype=self.dtype), 
                index.shape, dtype=self.dtype, device=self.device)
            
        assert self.dtype == source.dtype, \
            f'Input and source must have the same dtype.'
        
        if not self.device == index.device or not self.device == source.device:
            _devices = set([self.device, index.device, source.device])
            raise ValueError(
                f'Scatter expects all tensors to be on same device, but found '
                f'multiple devices: {list(_devices)}')
        
        if index.dtype != typing.int32:
            index = index.to(index.device, dtype=typing.int32)

        self_requires_grad = self.requires_grad
        source_requires_grad = source.requires_grad
        
        if self.device == 'cuda':
            out_data = cuda.indexing.scatter_add(self, index, dim, source)
        else: out_data = cpu.indexing.scatter_add(
            self.data, index.data, dim, source.data)
        out = Tensor._new(out_data, self.shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad: self.grad += out.grad
            if source_requires_grad: source.grad += out.grad.gather(dim, index)
        
        out._backward = _backward
        return out

    def masked_fill(
        self:  Tensor, 
        mask:  Tensor, 
        value: builtins.float
    ) -> Tensor:
        x = self * 0 + value
        return x * mask + self * (1 - mask)



