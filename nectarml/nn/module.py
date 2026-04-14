from __future__ import annotations

from typing import Any, Literal, Self

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32

class Module():
    _submodules:           dict[str, Module]
    _parameters:           dict[str, Tensor]
    _buffers:              dict[str, Tensor]
    _pinned_buffer_dtypes: dict[str, DTypeLike]
    
    def __init__(
        self: Module,
        dtype: DTypeLike = float32
    ) -> None:
        super().__setattr__('_submodules', {})
        super().__setattr__('_parameters', {})
        super().__setattr__('_buffers', {})
        super().__setattr__('_pinned_buffer_dtypes', {})
        
        self.dtype: DTypeLike = dtype
        self.training:   bool = True
        
        self._device_id:        int | None = None
        self._persistent_buffers: set[str] = set()
                
    # PROPERTIES
    
    @property
    def dtype(self: Module) -> DTypeLike:
        return self._dtype
    
    @dtype.setter
    def dtype(self: Module, value: DTypeLike) -> None:
        self._dtype = value
            
    # REGISTRATION
    
    def register_parameter(self: Module, name: str, tensor: Tensor) -> None:
        self._parameters[name] = tensor
    
    def register_submodule(self: Module, name: str, module: Module) -> None:
        self._submodules[name] = module
        
    def register_buffer(
        self,
        name: str,
        tensor: Tensor,
        persistent: bool = True,
        pin_dtype: DTypeLike | None = None
    ) -> None:
        self._buffers[name] = tensor
        if persistent: self._persistent_buffers.add(name)
        if pin_dtype is not None:
            self._pinned_buffer_dtypes[name] = pin_dtype
    
    # GETTERS / SETTERS
    
    def __setattr__(self: Module, name: str, value: Any) -> None:
        if '_buffers' in self.__dict__ and name in self._buffers:
            self._buffers[name] = value
        if isinstance(value, Module):
            self.register_submodule(name, value)
        elif isinstance(value, Tensor):
            self.register_parameter(name, value)
        else: super().__setattr__(name, value)
        
    def __getattr__(self: Module, name: str) -> Any:
        if '_buffers' in self.__dict__ and name in self._buffers:
            return self._buffers[name]
        if '_parameters' in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        if '_submodules' in self.__dict__ and name in self._submodules:
            return self._submodules[name]
        else: raise AttributeError(
                f'{type(self).__name__} has no attribute "{name}"')
        
    # SUBMODULES
    
    def _walk_module_tree(self: Module) -> list[tuple[str, Module]]:
        result = [('', self)]
        for name, module in self._submodules.items():
            for subname, submodule in module._walk_module_tree():
                full_name = f'{name}.{subname}' if subname else name
                result.append((full_name, submodule))
        return result
        
    # GRADIENTS
    
    def zero_grad(self: Module) -> None:
        for _, module in self._walk_module_tree():
            for name, buffer in module._buffers.items():
                if not buffer.requires_grad: continue
                buffer.zero_grad()
                if buffer.grad is not None:
                    buffer.grad._prev.clear()
                
            for parameter in module._parameters.values():
                if not parameter.requires_grad: continue
                parameter.zero_grad()
                if parameter.grad is not None:
                    parameter.grad._prev.clear()
    
    # DEVICE / DTYPE
    
    def to(
        self: Module,
        device: Literal['cpu', 'cuda'] | None = None,
        dtype: DTypeLike | None = None
    ) -> Self:
        for _, module in self._walk_module_tree():
            for name, buffer in module._buffers.items():
                pinned = module._pinned_buffer_dtypes.get(name)
                target_dtype = pinned if pinned is not None \
                               and dtype is not None \
                               else dtype or buffer.dtype
                module._buffers[name] = buffer.to(
                    device or buffer.device, target_dtype)
            
            for name, param in module._parameters.items():
                moved = param.to(device or param.device, dtype or param.dtype)
                moved._prev.clear()
                moved._backward = lambda: None
                module._parameters[name] = moved            
        
        if dtype is not None: self.dtype = dtype
        return self
        
    def cuda(self: Module) -> Self:
        '''Convenience function to cast given Module to CUDA device.
        
        When called on a Module who's device is already "cuda", this method 
        will return a reference to the original Module object.
        
        Returns:
            Module : The resulting CUDA Module from the cast operation.
        '''
        return self.to(device='cuda')

    def cpu(self: Module) -> Self:
        '''Convenience function to cast given Module to CPU device.
        
        When called on a Module who's device is already "cpu", this method 
        will return a reference to the original Module object.
        
        Returns:
            Module : The resulting CPU Module from the cast operation.
        '''
        return self.to(device='cpu')
    
    # STATES
    
    def train(self: Module) -> None: 
        modules = self._walk_module_tree()
        for _, module in modules:
            module.training = True
    
    def eval(self: Module) -> None:
        modules = self._walk_module_tree()
        for _, module in modules:
            module.training = False
        
    # INSPECTION
    
    def list_parameters(self: Module) -> list[tuple[str, Tensor]]:
        output = []
        modules = self._walk_module_tree()
        for module_name, module in modules:
            for parameter_name, parameter in module._parameters.items():
                name = f'{module_name}.{parameter_name}' if module_name \
                    else parameter_name
                output.append((name, parameter))
        return output
    
    def parameters(self: Module) -> dict[str, Any]:
        output = []
        seen = set()

        for module_name, module in self._walk_module_tree():
            params = { 'params': [], 'param_names': [] }
            
            for parameter_name, parameter in module._parameters.items():
                if id(parameter) not in seen:
                    name = f'{module_name}.{parameter_name}' if module_name \
                        else parameter_name
                    params['params'].append(parameter)
                    params['param_names'].append(name)
                    seen.add(id(parameter))
                
            output.append(params)
        
        return output
    
    def list_submodules(self: Module) -> list[tuple[str, Module]]:
        return [(name, module) for name, module in self._walk_module_tree()]
    
    def __repr__(self, indent: int = 0) -> str:
        pad = '  ' * indent
        child_pad = '  ' * (indent + 1)
        
        lines = [f'{type(self).__name__}(']
        
        for name, module in self._submodules.items():
            child_repr = module.__repr__(indent + 1)
            lines.append(f'{child_pad}({name}): {child_repr}')
        
        for name, param in self._parameters.items():
            parameter_line = (
                f'{child_pad}({name}): '
                f'Tensor(shape={param.shape}, dtype={param.dtype})')
            lines.append(parameter_line)
        
        lines.append(f'{pad})')
        
        return '\n'.join(lines)
        
    def __hash__(self: Module) -> int: return id(self)
    
    # FORWARD
    
    def forward(self: Module, *args, **kwargs) -> Any: 
        raise NotImplementedError('forward() not implemented by child class.')

    def __call__(self: Module, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)



