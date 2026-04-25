from __future__ import annotations

from enum   import Enum
from typing import Any, Self

from nectarml      import typing
from nectarml.core import Tensor

class ModuleState(Enum):
    TRAINING = 0
    EVAL     = 1

class Module:
    _class_type_nectar_module = True
    _module_state             = ModuleState.TRAINING
    
    _submodules:           dict[str, Module]
    _parameters:           dict[str, Tensor]
    _buffers:              dict[str, Tensor]
    _pinned_buffer_dtypes: dict[str, typing.dtype]
    _persistent_buffers:    set[str]
    
    def __init__(self: Module) -> None:
        super().__setattr__('_submodules',            {})
        super().__setattr__('_parameters',            {})
        super().__setattr__('_buffers',               {})
        super().__setattr__('_pinned_buffer_dtypes',  {})
        super().__setattr__('_persistent_buffers', set())
                        
    # PROPERTIES
    
    @property
    def state(self: Module) -> ModuleState:
        '''Module state property access.
        
        Returns:
            ModuleState : Returns the module's current state.
        '''
        return self._module_state
    
    @property
    def training(self: Module) -> bool:
        '''Module training attribute property access.
        
        Returns:
            bool : True if the module's state is ModuleState.TRAINING, 
                otherwise False.
        '''
        return self._module_state == ModuleState.TRAINING
            
    # REGISTRATION
    
    def register_parameter(self: Module, name: str, tensor: Tensor) -> None:
        '''Registers a parameter with the module.
        
        Args:
            name   : The name to assign to the new parameter.
            tensor : The parameter tensor to register.
        '''
        self._parameters[name] = tensor
    
    def register_submodule(self: Module, name: str, module: Module) -> None:
        '''Registers a submodule with the module.
        
        Args:
            name   : The name to assign to the new submodule.
            tensor : The module to to register as a submodule.
        '''
        self._submodules[name] = module
        
    def register_buffer(
        self:       Module,
        name:       str,
        tensor:     Tensor,
        persistent: bool = True,
        pin_dtype:  typing.dtype | None = None
    ) -> None:
        '''Registers a buffer with the module.
        
        Args:
            name       : The name to assign to the new buffer.
            tensor     : The buffer tensor to register.
            persistent : If True, the buffer will be saved and loaded as part
                of the model state when the model is checkpointed. If False, 
                it will be ignored.
            pin_dtype  : If provided, when the model owning the buffer is cast
                to a new dtype, this buffer will ignore the cast, always 
                remaining as the given dtype.
        '''
        self._buffers[name] = tensor
        if persistent: self._persistent_buffers.add(name)
        if pin_dtype is not None:
            self._pinned_buffer_dtypes[name] = pin_dtype
    
    # GETTERS / SETTERS
    
    def __setattr__(self: Module, name: str, value: Any) -> None:
        '''Processes attribute set ops to assign values to correct locations.
        
        If the given name is in the modules buffer dict, the value will 
        overwrite the entry in the dict, then return early to avoid double
        registration (parameter + buffer).
        
        If the value is a Module, it will be added to this module's submodules.
        
        If the value is a Tensor (and the name is not in the buffers dict),
        the provided tensor will be added to the module's parameters dict, 
        using the provided name as a key.
        
        If the value is none of these things, it will be treated as a normal
        attribute.
        
        Args:
            name  : The name of the attribute to set.
            value : The value to set the attribute to.
        
        '''
        if '_buffers' in self.__dict__ and name in self._buffers:
            self._buffers[name] = value
            return
        
        if   isinstance(value, Module): self.register_submodule(name, value)
        elif isinstance(value, Tensor): self.register_parameter(name, value)
        else: super().__setattr__(name, value)
        
    def __getattr__(self: Module, name: str) -> Any:
        '''Gets attributes from module by name.
        
        This function retrives attributes from three main locations on the
        module:
        
            1. The buffer dictionary.
            2. The parameter dictionary.
            3. The submodule dictionary.
            
        If the provided name is not found in one of those three locations, an
        exception will be raised.
        
        Args:
            name : The name of the object to retrieve.
            
        Raises:
            AttributeError : If the given name is not found in the module's
                buffers, parameters, or submodules.
        '''
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
        '''Walks the module tree of the given module recursively.'''
        result = [('', self)]
        for name, module in self._submodules.items():
            for subname, submodule in module._walk_module_tree():
                full_name = f'{name}.{subname}' if subname else name
                result.append((full_name, submodule))
        return result
        
    # GRADIENTS
    
    def zero_grad(self: Module) -> None:
        '''Zeros gradients of all parameters and buffers of the module.
        
        The function walks all submodules of the given module recursively,
        zeroing the gradients of all of their parameters and buffers as well.
        '''
        for _, module in self._walk_module_tree():
            for _, buffer in module._buffers.items():
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
        self:   Module,
        device: typing.DeviceLikeType | None = None,
        dtype:  typing.dtype | None = None
    ) -> Self:
        '''Casts the module to a given device/dtype.
        
        The function walks all submodules of the given module recursively,
        casting all parameters and buffers of each module to the provided
        dtype and/or device.
        
        The only exception to this is buffers which have pinned dtypes. They
        will be moved to the given device, if provided, but their dtype will
        remain as their pinned value.
        
        Args:
            device : The device to move the module to. Can be either a string
                ['cpu', 'cuda'], a nectarml.typing.device, or None to leave
                the module's device unchanged.
            dtype : The dtype to cast the module to, or None to leave the 
                module's dtype unchanged.
        
        Returns:
            Self : The resulting module from the cast operation.
        '''
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
        
        if dtype  is not None: self._dtype  = dtype
        if device is not None: self._device = typing.device(device)
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
    
    def float(self: Module) -> Self:
        '''Convenience function to cast given Module's dtype to float32.
        
        When called on a Module who's dtype is already "float32", this method 
        will return a reference to the original Module object.
        
        Returns:
            Module : The resulting Module from the cast operation.
        '''
        return self.to(dtype=typing.float32)
    
    def half(self: Module) -> Self:
        '''Convenience function to cast given Module's dtype to float16.
        
        When called on a Module who's dtype is already "float16", this method 
        will return a reference to the original Module object.
        
        Returns:
            Module : The resulting Module from the cast operation.
        '''
        return self.to(dtype=typing.float16)
    
    # STATES
    
    def train(self: Module) -> None:
        '''Enables training mode on the module when called.'''
        modules = self._walk_module_tree()
        for _, module in modules:
            module._module_state = ModuleState.TRAINING
    
    def eval(self: Module) -> None:
        '''Enables eval mode on the module when called.'''
        modules = self._walk_module_tree()
        for _, module in modules:
            module._module_state = ModuleState.EVAL
        
    # INSPECTION
    
    def parameters(self: Module, recurse: bool = True) -> list[Tensor]:
        '''Returns a list of all parameters associated with the module.
        
        Args:
            recurse : If True, the function will collect all of the parameters 
                from this module, and all of its submodules recursively. If
                False, only the parameters of the module from which this 
                function was called will be returned.
                
        Returns:
            list[Tensor] : A list containing all the parameter tensors.
        '''
        output = []
        seen   = set()
        valid  = self._walk_module_tree() if recurse else [None, self]

        for _, module in valid:
            for _, parameter in module._parameters.items():
                if id(parameter) not in seen:
                    output.append(parameter)
                    seen.add(id(parameter))

        return output
    
    def named_parameters(
        self:              Module, 
        prefix:            str  = '',
        recurse:           bool = True,
        remove_duplicates: bool = True
    ) -> list[tuple[str, Tensor]]:
        '''Returns a list of all of the Modules parameters, and their names.
        
        Args:
            prefix : And optional prefix to append to the parameter names. The
                resulting name will have the format:
                
                    - {prefix}.parameter_name
        
            recurse : If True, the function will collect all of the parameters 
                from this module, and all of its submodules recursively. If
                False, only the parameters of the module from which this 
                function was called will be returned.
            
            remove_duplicates : If True, any duplicate parameters will be
                removed from the returned list. If False, duplicates will be
                left in.
                
        Returns:
            list[tuple[str, Tensor]] : A list containing tuples whos first item
                is the parameter name as a string, and the second is the 
                parameter tensor itself.
        '''
        output = []
        seen   = set()
        valid  = self._walk_module_tree() if recurse else [self]
        prefix = f'{prefix}.' if prefix != '' else prefix

        for module_name, module in valid:            
            for parameter_name, parameter in module._parameters.items():
                if remove_duplicates and id(parameter) in seen: continue
            
                name = f'{prefix}.{module_name}.{parameter_name}' \
                    if module_name else f'{prefix}{parameter_name}'
                output.append((name, parameter))
                seen.add(id(parameter))
                    
        return output
    
    def num_parameters(self: Module, recurse: bool = True) -> int:
        '''Returns the total parameter count of the module.
        
        Args:
            recurse : If True, the function will count all of the parameters 
                from this module, and all of its submodules recursively. If
                False, only the parameters of the module from which this 
                function was called will be counted.
        
        Returns:
            int : The total parameter count of the module.
        '''
        return sum([i.numel() for i in self.parameters(recurse)])
    
    def list_submodules(self: Module) -> list[tuple[str, Module]]:
        '''Returns a list of all submodules of the given module.
        
        Returns:
            list[tuple[str, Module]] : A list containing tuples where the first
                item is the submodule's name as a string, and the second is
                the module object itself. 
        '''
        return [(name, module) for name, module in self._walk_module_tree()]
    
    def __repr__(self, indent: int = 0) -> str:
        pad       = '  ' * indent
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



