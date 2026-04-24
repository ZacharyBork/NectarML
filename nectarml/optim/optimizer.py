from __future__ import annotations

from typing          import Any
from collections.abc import Callable
from dataclasses     import dataclass

from nectarml.core import Tensor

@dataclass
class HookHandle:
    hook_list: list[Callable]
    hook:      Callable
    
    def remove(self: HookHandle) -> None:
        '''Removes the hook from the optimizer it is registered with.'''
        self.hook_list.remove(self.hook)
    
class Optimizer:
    _defaults: dict[str, Any] = None
    
    def __init__(
        self: Optimizer,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        defaults: dict[str, Any] | None = None
    ) -> None:
        super().__setattr__('_defaults', defaults or {})
        self.state:   dict[int, dict[str, Any]] = {}
        self.param_groups: list[dict[str, Any]] = []
        
        self._add_init_parameters(parameters)
        for idx, _ in enumerate(self.params()): self.state[idx] = {}
        self._param_to_idx = {
            id(p): idx for idx, p in enumerate(self.params())}
        
        for group in self.param_groups:
            for key, value in self._defaults.items():
                if key not in group: group[key] = value
                
        self.state_dict_pre_hooks:       list[Callable] = []
        self.state_dict_post_hooks:      list[Callable] = []
        
        self.load_state_dict_pre_hooks:  list[Callable] = []
        self.load_state_dict_post_hooks: list[Callable] = []
        
        self.step_pre_hooks:             list[Callable] = []
        self.step_post_hooks:            list[Callable] = []
    
    ### PARAM GROUPS ###
    
    def _add_init_parameters(self: Optimizer, parameters: Any) -> None:
        try:
            if isinstance(parameters[0], Tensor):
                self.param_groups.append({ 'params': parameters })
            elif isinstance(parameters[0], tuple):
                _group = { 'params': [], 'param_names': [] }
                for param in parameters:
                    assert len(param) == 2 \
                       and isinstance(param[0], str) \
                       and isinstance(param[1], Tensor), (
                        'Parameter tuples must have a string at index 0 '
                        'and a Tensor at index 1 to be valid.')
                    _group['params'].append(param[1])
                    _group['param_names'].append(param[0])
                self.param_groups.append(_group)
            elif isinstance(parameters[0], dict):
                
                for group in parameters:
                    assert 'params' in group, \
                        'Parameter dicts must contain "params" key.'
                self.param_groups = parameters
            else: raise ValueError(
                f'Unable to initialize optimizer parameters from type '
                f'list[{type(parameters[0])}]')
        except KeyError:
            raise ValueError(
                'Optimizer parameters must be either list[nectarml.Tensor], '
                'list[tuple[str, Tensor]], or list[dict[str, Any]].')
    
    def _get_parameter_state_index(self: Optimizer, parameter: Tensor) -> int:
        return self._param_to_idx[id(parameter)]
    
    def add_param_group(self: Optimizer, param_group: dict[str, Any]) -> None:
        assert 'params' in param_group, \
            'param_group must contain "params" key.'
        
        if self._defaults is not None:
            for key, value in self._defaults.items():
                if key not in param_group: param_group[key] = value
            
        self.param_groups.append(param_group)
        start_idx = len(self._param_to_idx)
        for i, param in enumerate(param_group['params']):
            idx = start_idx + i
            self.state[idx] = {}
            self._param_to_idx[id(param)] = idx
        
    def params(self: Optimizer) -> list[Tensor]:
        '''Returns a flat list of all parameters managed by the optimizer.'''
        return [
            p for group in self.param_groups 
            for p in group['params']]
    
    def print_param_stats(self: Optimizer) -> None:
        '''Prints debug information about optimizer's parameters.'''
        for group in self.param_groups:
            for name, param in zip(group['param_names'], group['params']):
                if param.grad is not None:

                    grad_max  = param.grad.abs().max().item()
                    grad_mean = param.grad.abs().mean().item()
                    param_max = param.abs().max().item()
                    
                    print(f'{name:60s} | '
                        f'param_max: {param_max:10.4f} | '
                        f'grad_max: {grad_max:10.4f} | '
                        f'grad_mean: {grad_mean:10.6f}')
                else:
                    print(f'{name:60s} | no grad')
    
    ### HOOKS ###
    
    def hooks(self: Optimizer) -> dict[str, list[Callable]]:
        '''Returns a dict of all hooks registered to the optimizer.
        
        Returns:
            dict[str, list[Callable]] : All of the hooks registered to the
                optimizer, broken down by category.
        '''
        return {
            'state_dict': {
                'pre_hooks':  self.state_dict_pre_hooks,
                'post_hooks': self.state_dict_post_hooks
            },
            'load_state_dict': {
                'pre_hooks':  self.load_state_dict_pre_hooks,
                'post_hooks': self.load_state_dict_post_hooks
            },
            'step': {
                'pre_hooks':  self.step_pre_hooks,
                'post_hooks': self.step_post_hooks
            }
        }
    
    ### STATE DICT ###
    
    def register_state_dict_pre_hook(
        self:    Optimizer, 
        hook:    Callable,
        prepend: bool = False
    ) -> HookHandle:
        hooks = self.state_dict_pre_hooks
        if not prepend: hooks.append(hook)
        else: hooks.insert(0, hook)
        return HookHandle(hooks, hook)
    
    def register_state_dict_post_hook(
        self:    Optimizer,
        hook:    Callable,
        prepend: bool = False
    ) -> HookHandle:
        hooks = self.state_dict_post_hooks
        if not prepend: hooks.append(hook)
        else: hooks.insert(0, hook)
        return HookHandle(hooks, hook)
    
    def _build_state_dict(self: Optimizer) -> dict[str, Any]:
        param_groups = []
        idx = 0
        for group in self.param_groups:
            _group = {}
            for key, value in group.items():
                if key == 'params':
                    _group['params'] = list(range(idx, idx + len(value)))
                    idx += len(value)
                else: _group[key] = value
            param_groups.append(_group)

        return {
            'state': self.state.copy(),
            'param_groups': param_groups
        }
    
    def state_dict(self: Optimizer) -> dict[str, Any]:
        for hook in self.state_dict_pre_hooks: hook(self)
        result = self._build_state_dict()
        for hook in self.state_dict_post_hooks: hook(self, result)
        return result
    
    ### LOAD STATE DICT ###
    
    def register_load_state_dict_pre_hook(
        self:    Optimizer, 
        hook:    Callable,
        prepend: bool = False
    ) -> HookHandle:
        hooks = self.load_state_dict_pre_hooks
        if not prepend: hooks.append(hook)
        else: hooks.insert(0, hook)
        return HookHandle(hooks, hook)
    
    def register_load_state_dict_post_hook(
        self:    Optimizer, 
        hook:    Callable,
        prepend: bool = False
    ) -> HookHandle:
        hooks = self.load_state_dict_post_hooks
        if not prepend: hooks.append(hook)
        else: hooks.insert(0, hook)
        return HookHandle(hooks, hook)
    
    def load_state_dict(self: Optimizer, state_dict: dict[str, Any]) -> None:
        for hook in self.load_state_dict_pre_hooks: hook(self, state_dict)
        
        assert 'state' in state_dict, \
            'Unable to locate "state" in input state_dict.'
        assert 'param_groups' in state_dict, \
            'Unable to locate "param_groups" in input state_dict.'
            
        load_states = state_dict['state']
        load_param_groups: list[dict[str, Any]] = state_dict['param_groups']
        
        assert len(load_param_groups) == len(self.param_groups), (
            f'Number of param groups [{len(load_param_groups)}] in state_dict '
            f'does not match expected value: [{len(self.param_groups)}]')
        
        for group, load_group in zip(self.param_groups, load_param_groups):
            assert len(group['params']) == len(load_group['params']), \
                f'Number of parameters in group does not match state_dict.'
            for key, value in load_group.items():
                if key == 'params': continue
                assert key in group, \
                    f'Found unmatched key in loaded parameter group: {key}'
                group[key] = value
        
        for idx, _ in enumerate(self.params()):
            if idx in load_states: self.state[idx] = load_states[idx]
        
        for hook in self.load_state_dict_post_hooks: hook(self)

    ### GRADIENTS ###

    def zero_grad(self: Optimizer) -> None:
        params = [p for p in self.params() if p.grad is not None]
        for param in params: param.zero_grad()
    
    ### STEP ###
    
    def register_step_pre_hook(
        self:    Optimizer, 
        hook:    Callable,
        prepend: bool = False
    ) -> HookHandle:
        hooks = self.step_pre_hooks
        if not prepend: hooks.append(hook)
        else: hooks.insert(0, hook)
        return HookHandle(hooks, hook)
    
    def register_step_post_hook(
        self:    Optimizer, 
        hook:    Callable,
        prepend: bool = False
    ) -> HookHandle:
        hooks = self.step_post_hooks
        if not prepend: hooks.append(hook)
        else: hooks.insert(0, hook)
        return HookHandle(hooks, hook)
        
    def _update(self: Optimizer) -> None:
        raise NotImplementedError
    
    def step(self: Optimizer) -> None:
        for hook in self.step_pre_hooks: hook(self)
        self._update()
        for hook in self.step_post_hooks: hook(self)


