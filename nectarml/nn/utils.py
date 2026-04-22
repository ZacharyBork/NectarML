from __future__ import annotations

import pickle
import tarfile

from os      import PathLike
from pathlib import Path
from typing  import Any
from collections.abc import Callable

from nectarml.tensor          import Tensor
from nectarml.nn.module       import Module
from nectarml.optim.optimizer import Optimizer
from nectarml.utils.save      import _save_tarfile, _load_tarfile

### GRAD UTILS ###

def clip_grad_norm(params: list[Tensor], max_norm: float = 1.0) -> float:
    total_sq = sum(
        param.grad.norm(p='fro').item()**2
        for param in params if param.grad is not None)
    total_norm = total_sq ** 0.5
    
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for param in params:
            if param.grad is not None:
                param.grad *= scale
    
    return total_norm

### Utility Modules ###

class Lambda(Module):
    def __init__(
        self: Lambda,
        func: Callable[[Any], Any]
    ) -> None:
        super().__init__()
        self.func = func
        
    def forward(
        self:     Lambda, 
        *args:    list[Any], 
        **kwargs: dict[str, Any]
    ) -> Tensor:
        return self.func(*args, **kwargs)
    

### CHECKPOINTING ###

class checkpoint:
    def __init__(
        self:      checkpoint, 
        model:     Module,
        optimizer: Optimizer | None = None
    ) -> None:
        self.model     = model
        self.optimizer = optimizer
    
    ### SAVING ###
    
    def _serialize_parameters(self: checkpoint) -> None:
        for name, param in self.model.list_parameters():
            self._model_state[name] = {
                'data':  param.numpy(),
                'dtype': param.dtype,
                'shape': param.shape
            }
    
    def _serialize_buffers(self: checkpoint) -> None:
        for module_name, module in self.model._walk_module_tree():
            for buffer_name, buffer in module._buffers.items():
                if buffer_name in module._persistent_buffers:
                    full_name = f'{module_name}.{buffer_name}' \
                                if module_name else buffer_name
                    
                    self._model_state[full_name] = {
                        'data':  buffer.cpu().numpy(),
                        'dtype': buffer.dtype,
                        'shape': buffer.shape,
                        'is_buffer': True 
                    }
    
    def _serialize_optimizer_state(self: checkpoint) -> None:
        if self.optimizer is not None:
            self._optimizer_state = { 'param_groups': [], 'state': {} }
            
            for group in self.optimizer.param_groups:
                self._optimizer_state['param_groups'].append({
                    k: v for k, v in group.items()
                    if k not in ('params',)
                })
                
            for idx, state in self.optimizer.state.items():
                self._optimizer_state['state'][idx] = {}
                for k, v in state.items():
                    if isinstance(v, Tensor):
                        self._optimizer_state['state'][idx][k] = {
                            'data':  v.numpy(),
                            'dtype': v.dtype,
                            'shape': v.shape
                        }
                    else: self._optimizer_state['state'][idx][k] = v
    
    def save(
        self:      checkpoint,
        path:      PathLike,
        epoch:     int  = 0,
        iteration: int  = 0,
        metadata:  dict = None,
        overwrite: bool = False
    ) -> None:
        self.checkpoint_path = Path(path).resolve()
        if not overwrite and self.checkpoint_path.exists():
            raise FileExistsError(
                f'File exists at {self.checkpoint_path}. Use overwrite=True.')

        suffixes = [suffix.lower() for suffix in self.checkpoint_path.suffixes]
        assert suffixes[0] in ['.pt', '.pth'], \
            f'save() requires output to be of type ".pt" or ".pth", not ' \
            f'[{self.checkpoint_path.suffix}]'

        self._model_state     = {}
        self._optimizer_state = None
        
        self._serialize_parameters()
        self._serialize_buffers()
        self._serialize_optimizer_state()

        serialized = {
            'model_state': self._model_state,
            'opt_state':   self._optimizer_state,
            'epoch':       epoch,
            'iteration':   iteration,
            'metadata':    metadata or {}
        }

        if len(suffixes) == 1:
            with open(self.checkpoint_path, 'wb') as file:
                pickle.dump(serialized, file, pickle.HIGHEST_PROTOCOL)
        elif '.tar' in suffixes: 
            _save_tarfile(serialized, self.checkpoint_path)
        else: raise ValueError(
            f'Unable to save checkpoint data with file suffixes: {suffixes}')

    ### LOADING ###

    def _build_lookups(self: checkpoint) -> None:
        self._param_lookup  = {}
        self._buffer_lookup = {}
        
        for module_name, module in self.model._walk_module_tree():
            for param_name in module._parameters:
                full_name = f'{module_name}.{param_name}' if module_name \
                            else param_name
                self._param_lookup[full_name] = (module, param_name)
            
            for buffer_name in module._buffers:
                if buffer_name not in module._persistent_buffers: continue
                full_name = f'{module_name}.{buffer_name}' if module_name \
                            else buffer_name
                self._buffer_lookup[full_name] = (module, buffer_name)

    def _restore_parameters(
        self:  checkpoint, 
        name:  str, 
        saved: dict[str, Any]
    ) -> None:
        if saved.get('is_buffer'): return
        if name not in self._param_lookup:
            raise KeyError(f'Parameter {name} not found in model')
        
        module, param_name = self._param_lookup[name]
        data     = saved['data'].astype(saved['dtype'].numpy)
        restored = Tensor._new(
            data, saved['shape'], saved['dtype'], 'cpu', True)
        restored._prev.clear()
        restored._backward = lambda: None
        module._parameters[param_name] = restored.to(self._target_device)

    def _restore_buffers(
        self:  checkpoint,
        name:  str,
        saved: dict[str, Any]
    ) -> None:
        if not saved.get('is_buffer'):      return
        if name not in self._buffer_lookup: return
        
        module, buffer_name = self._buffer_lookup[name]
        data     = saved['data'].astype(saved['dtype'].numpy)
        restored = Tensor._new(data, saved['shape'], saved['dtype'], 'cpu')
        restored._prev.clear()
        restored._backward = lambda: None
        module._buffers[buffer_name] = restored.to(self._target_device)

    def _restore_optimizer_state(
        self:      checkpoint, 
        opt_state: dict[str, Any]
    ) -> None:
        for idx, state in opt_state['state'].items():
            self.optimizer.state[int(idx)] = {}
            for k, v in state.items():
                if isinstance(v, dict) and 'data' in v:
                    data     = v['data'].astype(v['dtype'].numpy)
                    restored = Tensor._new(
                        data, v['shape'], v['dtype'], 'cpu',
                        requires_grad=False)
                    restored = restored.to(self._target_device)
                    restored._prev.clear()
                    restored._backward = lambda: None
                    restored._requires_grad = False
                    self.optimizer.state[int(idx)][k] = restored
                else: self.optimizer.state[int(idx)][k] = v

    def load(
        self: checkpoint,
        path: PathLike
    ) -> dict[str, Any]:
        self.checkpoint_path = Path(path).resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f'Unable to locate checkpoint file at: {self.checkpoint_path}')
            
        if not tarfile.is_tarfile(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as file:
                loaded = pickle.load(file)
        else:   loaded = _load_tarfile(self.checkpoint_path)

        self._target_device = 'cpu'
        for _, p in self.model.list_parameters():
            self._target_device = p.device
            break

        model_state = loaded['model_state']
        self._build_lookups()

        for name, saved in model_state.items():
            self._restore_parameters(name, saved)
            self._restore_buffers(name, saved)
            
        if self.optimizer is not None and loaded.get('opt_state'):
            opt_state = loaded['opt_state']
            self._restore_optimizer_state(opt_state)

        return {
            'epoch':     loaded['epoch'],
            'iteration': loaded['iteration'],
            'metadata':  loaded['metadata']
        }

