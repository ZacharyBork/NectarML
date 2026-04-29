from __future__ import annotations

import pickle
import tarfile

from os              import PathLike
from pathlib         import Path
from typing          import Any, Self
from collections.abc import Callable

from nectarml.core            import Tensor
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
    
### EMA ###

class EMA:
    def __init__(
        self:  EMA, 
        model: Module, 
        decay: float = 0.999
    ) -> None:
        '''
        Not currently functional!!
        '''
        raise NotImplementedError
        self.model  = model
        self.decay  = decay
        self.shadow = {}
        
        for name, param in model.named_parameters():
            self.shadow[name] = param.detach().clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = (
                self.decay * self.shadow[name] + 
                (1 - self.decay) * param.detach())
    
    def apply(self):
        self._backup = {}
        for name, param in self.model.named_parameters():
            self._backup[name] = param.detach().clone()
            
    
    def restore(self):
        pass


### CHECKPOINTING ###

class checkpoint:
    def __init__(
        self:      checkpoint, 
        model:     Module,
        optimizer: Optimizer | None = None
    ) -> None:
        '''Utility class to save and load model checkpoints.
        
        Args:
            model     : The nn.Module to save/load a checkpoint for.
            optimizer : The model's optimizer, if applicable.
        '''
        self.model     = model
        self.optimizer = optimizer
    
    ### SAVING ###
    
    @classmethod
    def _serialize_tensor(
        cls:       type[Self],
        tensor:    Tensor, 
        is_buffer: bool = False
    ) -> dict[str, Any]:
        '''Serializes a tensor in a pickleable format for saving.
        
        Data which gets serialized is:
        - The tensor's data (as a numpy array)
        - The tensor's DType
        - The tensor's shape
        
        Args:
            tensor    : The tensor to serialize.
            is_buffer : Set true if the tensor is a buffer, otherwise False.
                        Adds a flag to the the tensor in the checkpoint file
                        which is needed for for loading.
        Returns:
            dict[str, Any] : The tensor, serialized to a pickleable format.
        '''
        return {
            'data':      tensor.cpu().numpy(),
            'dtype':     tensor.dtype,
            'shape':     tensor.shape,
            'is_buffer': is_buffer
        }
    
    def _serialize_parameters(self: checkpoint) -> None:
        '''Walks the model's module tree and serializes all parameters.'''
        for module_name, module in self.model._walk_module_tree():
            for param_name, param in module._parameters.items():
                full_name = f'{module_name}.{param_name}' \
                         if module_name else param_name
                self._model_state[full_name] = \
                    checkpoint._serialize_tensor(param)
        
    def _serialize_buffers(self: checkpoint) -> None:
        '''Serializes all of the model's persistent buffers.'''
        for module_name, module in self.model._walk_module_tree():
            for buffer_name, buffer in module._buffers.items():
                if buffer_name in module._persistent_buffers:
                    full_name = f'{module_name}.{buffer_name}' \
                                if module_name else buffer_name
                    
                    self._model_state[full_name] = \
                        checkpoint._serialize_tensor(buffer, is_buffer=True)
    
    def _serialize_optimizer_state(self: checkpoint) -> None:
        '''Serializes all optimizer state tensors.'''
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
                        self._optimizer_state['state'][idx][k] = \
                            checkpoint._serialize_tensor(v)
                    else: self._optimizer_state['state'][idx][k] = v
    
    def save(
        self:                checkpoint,
        path:                PathLike,
        epoch:               int = 0,
        iteration:           int = 0,
        metadata: dict[str, Any] = None,
        overwrite:          bool = False
    ) -> None:
        '''Saves a checkpoint for the model.

        Checkpoints should be saved as `.nml` files. If you also add `.tar`, 
        (e.g. `filename.nml.tar`), the checkpoint will be saved as a tar 
        archive. 

        When saving as a tar archive, you may also add an additional
        compression type suffix, and the tarfile will be automatically
        compressed using the given format. Options are:
        - `.tar.gz`  : gzip compression
        - `.tar.bz2` : bzip2 compression
        - `.tar.xz`  : xz compression
        - `.tar.zst` : Zstandard compression

        Args:
            path      : The system path to the checkpoint file to write.
            epoch     : Optional epoch value to save with the checkpoint. Can 
                        be retrieved upon loading a checkpoint like so:
                        ```
                        info        = checkpoint.load(checkpoint_path)
                        start_epoch = info['epoch']
                        ```
            iteration : Optional iteration value, used the same way as `epoch`:
                        ```
                        info      = checkpoint.load(checkpoint_path)
                        iteration = info['iteration']
                        ```
            metadata  : Optional dict[str, Any] of metadata to save with the
                        checkpoint file. Retrived on load like so:
                        ```
                        info     = checkpoint.load(checkpoint_path)
                        metadata = info['metadata']
                        ```
            overwrite : If True, the save function will be allowed to overwrite
                        existing checkpoint files on disk if one exists with
                        the same path. If False, it will throw an error if it
                        finds an existing checkpoint file at the output path.
                        
        Raises:
            FileExistsError : When trying to save a checkpoint if a file 
                              already exists at the output path and `overwrite`
                              if False.
            ValueError      : If provided invalid suffixes.
        '''
        self.checkpoint_path = Path(path).resolve()
        if not overwrite and self.checkpoint_path.exists():
            raise FileExistsError(
                f'File exists at {self.checkpoint_path}. Use overwrite=True.')

        suffixes = [suffix.lower() for suffix in self.checkpoint_path.suffixes]
        assert suffixes[0] == '.nml', \
            f'save() requires output to be of type ".nml", not ' \
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
        '''Walks model's module tree and builds parameter/buffer lookups.'''
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
        '''Restores model parameter state from checkpoint data.
        
        Args:
            name  : The name of the parameter to restore.
            saved : The serialized tensor data for the given parameter.
        
        Raises:
            KeyError : If parameter name not found in model.
        '''
        if saved.get('is_buffer'): return
        if name not in self._param_lookup:
            raise KeyError(f'Parameter {name} not found in model')
        
        module, param_name = self._param_lookup[name]
        data     = saved['data'].astype(saved['dtype'].numpy)
        restored = Tensor._new(data, saved['shape'], saved['dtype'], 'cpu')
        
        param = restored.to(self._target_device)
        param._prev.clear()
        param._backward = lambda: None
        param.requires_grad_(True)
        module._parameters[param_name] = param

    def _restore_buffers(
        self:  checkpoint,
        name:  str,
        saved: dict[str, Any]
    ) -> None:
        '''Restores model buffer state from checkpoint data.
        
        Args:
            name  : The name of the buffer to restore.
            saved : The serialized tensor data for the given buffer.
        '''
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
        '''Restores optimizer state from checkpoint data.
        
        Args:
            opt_state : The serialized optimizer state data from the checkpoint
                        file.
        '''
        for idx, state in opt_state['state'].items():
            self.optimizer.state[int(idx)] = {}
            for k, v in state.items():
                if isinstance(v, dict) and 'data' in v:
                    data     = v['data'].astype(v['dtype'].numpy)
                    restored = Tensor._new(
                        data, v['shape'], v['dtype'], 'cpu',
                        requires_grad=False)
                    
                    tensor = restored.to(self._target_device)
                    tensor._prev.clear()
                    tensor._backward = lambda: None
                    tensor._requires_grad = False
                    self.optimizer.state[int(idx)][k] = tensor
                else: self.optimizer.state[int(idx)][k] = v

    def load(
        self: checkpoint,
        path: PathLike
    ) -> dict[str, Any]:
        '''Loads a checkpoint file from a filepath.

        This will load the checkpoint file and update all paramters and buffers
        of the Module provided for `model` when the checkpoint instance was 
        initialized. 

        It will also update the optimizer's state dictionary with
        the one from the checkpoint, if an `optimizer` was provided for 
        checkpoint init, and if an optimizer state is present in the loaded
        checkpoint file.

        Args:
            path : The path to the checkpoint file to load.
            
        Returns:
            dict : A dict containing the epoch, iteration, and metadata from 
                   the loaded checkpoint file. If no `iteration` of `epoch` was 
                   provided during checkpoint saving, they will default to 0. 
                   `metadata` defaults to an empty dict. This data can be 
                   retrieved like so:
                   ```
                    info        = checkpoint.load(checkpoint_path)
                    start_epoch = info['epoch']
                    iteration   = info['iteration']
                    metadata    = info['metadata']
                    ```
        Raises:
            FileNotFoundError : If unable to locate a checkpoint file at the 
                                provided path.
        '''
        self.checkpoint_path = Path(path).resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f'Unable to locate checkpoint file at: {self.checkpoint_path}')
            
        if not tarfile.is_tarfile(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as file:
                loaded = pickle.load(file)
        else:   loaded = _load_tarfile(self.checkpoint_path)

        self._target_device = 'cpu'
        for p in self.model.parameters():
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

            param_iter = iter([p for p in self.model.parameters()])
            for group in self.optimizer.param_groups:
                group['params'] = [next(param_iter) for _ in group['params']]
            
            self.optimizer._param_to_idx = {
                id(p): idx for idx, p in enumerate(self.optimizer.params())
            }

        return {
            'epoch':     loaded['epoch'],
            'iteration': loaded['iteration'],
            'metadata':  loaded['metadata']
        }

