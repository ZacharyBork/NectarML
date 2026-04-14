import io
import time
import pickle
import tarfile
from os import PathLike
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from nectarml.tensor import Tensor
from nectarml.nn.module import Module
from nectarml.optim.optimizer import Optimizer

### UTILS ###

def _save_tarfile(data: dict[str, Any], path: Path) -> None:
    suffixes = path.suffixes
    if len(suffixes) == 2: mode = 'w'
    else: 
        assert suffixes[-1] in ['.gz', '.bz2', '.xz', '.zst'], \
            f'Unable to save Tensor data with file suffixes: {suffixes}'
        mode = suffixes[-1].replace('.', 'w:')
    
    pickled = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
    info = tarfile.TarInfo()
    info.size = len(pickled)
    info.mtime = time.time()
    with tarfile.open(path, f'{mode}') as tar:
        tar.addfile(info, io.BytesIO(pickled))
        
def _load_tarfile(path: Path) -> dict[str, Any]:
    with tarfile.open(path, 'r') as tar:
        for member in tar:
            file_object = tar.extractfile(member)
            if file_object is not None:
                data = pickle.load(file_object)
                file_object.close()
    return data

### PICKLING ###

def save(
    input: Tensor | Iterable[Tensor], 
    path: PathLike,
    overwrite: bool = False
) -> None:
    path = Path(path).resolve()
    if not path.parent.exists():
        raise FileNotFoundError(
            f'Unable to locate output directory at path: {path.as_posix()}')

    if not overwrite and path.exists():
        raise FileExistsError(
            f'Found existing file at path: {path.as_posix()}\n'
            f'To allow overwriting of existing files, please run save() with '
            f'overwrite=True.')

    suffixes = path.suffixes
    assert suffixes[0] in ['.pt', '.pth'], \
        f'save() requires output to be of type ".pt" or ".pth", not ' \
        f'[{path.suffix}]'
    
    if isinstance(input, Tensor): input = [input]
    data = []
    for tensor in input:
        data.append({
            'dtype': tensor.dtype,
            'shape': tensor.shape,
            'data' : tensor.numpy()
        })
    
    if len(suffixes) == 1:
        with open(path, 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    elif '.tar' in suffixes: _save_tarfile(data, path)
    else: raise ValueError(
        f'Unable to save Tensor data with file suffixes: {suffixes}')
    
def load(path: PathLike) -> Tensor | list[Tensor]:
    path = Path(path).resolve()
    if not path.parent.exists():
        raise FileNotFoundError(
            f'Unable to locate input file at path: {path.as_posix()}')

    if not tarfile.is_tarfile(path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
    else: data = _load_tarfile(path)
        
    output = []
    for i in data: output.append(Tensor(i['data'], i['shape'], i['dtype']))
    if len(output) == 1: output = output[0]
    return output

### CHECKPOINTING ###

def save_checkpoint(
    path: PathLike,
    model: Module,
    optimizer: Optimizer | None = None,
    epoch: int = 0,
    iteration: int = 0,
    metadata: dict = None,
    overwrite: bool = False
) -> None:
    path = Path(path).resolve()
    
    if not overwrite and path.exists():
        raise FileExistsError(f'File exists at {path}. Use overwrite=True.')

    suffixes = [suffix.lower() for suffix in path.suffixes]
    assert suffixes[0] in ['.pt', '.pth'], \
        f'save() requires output to be of type ".pt" or ".pth", not ' \
        f'[{path.suffix}]'

    model_state = {}
    for name, param in model.list_parameters():
        model_state[name] = {
            'data':  param.numpy(),
            'dtype': param.dtype,
            'shape': param.shape
        }

    for module_name, module in model._walk_module_tree():
        for buffer_name, buffer in module._buffers.items():
            if buffer_name in module._persistent_buffers:
                full_name = f'{module_name}.{buffer_name}' \
                            if module_name else buffer_name
                model_state[full_name] = {
                    'data':  buffer.cpu().numpy(),
                    'dtype': buffer.dtype,
                    'shape': buffer.shape,
                    'is_buffer': True 
                }

    opt_state = None
    if optimizer is not None:
        opt_state = { 'param_groups': [], 'state': {} }
        
        for group in optimizer.param_groups:
            opt_state['param_groups'].append({
                k: v for k, v in group.items()
                if k not in ('params',)
            })
            
        for idx, state in optimizer.state.items():
            opt_state['state'][idx] = {}
            for k, v in state.items():
                if isinstance(v, Tensor):
                    opt_state['state'][idx][k] = {
                        'data':  v.numpy(),
                        'dtype': v.dtype,
                        'shape': v.shape
                    }
                else: opt_state['state'][idx][k] = v

    checkpoint = {
        'model_state': model_state,
        'opt_state':   opt_state,
        'epoch':       epoch,
        'iteration':   iteration,
        'metadata':    metadata or {}
    }

    if len(suffixes) == 1:
        with open(path, 'wb') as file:
            pickle.dump(checkpoint, file, pickle.HIGHEST_PROTOCOL)
    elif '.tar' in suffixes: _save_tarfile(checkpoint, path)
    else: raise ValueError(
        f'Unable to save checkpoint data with file suffixes: {suffixes}')

def load_checkpoint(
    path: PathLike,
    model: Module,
    optimizer: Optimizer | None = None
) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f'Unable to locate checkpoint file at: {path}')
        
    if not tarfile.is_tarfile(path):
        with open(path, 'rb') as file:
            checkpoint = pickle.load(file)
    else: checkpoint = _load_tarfile(path)

    model_state = checkpoint['model_state']
    for name, param in model.list_parameters():
        if name not in model_state:
            raise KeyError(f'Parameter {name} not found in checkpoint')
        saved = model_state[name]
        param.data = saved['data'].astype(saved['dtype'])
        param.shape = saved['shape']

    for module_name, module in model._walk_module_tree():
        for buffer_name, buffer in module._buffers.items():
            if buffer_name not in module._persistent_buffers:
                continue
            full_name = f'{module_name}.{buffer_name}' \
                        if module_name else buffer_name
            if full_name not in model_state: continue
            saved = model_state[full_name]
            restored = Tensor(
                saved['data'].astype(saved['dtype']),
                saved['shape'], saved['dtype'], 'cpu')
            module._buffers[buffer_name] = restored.to(buffer.device)

    if optimizer is not None and checkpoint['opt_state'] is not None:
        opt_state = checkpoint['opt_state']
        for idx, state in opt_state['state'].items():
            optimizer.state[idx] = {}
            for k, v in state.items():
                if isinstance(v, dict) and 'data' in v:
                    t = Tensor(v['data'], v['shape'], v['dtype'])
                    optimizer.state[idx][k] = t
                else: optimizer.state[idx][k] = v

    return {
        'epoch':     checkpoint['epoch'],
        'iteration': checkpoint['iteration'],
        'metadata':  checkpoint['metadata']
    }

