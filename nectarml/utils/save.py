import io
import time
import pickle
import tarfile
from os import PathLike
from pathlib import Path
from typing import Any

from nectarml.tensor import Tensor

### UTILS ###

def _save_tarfile(data: dict[str, Any], output_path: Path) -> None:
    suffixes = output_path.suffixes
    if len(suffixes) == 2: mode = 'w'
    else: 
        assert suffixes[-1] in ['.gz', '.bz2', '.xz', '.zst'], \
            f'Unable to save Tensor data with file suffixes: {suffixes}'
        mode = suffixes[-1].replace('.', 'w:')
    
    pickled = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
    info = tarfile.TarInfo()
    info.size = len(pickled)
    info.mtime = time.time()
    with tarfile.open(output_path, f'{mode}') as tar:
        tar.addfile(info, io.BytesIO(pickled))
        
def _load_tarfile(file_path: Path) -> dict[str, Any]:
    with tarfile.open(file_path, 'r') as tar:
        for member in tar:
            file_object = tar.extractfile(member)
            if file_object is not None:
                data = pickle.load(file_object)
                file_object.close()
    return data

### PICKLING ###

def save(
    tensor: Tensor, 
    output_path: PathLike,
    overwrite: bool = False
) -> None:
    output_path = Path(output_path).resolve()
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f'Unable to locate output directory at path: '
            f'{output_path.as_posix()}')

    if not overwrite and output_path.exists():
        raise FileExistsError(
            f'Found existing file at path: {output_path.as_posix()}\n'
            f'To allow overwriting of existing files, please run save() with '
            f'overwrite=True.')

    suffixes = output_path.suffixes
    assert suffixes[0] in ['.pt', '.pth'], \
        f'save() requires output to be of type ".pt" or ".pth", not ' \
        f'[{output_path.suffix}]'
    
    data = {
        'dtype': tensor.dtype,
        'shape': tensor.shape,
        'data' : tensor.numpy()
    }
    
    if len(suffixes) == 1:
        with open(output_path, 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    elif '.tar' in suffixes: _save_tarfile(data, output_path)
    else: raise ValueError(
        f'Unable to save Tensor data with file suffixes: {suffixes}')
    

def load(file_path: PathLike) -> Tensor:
    file_path = Path(file_path).resolve()
    if not file_path.parent.exists():
        raise FileNotFoundError(
            f'Unable to locate input file at path: {file_path.as_posix()}')

    if tarfile.is_tarfile(file_path):
        data = _load_tarfile(file_path)
    else:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
    tensor = Tensor(data['data'], data['shape'], data['dtype'])
    return tensor

