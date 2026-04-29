import io
import time
import pickle
import tarfile

from os      import PathLike
from pathlib import Path
from typing  import Any

from nectarml.core import Tensor

### UTILS ###

def _save_tarfile(data: dict[str, Any], path: Path) -> None:
    suffixes = path.suffixes
    if len(suffixes) == 2: mode = 'w'
    else: 
        assert suffixes[-1] in ['.gz', '.bz2', '.xz', '.zst'], \
            f'Unable to save Tensor data with file suffixes: {suffixes}'
        mode = suffixes[-1].replace('.', 'w:')
    
    pickled    = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
    info       = tarfile.TarInfo()
    info.size  = len(pickled)
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
    input:     Tensor | list[Tensor], 
    path:      PathLike,
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
    assert suffixes[0] == '.nml', \
        f'save() requires output to be of type ".nml", not ' \
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
    else:   data = _load_tarfile(path)
        
    output = []
    for i in data: output.append(Tensor(i['data'], i['shape'], i['dtype']))
    if len(output) == 1: output = output[0]
    return output



