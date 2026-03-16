from __future__ import annotations

from typing import Any, Literal
from os import PathLike
from pathlib import Path
from collections.abc import Iterable

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32
from nectarml.vision.transforms import Transform
from nectarml.vision.utils import load_image

### ABSTRACTS ###

class Dataset:
    def __len__(self: Dataset) -> int:
        raise NotImplementedError
    
    def __getitem__(self: Dataset, index: int) -> Any:
        raise NotImplementedError

class IterableDataset:
    def __len__(self: IterableDataset) -> int:
        raise NotImplementedError
    
    def __iter__(self: IterableDataset) -> Iterable[Any]:
        raise NotImplementedError

### CORE DATASETS ###

class ImageDataset(Dataset):
    def __init__(
        self: ImageDataset,
        image_directory: str | PathLike,
        extensions = ['.jpg', '.jpeg', '.png', '.bmp'],
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32,
        normalize: bool = False,
        value_range: tuple[int | float, int | float] = [0.0, 1.0],
        transform: Transform = None
    ) -> None:
        super().__init__()
        self.image_directory = Path(image_directory).resolve()
        assert self.image_directory.exists(), (
            f'Unable to locate image directory at path: '
            f'{self.image_directory.as_posix()}')
        
        self.image_files = [
            f for f in self.image_directory.glob('*') 
            if f.suffix.lower() in extensions]
        assert len(self.image_files) > 0, (
            f'Unable to locate image files in directory: '
            f'{self.image_directory.as_posix()}')
        
        self.length = len(self.image_files)
        self.transform = transform
        self.load = lambda x : load_image(
            x, dtype, device, normalize, value_range, batch_dim=False)
        
    def __len__(self: ImageDataset) -> int:
        return self.length
    
    def __getitem__(self: ImageDataset, index: int) -> Tensor:
        sample = self.load(self.image_files[index])
        if self.transform: sample = self.transform(sample)
        return sample

class TensorDataset(Dataset):
    def __init__(
        self: TensorDataset, 
        *tensors: Tensor | Iterable[Tensor],
        transform: Transform = None
    ) -> None:
        super().__init__()
        self.tensors = list(tensors)
        assert len(self.tensors) > 0, \
            'At least one Tensor is required to intialize a TensorDataset.'
        
        self.transform = transform
        self.length = self.tensors[0][0]
        
    def __len__(self: TensorDataset) -> int:
        return self.length
    
    def __getitem__(self: TensorDataset, index: int) -> tuple[Tensor]:
        sample = tuple(t[index] for t in self.tensors)
        if self.transform: sample = self.transform(sample)
        return sample

class StackDataset(Dataset):
    def __init__(
        self: StackDataset,
        datasets: Iterable[Dataset]
    ) -> None:
        super().__init__()
        self.datasets = datasets
    
    def __len__(self: StackDataset) -> int:
        raise NotImplementedError
    
    def __getitem__(self: StackDataset, index: int) -> Any:
        raise NotImplementedError
    
    def __iter__(self: StackDataset) -> Any:
        raise NotImplementedError
        
class ChainDataset(Dataset):
    def __len__(self: ChainDataset) -> int:
        raise NotImplementedError
    
    def __getitem__(self: ChainDataset, index: int) -> Any:
        raise NotImplementedError
    
    def __iter__(self: ChainDataset) -> Any:
        raise NotImplementedError
    
