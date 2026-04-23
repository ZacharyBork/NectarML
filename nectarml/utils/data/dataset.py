from __future__ import annotations

import csv
from os              import PathLike
from pathlib         import Path
from typing          import Any
from collections.abc import Iterable

from nectarml                   import typing
from nectarml.tensor            import Tensor
from nectarml.vision.transforms import Transform
from nectarml.vision.utils      import load_image

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

class ImageFolderDataset(Dataset):
    def __init__(
        self:            ImageFolderDataset,
        image_directory: str | PathLike,
        extensions:      list[str]    = ['.jpg', '.jpeg', '.png', '.bmp'],
        dtype:           typing.dtype = typing.float32,
        normalize:       bool         = False,
        transform:       Transform    = None
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
            x, dtype, normalize, batch_dim=False)
        
    def __len__(self: ImageFolderDataset) -> int:
        return self.length
    
    def __getitem__(self: ImageFolderDataset, index: int) -> Tensor:
        sample = self.load(self.image_files[index])
        if self.transform: sample = self.transform(sample)
        return sample

class TensorDataset(Dataset):
    def __init__(
        self:      TensorDataset, 
        *tensors:  Tensor | Iterable[Tensor],
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
    
class Subset(Dataset):
    def __init__(
        self:    Subset,
        dataset: Dataset,
        indices: Iterable[int]
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.length  = len(self.indices)
        
    def __len__(self: Subset) -> int:
        return self.length
    
    def __getitem__(self: Subset, index: int) -> Tensor:
        sample = self.dataset[self.indices[index]]
        if self.transform: sample = self.transform(sample)
        return sample
    
class CSVDataset(Dataset):
    def __init__(
        self:       CSVDataset, 
        csv_file:   str | PathLike, 
        has_header: bool = True, 
        **csv_kwargs
    ) -> None:
        self.csv_file = Path(csv_file)
        assert self.csv_file.exists(), \
            f'Unable to locate CSV file at path: {self.csv_file.as_posix()}'
        
        with open(self.csv_file, newline='') as file:
            reader = csv.reader(file, **csv_kwargs)
            rows = list(reader)
        
        if has_header:
            self.header = rows[0]
            self.rows = rows[1:]
        else:
            self.header = None
            self.rows = rows
        
    def __len__(self: CSVDataset) -> int:
        return len(self.rows)
    
    def __getitem__(self: CSVDataset, index: int) -> Tensor | tuple[Any, ...]:
        return tuple(self.rows[index])

### COMBINED DATASETS ###

class ConcatDataset(Dataset):
    def __init__(
        self:     ConcatDataset,
        datasets: Iterable[Dataset]
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.lengths  = [len(i) for i in self.datasets]
        self.length   = sum(self.lengths)
    
    def __len__(self: ConcatDataset) -> int:
        return self.length
    
    def __getitem__(self, index: int) -> Any:
        offset = index
        for idx, length in enumerate(self.lengths):
            if offset < length:
                return self.datasets[idx][offset]
            offset -= length
        raise IndexError(
            f'Index {index} out of range for ConcatDataset of '
            f'length {self.length}')
    
class ChainDataset(IterableDataset):
    def __init__(
        self:     ConcatDataset, 
        datasets: Iterable[IterableDataset]
    ) -> None:
        super().__init__()
        self.datasets = list(datasets)
        self.length   = sum(len(i) for i in self.datasets)
    
    def __len__(self: ChainDataset) -> int:
        return self.length
    
    def __iter__(self: ChainDataset) -> Iterable[Tensor]:
        for dataset in self.datasets: yield from dataset

class StackDataset(Dataset):
    def __init__(
        self:     StackDataset, 
        datasets: Iterable[Dataset]
    ) -> None:
        super().__init__()
        assert len(datasets) >= 2, \
            'StackDataset requires at least two datasets.'
        assert all(len(d) == len(datasets[0]) for d in datasets), \
            'All datasets must have the same length.'
        self.datasets = datasets
    
    def __len__(self: StackDataset) -> int:
        return len(self.datasets[0])
    
    def __getitem__(self: StackDataset, index: int) -> tuple:
        return tuple(d[index] for d in self.datasets)
        

