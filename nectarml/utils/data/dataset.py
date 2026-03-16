from __future__ import annotations

from typing import Any
from collections.abc import Iterable

class Dataset:
    def __len__(self: Dataset) -> int:
        raise NotImplementedError
    
    def __getitem__(self: Dataset, index: int) -> Any:
        raise NotImplementedError

class IterableDataset(Dataset):
    def __len__(self: IterableDataset) -> int:
        raise NotImplementedError
    
    def __getitem__(self: IterableDataset, index: int) -> Any:
        raise NotImplementedError
    
    def __iter__(self: IterableDataset) -> Any:
        raise NotImplementedError

class TensorDataset(Dataset):
    def __len__(self: TensorDataset) -> int:
        raise NotImplementedError
    
    def __getitem__(self: TensorDataset, index: int) -> Any:
        raise NotImplementedError
    
    def __iter__(self: TensorDataset) -> Any:
        raise NotImplementedError

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
    
