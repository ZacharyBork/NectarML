from __future__ import annotations

from typing import Any
from collections.abc import Callable

from nectarml.utils.data.dataset import Dataset, StackDataset
from nectarml.utils.data.sampling import Sampler, BatchSampler
from nectarml.utils.data.collate import default_collate

class Dataloader:
    def __init__(
        self: Dataloader,
        dataset: Dataset,
        batch_size:             int = 1,
        shuffle:               bool = False,
        sampler:            Sampler = None,
        batch_sampler: BatchSampler = None,
        num_workers:            int = 0,
        collate_fn:        Callable = None,
        pin_memory:            bool = False,
        drop_last:             bool = False,
        timeout:                int = 0,
        worker_init_fn:    Callable = None,
        prefetch_factor:        int = 2,
        persistent_workers:    bool = False
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn or default_collate
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        
        self._setup_complete = False
        self._prepare_complete = False
        
    def setup(self: Dataloader) -> None:
        self._setup_complete = True
    
    def prepare(self: Dataloader) -> None:
        self._prepare_complete = True
        
    def _get_indices(self: Dataloader) -> list[int]:
        pass
    
    def _batch(self: Dataloader, indices: int) -> list[Any]:
        return [self.dataset[i] for i in indices]
        
    def __len__(self: Dataloader) -> int:
        return len(self.dataset)

    def __iter__(self: Dataloader) -> Any:
        if not self._setup_complete: self.setup()
        if not self._prepare_complete: self.prepare()
        
        indices = self._get_indices()
        for batch_indices in self._batch(indices):
            samples = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(samples)
    
    def __add__(self: Dataloader, other: Dataloader) -> Dataloader:
        if not self._setup_complete: self.setup()
        if not other._setup_complete: other.setup()
        if not self._prepare_complete: self.prepare()
        if not other._prepare_complete: other.prepare()
        
        new_dataset = StackDataset([self.dataset, other.dataset])
        new_dataloader = Dataloader(
            dataset=new_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=self.sampler,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.timeout,
            worker_init_fn=self.worker_init_fn,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers)

        new_dataloader._setup_complete = self._setup_complete
        new_dataloader._prepare_complete = self._prepare_complete

        return new_dataloader

    def __radd__(self: Dataloader, other: Dataloader) -> Dataloader:
        return self + other
    
    def __iadd__(self: Dataloader, other: Dataloader) -> Dataloader:
        self.dataset = StackDataset([self.dataset, other.dataset])
        return self
    
    def __repr__(self: Dataloader) -> str:
        return f'Dataloader(Type={self.__class__()}, Size={len(self)})'
