from __future__ import annotations

import random
import multiprocessing
from typing import Any
from collections.abc import Callable, Iterable

import numpy as np

from nectarml.utils.data.dataset import (
    Dataset, IterableDataset, ConcatDataset)
from nectarml.utils.data.sampling import (
    Sampler, SequentialSampler, RandomSampler, BatchSampler)
from nectarml.utils.data.collate import default_collate

class Dataloader:
    def __init__(
        self:    Dataloader,
        dataset: Dataset | IterableDataset,
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
        self.num_workers = num_workers
        self.collate_fn = collate_fn or default_collate
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        
        if num_workers > 0:
            multiprocessing.set_start_method('spawn')
            self.queue = multiprocessing.Queue()
        
        if sampler is not None: self.sampler = sampler
        elif shuffle: self.sampler = RandomSampler(len(self.dataset))
        else: self.sampler = SequentialSampler(len(self.dataset))
        
        if batch_sampler is not None:
            assert sampler is None and not shuffle and batch_size == 1, \
                'Cannot specify batch_sampler with sampler, shuffle, ' \
                'or batch_size'
            self.batch_sampler = batch_sampler
        else:
            if sampler is not None:
                assert not shuffle, 'Cannot specify both sampler and shuffle'
                self.sampler = sampler
            else:
                if shuffle: self.sampler = RandomSampler(len(dataset))
                else: self.sampler = SequentialSampler(len(dataset))
            self.batch_sampler = BatchSampler(
                self.sampler, batch_size, drop_last)
        
        self._setup_complete = False
        self._prepare_complete = False
        
    def _worker_fn(
        dataset: Dataset,
        index_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
        collate_fn: Callable,
        worker_id: int,
        worker_init_fn: Callable | None
    ) -> None:
        random.seed(worker_id)
        np.random.seed(worker_id)
        
        if worker_init_fn is not None:
            worker_init_fn(worker_id)
        
        while True:
            item = index_queue.get()
            if item is None: break
            
            batch_idx, indices = item
            samples = [dataset[idx] for idx in indices]
            result = collate_fn(samples)
            result_queue.put((batch_idx, result))
            
    def setup(self: Dataloader) -> None:
        self._setup_complete = True
    
    def prepare(self: Dataloader) -> None:
        self._prepare_complete = True

    def _single_process_iter(self: Dataloader) -> Iterable[Any]:
        for batch_indices in self.batch_sampler:
            samples = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(samples)

    def _multi_process_iter(self: Dataloader) -> Iterable[Any]:
        index_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()
        
        workers = []
        for worker_id in range(self.num_workers):
            w = multiprocessing.Process(
                target=self._worker_fn,
                args=(self.dataset, index_queue, result_queue,
                    self.collate_fn, worker_id, self.worker_init_fn),
                daemon=True)
            w.start()
            workers.append(w)
        
        batches = list(self.batch_sampler)
        n_batches = len(batches)
        
        prefetch = min(self.num_workers * self.prefetch_factor, n_batches)
        for batch_idx in range(prefetch):
            index_queue.put((batch_idx, batches[batch_idx]))
        
        results = {}
        next_to_yield = 0
        next_to_send = prefetch
        
        while next_to_yield < n_batches:
            batch_idx, batch = result_queue.get()
            results[batch_idx] = batch
            
            if next_to_send < n_batches:
                index_queue.put((next_to_send, batches[next_to_send]))
                next_to_send += 1
            
            while next_to_yield in results:
                yield results.pop(next_to_yield)
                next_to_yield += 1
        
        for _ in workers: index_queue.put(None)
        for w in workers: w.join()
            
    def __iter__(self: Dataloader) -> Iterable[Any]:
        if self.num_workers == 0: yield from self._single_process_iter()
        else: yield from self._multi_process_iter()
        
    def __len__(self: Dataloader) -> int:
        return len(self.batch_sampler)
    
    def __add__(self: Dataloader, other: Dataloader) -> Dataloader:
        if not self._setup_complete: self.setup()
        if not other._setup_complete: other.setup()
        if not self._prepare_complete: self.prepare()
        if not other._prepare_complete: other.prepare()
        
        new_dataset = ConcatDataset([self.dataset, other.dataset])
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
        self.dataset = ConcatDataset([self.dataset, other.dataset])
        return self
    
    def __repr__(self: Dataloader) -> str:
        return f'Dataloader(Type={self.__class__()}, Size={len(self)})'
