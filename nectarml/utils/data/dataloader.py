from __future__ import annotations

import signal
import multiprocessing as mp
from   typing          import Any, Iterator
from   collections.abc import Callable

from nectarml import random
from nectarml.utils.data.dataset import (
    Dataset, IterableDataset, ConcatDataset)
from nectarml.utils.data.sampling import (
    Sampler, SequentialSampler, RandomSampler, BatchSampler)
from nectarml.utils.data.collate import default_collate

_WORKER_SHUTDOWN = None 

def _worker_loop(
    dataset:        Dataset,
    index_queue:    mp.Queue,
    result_queue:   mp.Queue,
    collate_fn:     Callable,
    worker_id:      int,
    worker_init_fn: Callable | None,
    seed:           int
) -> None:
    random.manual_seed(seed + worker_id)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if worker_init_fn is not None: worker_init_fn(worker_id)

    while True:
        item = index_queue.get()
        if item is _WORKER_SHUTDOWN: break

        batch_idx, indices = item
        try:
            samples = [dataset[i] for i in indices]
            result  = collate_fn(samples)
            result_queue.put((batch_idx, result))
        except Exception as e:
            result_queue.put((batch_idx, _WorkerError(e, worker_id)))

class _WorkerError:
    def __init__(self: _WorkerError, exc: Exception, worker_id: int):
        self.exc       = exc
        self.worker_id = worker_id

    def reraise(self: _WorkerError) -> None:
        raise RuntimeError(
            f'Worker {self.worker_id} raised: {type(self.exc).__name__}: '
            f'{self.exc}') from self.exc

class _SingleProcessIter:
    def __init__(self: _SingleProcessIter, loader: Dataloader) -> None:
        self._dataset       = loader.dataset
        self._batch_sampler = loader.batch_sampler
        self._collate_fn    = loader.collate_fn
        self._iter          = iter(self._batch_sampler)

    def __iter__(self: _SingleProcessIter) -> _SingleProcessIter:
        return self

    def __next__(self) -> Any:
        indices = next(self._iter)
        samples = [self._dataset[i] for i in indices]
        return self._collate_fn(samples)

class _MultiProcessIter:
    def __init__(self: _MultiProcessIter, loader: Dataloader) -> None:
        self._dataset         = loader.dataset
        self._collate_fn      = loader.collate_fn
        self._num_workers     = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._persistent      = loader.persistent_workers
        self._timeout         = loader.timeout or None

        self._batches         = list(loader.batch_sampler)
        self._n_batches       = len(self._batches)
        self._next_to_yield   = 0
        self._next_to_send    = 0
        self._buffer: dict[int, Any] = {}
        self._workers_alive   = False

        self._index_queue  = mp.Queue()
        self._result_queue = mp.Queue()
        self._workers: list[mp.Process] = []

        self._start_workers(loader.worker_init_fn, loader._seed)
        self._prefetch()

    def _start_workers(
        self:           _MultiProcessIter,
        worker_init_fn: Callable | None,
        seed:           int
    ) -> None:
        for worker_id in range(self._num_workers):
            p = mp.Process(
                target=_worker_loop,
                args=(
                    self._dataset, self._index_queue, self._result_queue,
                    self._collate_fn, worker_id, worker_init_fn, seed
                ),
                daemon=True
            )
            p.start()
            self._workers.append(p)
        self._workers_alive = True

    def _prefetch(self: _MultiProcessIter) -> None:
        depth = min(self._num_workers * self._prefetch_factor, self._n_batches)
        while self._next_to_send < depth: self._send_batch(self._next_to_send)

    def _send_batch(self: _MultiProcessIter, batch_idx: int) -> None:
        if batch_idx >= self._n_batches: return
        self._index_queue.put((batch_idx, self._batches[batch_idx]))
        self._next_to_send += 1

    def _shutdown_workers(self: _MultiProcessIter) -> None:
        if not self._workers_alive: return
        try:
            while True:
                self._result_queue.get_nowait()
        except Exception: pass
        
        for _ in self._workers: self._index_queue.put(_WORKER_SHUTDOWN)
        for w in self._workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()
        self._workers_alive = False

    def __iter__(self: _MultiProcessIter) -> _MultiProcessIter:
        return self

    def __next__(self: _MultiProcessIter) -> Any:
        if self._next_to_yield >= self._n_batches:
            self._shutdown_workers()
            raise StopIteration

        while self._next_to_yield not in self._buffer:
            try:
                batch_idx, result = self._result_queue.get(
                    timeout=self._timeout)
            except mp.queues.Empty:
                raise RuntimeError(
                    f'Dataloader worker timed out after {self._timeout}s. '
                    f'Consider increasing timeout.')

            if isinstance(result, _WorkerError):
                self._shutdown_workers()
                result.reraise()

            self._buffer[batch_idx] = result
            self._send_batch(self._next_to_send)

        batch = self._buffer.pop(self._next_to_yield)
        self._next_to_yield += 1
        return batch

    def __del__(self: _MultiProcessIter) -> None:
        self._shutdown_workers()

class Dataloader:
    def __init__(
        self:                Dataloader,
        dataset:             Dataset | IterableDataset,
        batch_size:              int = 1,
        shuffle:                bool = False,
        sampler:             Sampler = None,
        batch_sampler:  BatchSampler = None,
        num_workers:             int = 0,
        collate_fn:         Callable = None,
        pin_memory:             bool = False,
        drop_last:              bool = False,
        timeout:                 int = 0,
        worker_init_fn:     Callable = None,
        prefetch_factor:         int = 2,
        persistent_workers:     bool = False,
        seed:                    int = 0
    ) -> None:
        self.dataset             = dataset
        self.batch_size          = batch_size
        self.shuffle             = shuffle
        self.num_workers         = num_workers
        self.collate_fn          = collate_fn or default_collate
        self.pin_memory          = pin_memory
        self.drop_last           = drop_last
        self.timeout             = timeout
        self.worker_init_fn      = worker_init_fn
        self.prefetch_factor     = prefetch_factor
        self.persistent_workers  = persistent_workers
        self._seed               = seed

        if batch_sampler is not None:
            if sampler is not None or shuffle or batch_size != 1:
                raise ValueError(
                    'Cannot specify batch_sampler together with sampler, '
                    'shuffle, or batch_size != 1.')
                
            self.sampler       = None
            self.batch_sampler = batch_sampler
        else:
            if sampler is not None and shuffle:
                raise ValueError(
                    'Cannot specify both sampler and shuffle=True.')
                
            if sampler is not None: self.sampler = sampler
            elif shuffle: self.sampler = RandomSampler(len(dataset))
            else:         self.sampler = SequentialSampler(len(dataset))
            
            self.batch_sampler = BatchSampler(
                self.sampler, batch_size, drop_last)

        if num_workers > 0:
            try: mp.set_start_method('spawn')
            except RuntimeError: pass

    def __iter__(self: Dataloader) -> Iterator[Any]:
        if self.num_workers == 0: return _SingleProcessIter(self)
        return _MultiProcessIter(self)

    def __len__(self: Dataloader) -> int:
        return len(self.batch_sampler)

    def __add__(self: Dataloader, other: Dataloader) -> Dataloader:
        return Dataloader(
            dataset            = ConcatDataset([self.dataset, other.dataset]),
            batch_size         = self.batch_size,
            shuffle            = self.shuffle,
            num_workers        = self.num_workers,
            collate_fn         = self.collate_fn,
            pin_memory         = self.pin_memory,
            drop_last          = self.drop_last,
            timeout            = self.timeout,
            worker_init_fn     = self.worker_init_fn,
            prefetch_factor    = self.prefetch_factor,
            persistent_workers = self.persistent_workers,
            seed               = self._seed)

    def __iadd__(self: Dataloader, other: Dataloader) -> Dataloader:
        self.dataset = ConcatDataset([self.dataset, other.dataset])
        self.batch_sampler = BatchSampler(
            self.sampler, self.batch_size, self.drop_last)
        return self

    def __repr__(self: Dataloader) -> str:
        return (
            f'Dataloader('
            f'dataset={type(self.dataset).__name__}, '
            f'batch_size={self.batch_size}, '
            f'shuffle={self.shuffle}, '
            f'num_workers={self.num_workers}, '
            f'n_batches={len(self)})')

