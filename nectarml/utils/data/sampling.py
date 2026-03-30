from __future__ import annotations

import math
from collections.abc import Iterator

from nectarml.random import RNG

### INDEXING ###

class Sampler:
    def __init__(self: Sampler, dataset_length: int) -> None:
        self.dataset_length = dataset_length
        
    def __len__(self) -> int:
        return self.dataset_length
        
    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

class SequentialSampler(Sampler):
    def __init__(self: SequentialSampler, dataset_length: int) -> None:
        super().__init__(dataset_length)

    def __iter__(self: SequentialSampler) -> Iterator[int]:
        for idx in range(self.dataset_length): yield idx
    
class RandomSampler(Sampler):
    def __init__(self: RandomSampler, dataset_length: int) -> None:
        super().__init__(dataset_length)

    def __iter__(self: RandomSampler) -> Iterator[int]:
        indices = list(range(self.dataset_length))
        RNG.shuffle(indices)
        for idx in indices: yield idx
    
class WeightedRandomSampler(Sampler):
    def __init__(
        self: WeightedRandomSampler,
        weights: list[float],
        num_samples: int | None = None,
        replacement: bool = True
    ) -> None:
        self.weights = weights
        self.num_samples = num_samples or len(weights)
        self.replacement = replacement
        super().__init__(self.num_samples)
    
    def __iter__(self: WeightedRandomSampler) -> Iterator[int]:
        indices = list(range(len(self.weights)))
        indices = RNG.choices(
            indices, weights=self.weights, 
            k=self.num_samples) if self.replacement \
            else RNG.sample(indices, k=self.num_samples)
        for idx in indices: yield idx

class SubsetRandomSampler(Sampler):
    def __init__(
        self: SubsetRandomSampler, 
        indices: list[int]
    ) -> None:
        super().__init__(len(self.indices))
        self.indices = indices

    def __iter__(self: SubsetRandomSampler) -> Iterator[int]:
        indices = list(self.indices)
        RNG.shuffle(indices)
        for idx in indices: yield idx

### BATCHING ###

class BatchSampler:
    def __init__(
        self: BatchSampler,
        sampler: Sampler,
        batch_size: int = 1,
        drop_last: bool = False
    ) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
     
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return math.ceil(len(self.sampler) / self.batch_size)
    
    def __iter__(self: BatchSampler) -> Iterator[list[int]]:
        indices = list(self.sampler)
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch
            
    


