from __future__ import annotations

import random

### INDEXING ###

class Sampler:
    def __init__(self: Sampler, dataset_length: int) -> None:
        self.dataset_length = dataset_length
        
    def __len__(self) -> int:
        return self.dataset_length
        
    def __iter__(self) -> list[int]:
        raise NotImplementedError

class SequentialSampler(Sampler):
    def __init__(self: SequentialSampler, dataset_length: int) -> None:
        super().__init__(dataset_length)

    def __iter__(self: SequentialSampler) -> list[int]:
        return list(range(self.dataset_length))
    
class RandomSampler(Sampler):
    def __init__(self: RandomSampler, dataset_length: int) -> None:
        super().__init__(dataset_length)

    def __iter__(self: RandomSampler) -> list[int]:
        indices = list(range(self.dataset_length))
        random.shuffle(indices)
        return indices
    
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
    
    def sample(self: WeightedRandomSampler) -> list[int]:
        indices = list(range(len(self.weights)))
        return random.choices(
            indices, weights=self.weights, 
            k=self.num_samples) if self.replacement else random.sample(
            indices, k=self.num_samples)

class SubsetRandomSampler(Sampler):
    def __init__(
        self: SubsetRandomSampler, 
        indices: list[int]
    ) -> None:
        super().__init__(len(self.indices))
        self.indices = indices

    def __iter__(self: SubsetRandomSampler) -> list[int]:
        indices = list(self.indices)
        random.shuffle(indices)
        return indices

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
        
    def __iter__(self: BatchSampler) -> list[list[int]]:
        pass
    


