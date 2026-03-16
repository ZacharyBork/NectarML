import math
import random

from nectarml.utils.data import Dataset, Subset

def random_split(
    dataset: Dataset, 
    chances: list[float]
) -> list[Subset]:
    assert math.isclose(sum(chances), 1.0), 'Split chances must add up to 1.0.'
    total = len(dataset)
    indices = [int(math.floor(c * total)) for c in chances]
    indices[-1] = total - sum(indices[:-1])
    
    all_indices = list(range(total))
    random.shuffle(all_indices)
    
    subsets = []
    start = 0
    for idx in indices:
        subsets.append(Subset(dataset, all_indices[start:start+idx]))
        start += idx
        
    return subsets
    

