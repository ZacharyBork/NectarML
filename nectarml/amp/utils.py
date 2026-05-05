from nectarml import typing

### DTYPES ###

_DTYPE_RANK_CUDA = {
    typing.float32: 1,
    typing.float16: 0
}

def get_promotion_dtype(dtypes: list[typing.dtype]) -> typing.dtype:
    '''Utility to get the highest ranking DType from a group of tensors.

    Used for tensor combination functions inside of autocast blocks to ensure
    that all tensors have matching DTypes.

    Args:
        dtypes : List of DTypes to find the highest ranking one from.
        
    Returns:
        dtype : The highest ranking dtype from the list.
    '''
    highest = None
    rank    = -1
    valid   = [x for x in dtypes if x in _DTYPE_RANK_CUDA]
    for x in valid:
        x_rank = _DTYPE_RANK_CUDA[x]
        if x_rank > rank:
            highest = x
            rank = x_rank
        
    return highest


