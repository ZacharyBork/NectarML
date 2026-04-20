from nectarml import typing

### DTYPES ###

_DTYPE_RANK_CUDA = {
    typing.float32: 1,
    typing.float16: 0
}

def get_promotion_dtype(dtypes: list[typing.dtype]) -> typing.dtype:
    highest = None
    rank    = -1
    valid   = [x for x in dtypes if x in _DTYPE_RANK_CUDA]
    for x in valid:
        x_rank = _DTYPE_RANK_CUDA[x]
        if x_rank > rank:
            highest = x
            rank = x_rank
        
    return highest


