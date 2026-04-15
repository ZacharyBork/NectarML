import math

from nectarml.tensor import Tensor
from nectarml.typing import bool_, float16
from nectarml.creation import tril
from nectarml.functional import activation
from nectarml.functional.dropout import dropout as dropout_fn

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    attn_mask: Tensor | None = None,
    key_padding_mask: Tensor | None = None,
    dropout: float = 0.0,
    is_causal: bool = False,
    training: bool = True
) -> tuple[Tensor, Tensor]:
    '''Scaled dot product attention mechanism.
    
    Ref: https://arxiv.org/pdf/1706.03762 (3.2.1)
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Args:
        Q : The queries Tensor.
        K : The keys Tensor.
        V : The values Tensor.
        mask : (optional) The mask to apply to the attention weights.
        
    Returns:
        tuple[Tensor, Tensor] : The resulting Tensor from the attention 
            mechanism, and the raw weights applied to the attention.
    '''
    k_t           = K.transpose(-2, -1)
    scores        = Q @ k_t
    d_k           = K.shape[-1]
    scaled_scores = scores / math.sqrt(d_k)
    
    if attn_mask is not None:
        if attn_mask.dtype == bool_:
              scaled_scores = scaled_scores + attn_mask.to(dtype=Q.dtype)*-1e9
        else: scaled_scores = scaled_scores + attn_mask
    
    if key_padding_mask is not None:
        mask          = key_padding_mask.unsqueeze(1).unsqueeze(2)
        scaled_scores = scaled_scores + mask*-1e9
        
    if is_causal:
        T           = Q.shape[-2]
        causal      = tril(size=T, device=Q.device)
        mask_val    = -1e4 if Q.dtype == float16 else -1e9
        causal_mask = (1 - causal) * mask_val
        scaled_scores = scaled_scores + causal_mask
        
    weights = activation.softmax(scaled_scores)
    if dropout > 0.0 and training:
        weights = dropout_fn(weights, p=dropout, training=training)
    return weights @ V, weights

def multi_query_attention() -> tuple[Tensor]: 
    '''https://arxiv.org/pdf/1911.02150'''
    raise NotImplementedError

def grouped_query_attention() -> tuple[Tensor]:
    '''https://arxiv.org/pdf/2305.13245'''
    raise NotImplementedError

def cross_attention() -> tuple[Tensor]:
    '''https://arxiv.org/pdf/2512.19535'''
    raise NotImplementedError

def flash_attention() -> tuple[Tensor]:
    '''https://arxiv.org/pdf/2205.14135'''
    raise NotImplementedError

def linear_attention() -> tuple[Tensor]:
    '''https://arxiv.org/pdf/2006.16236'''
    raise NotImplementedError

def sliding_window_attention() -> tuple[Tensor]:
    '''https://arxiv.org/pdf/2502.18845'''
    raise NotImplementedError

def longformer_attention() -> tuple[Tensor]:
    '''https://arxiv.org/pdf/2004.05150'''
    raise NotImplementedError

def self_attention() -> tuple[Tensor]:
    '''https://arxiv.org/pdf/1706.03762'''
    raise NotImplementedError

def causal_attention() -> tuple[Tensor]:
    '''https://arxiv.org/pdf/2103.03493'''
    raise NotImplementedError

