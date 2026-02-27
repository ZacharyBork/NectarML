import math

from nectarml import Tensor
from nectarml.functional import activation, combination, reductions

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None
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
    k_t = K.transpose((-2, -1))
    scores = Q @ k_t
    
    d_k = K.shape[-1]
    scaled_scores = scores / math.sqrt(d_k)
    if mask is not None: scaled_scores += (mask * -1e9)
    
    weights = activation.Softmax(scaled_scores)
    return weights @ V, weights

def multihead_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    num_heads: int = 8,
    mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    '''Multi-head attention mechanism
    
    Ref: https://arxiv.org/pdf/1706.03762 (3.2.2)
    
    Args:
        Q : The queries Tensor.
        K : The keys Tensor.
        V : The values Tensor.
        num_heads: The number of heads to project the queries, keys, and values
            to before applying attention.
        mask : (optional) The mask to apply to the attention weights.
        
    Returns:
        tuple[Tensor, Tensor] : The resulting Tensor from the attention 
            mechanism, and the raw weights applied to the attention.
    '''
    d_k = K.shape[-1]
    assert d_k % num_heads == 0
    projection_dim = int(d_k / num_heads)

    values: list[Tensor] = []
    weights: list[Tensor] = []
    
    for i in range(num_heads):
        start = projection_dim * i
        end = projection_dim * (i + 1)
        
        Q_h = Q[:, :, start:end]
        K_h = K[:, :, start:end]
        V_h = V[:, :, start:end]
        
        value, weight = scaled_dot_product_attention(Q_h, K_h, V_h, mask=mask)
        values.append(value)
        weights.append(weight)
        
    out_value = combination.cat(values, dim=-1)
    out_weight = reductions.sum(combination.stack(weights, dim=0), dim=0)
    return out_value, out_weight


