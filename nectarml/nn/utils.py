from nectarml.tensor import Tensor

def clip_grad_norm(params: list[Tensor], max_norm: float = 1.0) -> float:
    total_sq = sum(
        param.grad.norm(p='fro').item()**2
        for param in params if param.grad is not None)
    total_norm = total_sq ** 0.5
    
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for param in params:
            if param.grad is not None:
                param.grad *= scale
    
    return total_norm

