import numpy as np

def eq_mask(a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
    return (a == b).astype(a.dtype)
    
def lt_mask(a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
    return (a < b).astype(a.dtype)

def le_mask(a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
    return (a <= b).astype(a.dtype)

def gt_mask(a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
    return (a > b).astype(a.dtype)

def ge_mask(a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
    return (a >= b).astype(a.dtype)


