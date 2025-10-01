from __future__ import annotations
import numpy as np
def fuse(structure_Z: np.ndarray, attribute_X: np.ndarray, method: str = "concat", **kwargs) -> np.ndarray:
    method = (method or "concat").lower()
    if method == "concat":
        return np.concatenate([structure_Z, attribute_X], axis=1)
    return np.concatenate([structure_Z, attribute_X], axis=1)
