from __future__ import annotations
import numpy as np

def auxiliary_loss(graph_vec: np.ndarray, df_nodes, method: str = "none", **params) -> float:
    method = (method or "none").lower()
    return float(0.0)
