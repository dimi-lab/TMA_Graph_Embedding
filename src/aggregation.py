# aggregation.py
from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


def _ensure_torch():
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except Exception as e:
        raise ImportError(
            "This aggregation method requires PyTorch and PyTorch Geometric. "
            "Install torch + torch_scatter/torch_sparse and torch_geometric."
        ) from e

def _nx_nodes_in_order(G: nx.Graph) -> List[Any]:
    # IMPORTANT: this must match the order used when you created Z.
    # By default we assume list(G.nodes()) was used upstream.
    return list(G.nodes())

def _batch_from_graph(
    G: nx.Graph,
    nodelist: Optional[Sequence[Any]] = None,
    roi_key: Optional[str] = None,
) -> Tuple["np.ndarray", List[Any]]:
    """
    Builds a PyG-style batch vector (node -> ROI index) and an ordered list of ROI IDs.

    Precedence:
      1) roi_key node attribute
      2) tuple node ids: (roi_id, local_node_id)
      3) connected components
    """
    nodelist = _nx_nodes_in_order(G) if nodelist is None else list(nodelist)
    # Map each node to an ROI id value (hashable)
    roi_vals: List[Any] = []
    for n in nodelist:
        if roi_key is not None and roi_key in G.nodes[n]:
            roi_vals.append(G.nodes[n][roi_key])
        elif isinstance(n, tuple) and len(n) >= 1:
            roi_vals.append(n[0])
        else:
            roi_vals.append(None)

    if all(v is not None for v in roi_vals):
        # Use first-appearance order to create contiguous indices
        roi_to_idx: Dict[Any, int] = {}
        ordered_roi_ids: List[Any] = []
        batch = np.empty(len(nodelist), dtype=np.int64)
        for i, rid in enumerate(roi_vals):
            if rid not in roi_to_idx:
                roi_to_idx[rid] = len(ordered_roi_ids)
                ordered_roi_ids.append(rid)
            batch[i] = roi_to_idx[rid]
        return batch, ordered_roi_ids

    # Fallback: connected components
    comp_map: Dict[Any, int] = {}
    for ci, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            comp_map[n] = ci
    # Preserve nodelist order in batch and create ordered ROI ids by first appearance
    seen = {}
    ordered_cc_ids: List[int] = []
    tmp = []
    for n in nodelist:
        ci = comp_map[n]
        tmp.append(ci)
        if ci not in seen:
            seen[ci] = True
            ordered_cc_ids.append(ci)
    # Remap to 0..K-1 in the order of appearance
    remap = {cid: i for i, cid in enumerate(ordered_cc_ids)}
    batch = np.array([remap[cid] for cid in tmp], dtype=np.int64)
    return batch, ordered_cc_ids

class _MILAttentionPooler:
    """
    ABMIL-style attention pooling (Ilse et al., 2018).
    a_i = softmax(w^T tanh(V x_i)) per bag;  z_bag = sum_i a_i x_i
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, device: Optional[str] = None):
        _ensure_torch()
        import torch
        import torch.nn as nn
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

    def __call__(self, x: "torch.Tensor", batch: "torch.Tensor") -> "torch.Tensor":
        """
        x: (N, d), batch: (N,) with values in 0..B-1
        returns: (B, d)
        """
        import torch
        from torch_geometric.utils import softmax
        from torch_scatter import scatter_add

        scores = self.attn(x).squeeze(-1)  # (N,)
        exp_scores = torch.exp(scores)
        denom = scatter_add(exp_scores, batch, dim=0, dim_size=int(batch.max().item()) + 1)
        alphas = exp_scores / (denom[batch] + 1e-12)  # (N,)
        # Weighted sum per bag:
        x_weighted = x * alphas.unsqueeze(-1)
        z = scatter_add(x_weighted, batch, dim=0, dim_size=int(batch.max().item()) + 1)  # (B, d)
        return z

def aggregate(
    Z: np.ndarray,
    G: Optional[nx.Graph] = None,
    method: str = "mean_pool",
    *,
    # Grouping
    batch: Optional["np.ndarray"] = None,
    roi_key: Optional[str] = None,
    nodelist: Optional[Sequence[Any]] = None,
    # PyG options
    set2set_steps: int = 3,
    set2set_layers: int = 1,
    attn_hidden: int = 128,
    softmax_t: float = 1.0,
    power_p: float = 1.0,
    lstm_hidden: Optional[int] = None,
    device: Optional[str] = None,
    # Output control
    return_group_ids: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[Any]]]:
    """
    Aggregate node embeddings Z -> ROI-level embeddings.

    Parameters
    ----------
    Z : np.ndarray, shape (N_nodes, d)
    G : optional nx.Graph used to build 'batch' if not provided
    method : str
        One of:
          PyG globals: 'mean_pool', 'add_pool'/'sum_pool', 'max_pool',
                       'set2set', 'global_attention',
                       'softmax_pool', 'powermean_pool', 'lstm_pool'
          MIL: 'mil_mean', 'mil_max', 'mil_attention'
        'attention_stub' kept for backward-compat; acts like mean.
    batch : np.ndarray[int], optional
        PyG-style group index for each node. If None, will be inferred from G.
    roi_key : str, optional
        Node attribute naming the ROI id (used to form groups).
    nodelist : sequence, optional
        Node order used to create Z. Defaults to list(G.nodes()).
    set2set_steps, set2set_layers : int
        For Set2Set.
    attn_hidden : int
        Hidden size for attention gates (GlobalAttention / MIL attention).
    softmax_t : float
        Temperature for SoftmaxAggregation (softmax_pool).
    power_p : float
        Exponent p for PowerMeanAggregation.
    lstm_hidden : int or None
        Hidden size for LSTMAggregation; default = Z.shape[1].
    device : str or None
        'cpu' or 'cuda'; auto-detect if None.
    return_group_ids : bool
        If True, also returns the list of ROI/group ids in output order.

    Returns
    -------
    E : np.ndarray of shape (N_groups, d_out)
    (optionally also) group_ids : list
    """
    method = (method or "mean_pool").lower()
    Z = np.asarray(Z)
    N, d = Z.shape
    if N == 0:
        out = np.zeros((0, d), dtype=Z.dtype)
        return (out, []) if return_group_ids else out

    # Build/validate batch
    if batch is None:
        if G is None:
            # Single-bag fallback
            batch = np.zeros(N, dtype=np.int64)
            group_ids = [0]
        else:
            nodelist = _nx_nodes_in_order(G) if nodelist is None else list(nodelist)
            b, group_ids = _batch_from_graph(G, nodelist=nodelist, roi_key=roi_key)
            batch = b
    else:
        batch = np.asarray(batch).reshape(-1)
        if batch.shape[0] != N:
            raise ValueError(f"batch length {batch.shape[0]} != Z rows {N}")
        group_ids = list(range(int(batch.max()) + 1)) if len(batch) else []

    # Fast path: pure NumPy mean (keeps your original default)
    if method in ("mean_pool", "attention_stub"):
        # NumPy groupby-mean
        k = int(batch.max()) + 1 if len(batch) else 0
        if k == 0:
            out = Z.mean(axis=0, keepdims=True)
        else:
            # Compute per-group mean
            sums = np.zeros((k, d), dtype=Z.dtype)
            counts = np.zeros((k, 1), dtype=np.int64)
            for i, g in enumerate(batch):
                sums[g] += Z[i]
                counts[g, 0] += 1
            out = sums / np.maximum(counts, 1)
        return (out, group_ids) if return_group_ids else out

    # From here on we require PyTorch + PyG
    _ensure_torch()
    import torch
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x = torch.from_numpy(Z).to(device_t, non_blocking=True).float()
    b = torch.from_numpy(batch).to(device_t, non_blocking=True).long()
    num_groups = int(b.max().item()) + 1 if b.numel() > 0 else 0

    # Globals
    if method in ("add_pool", "sum_pool", "max_pool") or method == "global_attention" or \
       method in ("softmax_pool", "powermean_pool", "lstm_pool", "set2set") or \
       method in ("mil_mean", "mil_max", "mil_attention"):
        from torch_geometric.nn import (
            global_add_pool, global_max_pool, global_mean_pool,
            Set2Set
        )
        # Some aggregators live in aggr.* API:
        from torch_geometric.nn.aggr import AttentionalAggregation as GlobalAttention
        from torch_geometric.nn.aggr import SoftmaxAggregation, PowerMeanAggregation, LSTMAggregation

    if method in ("sum_pool", "add_pool"):
        out = global_add_pool(x, b)
    elif method == "max_pool":
        out = global_max_pool(x, b)
    elif method == "set2set":
        s2s = Set2Set(d, processing_steps=int(set2set_steps), num_layers=int(set2set_layers)).to(device_t)
        out = s2s(x, b)  # shape: (B, 2d)
    elif method == "global_attention":
        gate_nn = torch.nn.Sequential(
            torch.nn.Linear(d, int(attn_hidden)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(attn_hidden), 1),
        ).to(device_t)
        ga = GlobalAttention(gate_nn).to(device_t)
        out = ga(x, b)  # (B, d)
    elif method == "softmax_pool":
        aggr = SoftmaxAggregation(t=float(softmax_t)).to(device_t)
        out = aggr(x, b)  # (B, d)
    elif method == "powermean_pool":
        aggr = PowerMeanAggregation(p=float(power_p)).to(device_t)
        out = aggr(x, b)  # (B, d)
    elif method == "lstm_pool":
        hidden = int(lstm_hidden or d)
        aggr = LSTMAggregation(in_channels=d, out_channels=hidden).to(device_t)
        out = aggr(x, b, dim_size=num_groups)  # (B, hidden)
    # MIL poolers
    elif method == "mil_mean":
        out = global_mean_pool(x, b)
    elif method == "mil_max":
        out = global_max_pool(x, b)
    elif method == "mil_attention":
        pool = _MILAttentionPooler(in_dim=d, hidden_dim=int(attn_hidden), device=str(device_t))
        out = pool(x, b)  # (B, d)
    else:
        raise NotImplementedError(f"Aggregation method not implemented: {method}")

    out_np = out.detach().cpu().numpy()
    return (out_np, group_ids) if return_group_ids else out_np

'''
# Z: (N_nodes, d) from any embedding method you already run
# G_all: a NetworkX graph whose nodes are either
#   - tuples like (roi_id, local_id), or
#   - have a node attribute 'roi_id'

# 1) Simple mean (NumPy only)
E = aggregate(Z, G_all, method="mean_pool")

# 2) PyG global pooling (needs torch + torch_geometric)
E_add = aggregate(Z, G_all, method="add_pool")
E_max = aggregate(Z, G_all, method="max_pool")
E_s2s = aggregate(Z, G_all, method="set2set", set2set_steps=3, set2set_layers=1)  # shape (B, 2d)
E_attn = aggregate(Z, G_all, method="global_attention", attn_hidden=128)

# 3) MIL pooling
E_mil_mean = aggregate(Z, G_all, method="mil_mean")
E_mil_max  = aggregate(Z, G_all, method="mil_max")
E_mil_attn = aggregate(Z, G_all, method="mil_attention", attn_hidden=128)

# 4) If you already have a PyG-style batch vector:
#    batch[i] = integer ROI index for node i
E_soft = aggregate(Z, G=None, method="softmax_pool", batch=batch, softmax_t=1.0, return_group_ids=True)

'''
