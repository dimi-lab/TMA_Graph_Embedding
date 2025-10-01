# graph_embeddings.py
from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Sequence, Any, Union
import numpy as np

# --------------------------- Optional heavy deps ---------------------------
def _require_torch():
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except Exception as e:
        raise ImportError(
            "This module requires PyTorch and PyTorch Geometric. "
            "Please install torch, torch_scatter/torch_sparse (per CUDA), and torch_geometric."
        ) from e


# --------------------------- Helpers ---------------------------
def _metadata_from_schema(
    edge_index_dict: Dict[Tuple[str, str, str], "Tensor"],
    num_nodes_dict: Dict[str, int],
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    node_types = list(num_nodes_dict.keys())
    edge_types = list(edge_index_dict.keys())
    return node_types, edge_types


def _device(device: Optional[str] = None):
    import torch
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _project_in_channels_if_needed(
    x_dict: Dict[str, "Tensor"],
    in_channels: int,
    device
) -> Tuple[Dict[str, "Tensor"], "torch.nn.ModuleDict"]:
    """
    Ensure every node type has feature dim == in_channels by adding a Linear
    projection when needed.
    """
    import torch
    import torch.nn as nn

    proj = nn.ModuleDict()
    x_proj = {}
    for ntype, x in x_dict.items():
        x = x.to(device)
        fdim = x.size(-1)
        if fdim != in_channels:
            layer = nn.Linear(fdim, in_channels, bias=True).to(device)
            proj[ntype] = layer
            x_proj[ntype] = layer(x)
        else:
            x_proj[ntype] = x
    return x_proj, proj


# --------------------------- HAN Encoder ---------------------------
class HANEncoder(torch.nn.Module):
    """
    Two-layer HAN using PyG's HANConv blocks.
    - Input: x_dict, edge_index_dict
    - Output: x_dict with updated embeddings (per node type)
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        metapaths: Optional[List[List[Tuple[str, str, str]]]] = None,
        in_channels: int = 64,
        hidden_channels: int = 64,
        out_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        from torch_geometric.nn import HANConv
        import torch.nn as nn

        self.metadata = metadata
        self.metapaths = metapaths  # optional; passed to conv forward if supported
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        # Layer 1
        self.conv1 = HANConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=heads,
            dropout=dropout,
        )
        # Layer 2
        self.conv2 = HANConv(
            in_channels=hidden_channels * heads,  # HANConv concatenates heads
            out_channels=out_channels,
            metadata=metadata,
            heads=heads,
            dropout=dropout,
        )

        self.act = nn.ReLU()
        self.dp = nn.Dropout(p=dropout)

    def _forward_conv(self, conv, x_dict, edge_index_dict):
        """Call HANConv while being robust to PyG version (metapaths arg)."""
        try:
            # Some versions accept metapaths positional
            if self.metapaths is not None:
                return conv(x_dict, edge_index_dict, self.metapaths)
            else:
                return conv(x_dict, edge_index_dict)
        except TypeError:
            # Others use kwarg
            if self.metapaths is not None:
                return conv(x_dict, edge_index_dict, metapaths=self.metapaths)
            else:
                return conv(x_dict, edge_index_dict)

    def forward(self, x_dict, edge_index_dict):
        x1 = self._forward_conv(self.conv1, x_dict, edge_index_dict)
        # Apply nonlinearity+dropout per node type
        for k in x1:
            x1[k] = self.dp(self.act(x1[k]))

        x2 = self._forward_conv(self.conv2, x1, edge_index_dict)
        # (Optionally) nonlinearity; for an encoder, many leave last layer raw
        return x2


# --------------------------- Public API ---------------------------
def graph_embedding(
    method: str = "HAN",
    *,
    # Hetero schema (REQUIRED for HAN)
    edge_index_dict: Dict[Tuple[str, str, str], "Tensor"],
    num_nodes_dict: Dict[str, int],
    # Node features (RECOMMENDED): dict of tensors keyed by node type
    x_dict: Optional[Dict[str, "Tensor"]] = None,
    # HAN-specific
    metapaths: Optional[List[List[Tuple[str, str, str]]]] = None,
    in_channels: int = 64,
    hidden_channels: int = 64,
    out_channels: int = 64,
    heads: int = 4,
    dropout: float = 0.2,
    # Generic
    return_node_type: Optional[str] = None,
    device: Optional[str] = None,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute node embeddings for a heterogeneous graph.

    Parameters
    ----------
    method : {"HAN", "protGNN"}
        Graph representation method. (Currently HAN implemented; protGNN is stubbed.)
    edge_index_dict : dict[(src, rel, dst) -> LongTensor[2, E]]
    num_nodes_dict : dict[node_type -> int]
    x_dict : optional dict[node_type -> FloatTensor[N_type, F_in]]
        Provide per-type node features. (Recommended: feed structure embeddings here.)
        If omitted, you must set `in_channels` to the intended feature size and
        pre-project your features before calling this function.
    metapaths : list[list[edge_type_triplet]], optional
        Each metapath is a list of edge-type triples, e.g.,
        [(A,"rel1",B), (B,"rel2",C), (C,"rel3",A)]
    in_channels / hidden_channels / out_channels / heads / dropout : HAN hyperparams
    return_node_type : str, optional
        If set, return only embeddings for this node type as np.ndarray (N_type, D).
        If None, return dict[node_type -> np.ndarray].
    device : "cpu" | "cuda" | None

    Returns
    -------
    np.ndarray or Dict[str, np.ndarray]
    """
    _require_torch()
    import torch

    method = (method or "HAN").upper()
    device_t = _device(device)
    metadata = _metadata_from_schema(edge_index_dict, num_nodes_dict)

    if method == "HANGNN" or method == "HAN":
        if x_dict is None:
            raise ValueError(
                "x_dict is required for HAN. Provide per-type node features (e.g., your structure embeddings)."
            )

        # Ensure all node types are present; if a type is missing in x_dict, init zeros
        x_dict_full: Dict[str, torch.Tensor] = {}
        for ntype, n in num_nodes_dict.items():
            if ntype in x_dict:
                x_nt = x_dict[ntype]
                if isinstance(x_nt, np.ndarray):
                    x_nt = torch.from_numpy(x_nt)
                x_nt = x_nt.to(device_t).float()
                x_dict_full[ntype] = x_nt
            else:
                # Create zero features as placeholder if not provided
                x_dict_full[ntype] = torch.zeros((n, in_channels), device=device_t)

        # Unify input dims to in_channels
        # (If some types have different F, project them)
        x_dict_full, proj = _project_in_channels_if_needed(x_dict_full, in_channels, device_t)

        # Build model
        model = HANEncoder(
            metadata=metadata,
            metapaths=metapaths,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
        ).to(device_t)

        model.eval()
        with torch.no_grad():
            x_out = model(x_dict_full, edge_index_dict)

        if return_node_type is not None:
            if return_node_type not in x_out:
                raise KeyError(f"return_node_type='{return_node_type}' not found in output types: {list(x_out.keys())}")
            return x_out[return_node_type].detach().cpu().numpy()
        else:
            return {k: v.detach().cpu().numpy() for k, v in x_out.items()}

    elif method == "PROTGNN" or method == "PROT-GNN" or method == "PROT_GNN":
        # Placeholder: Ready for future extension.
        raise NotImplementedError(
            "protGNN is not implemented yet. "
            "If you can share your target paper/variant (e.g., ProtoGNN/Prototypical GNN), "
            "I can wire it in with a matching API."
        )
    else:
        raise NotImplementedError(f"Unknown method '{method}'. Supported: 'HAN', 'protGNN'.")


# --------------------------- Convenience: build from NetworkX ---------------------------
def hetero_from_networkx(
    G: "nx.MultiDiGraph",
    *,
    node_type_attr: str = "ntype",
    edge_type_attr: str = "etype",
    rel_default: str = "to",
) -> Tuple[
    Dict[Tuple[str, str, str], "torch.LongTensor"],
    Dict[str, int],
    Dict[str, "torch.Tensor"],
]:
    """
    Convert a typed NetworkX MultiDiGraph into PyG hetero dicts.
    Nodes must have an attribute `node_type_attr` (e.g., 'ntype').
    Edges must have an attribute `edge_type_attr` (e.g., 'etype'); relation defaults to `rel_default`.
    Returns: (edge_index_dict, num_nodes_dict, id_mapping_per_type as LongTensor mapping original->compact ids)
    """
    _require_torch()
    import torch
    import networkx as nx

    if not isinstance(G, nx.MultiDiGraph):
        raise TypeError("hetero_from_networkx expects a nx.MultiDiGraph with typed nodes/edges.")

    # Partition nodes by type and assign compact ids per type
    type_to_nodes: Dict[str, List[Any]] = {}
    for n, data in G.nodes(data=True):
        ntype = data.get(node_type_attr, None)
        if ntype is None:
            raise KeyError(f"Node {n} missing '{node_type_attr}'")
        type_to_nodes.setdefault(ntype, []).append(n)

    idmap_dict: Dict[str, Dict[Any, int]] = {}
    num_nodes_dict: Dict[str, int] = {}
    for ntype, nodes in type_to_nodes.items():
        idmap = {n: i for i, n in enumerate(nodes)}
        idmap_dict[ntype] = idmap
        num_nodes_dict[ntype] = len(nodes)

    # Build edge_index_dict
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for u, v, data in G.edges(data=True):
        su = G.nodes[u].get(node_type_attr)
        sv = G.nodes[v].get(node_type_attr)
        rel = data.get(edge_type_attr, rel_default)
        etype = (su, rel, sv)
        ui = idmap_dict[su][u]
        vi = idmap_dict[sv][v]
        if etype not in edge_index_dict:
            edge_index_dict[etype] = []
        edge_index_dict[etype].append((ui, vi))

    # Tensorize
    for etype, edges in edge_index_dict.items():
        if len(edges) == 0:
            edge_index_dict[etype] = torch.empty((2, 0), dtype=torch.long)
        else:
            ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index_dict[etype] = ei

    # Also return a dict of LongTensor id maps (optional utility)
    id_tensors = {t: torch.tensor([idmap_dict[t][n] for n in type_to_nodes[t]], dtype=torch.long)
                  for t in type_to_nodes}
    return edge_index_dict, num_nodes_dict, id_tensors
