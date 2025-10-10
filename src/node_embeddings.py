# node_embeddings.py
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Optional
import warnings

from scipy.sparse import csr_matrix
import scipy.sparse as sp

def node_embedding(G_all: nx.Graph, df_aligned: pd.DataFrame, node_params: Dict) -> np.ndarray:      
    fusion_mode = str(node_params.get("fusion_mode", "none")).lower()

    attr_method = str(node_params.get("attr_method", "passthrough")).lower()
    attr_cols = node_params.get("attr_cols", None)  # e.g., ["area","intensity"]
    X_attr = attribute_embedding(df_aligned, attr_method=attr_method, cols=attr_cols)

    structure_method = str(node_params.get("structure_method", "random")).lower()
    if node_params['structure_method'] == 'fastrp-het':
        assert fusion_mode == 'none', "fusion_mode must be none if structure_method is fastrp-het"
        node_params['X_attr'] = sp.csr_matrix(X_attr)
    Z = structure_embedding(G_all, method=structure_method, **(node_params or {}))
    Z = np.asarray(Z, dtype=np.float64)
    # concat feature-wise
    
    return fuse_embedding(Z, X_attr, fusion_mode)


def fuse_embedding(structure_emb,attr_emb=None,fusion_mode: str = "none") -> np.ndarray:
    fusion_mode = (fusion_mode or "none").lower()
    if fusion_mode == "none":
        return structure_emb
    elif fusion_mode == "concat":
        assert attr_emb is not None, "attr_emb must be provided if fusion_mode is concat"
        assert attr_emb.shape[0] == structure_emb.shape[0], "attr_emb and structure_emb must have the same number of rows"
        return np.hstack([attr_emb, structure_emb])
    elif fusion_mode == "concat_power":
        assert attr_emb is not None, "attr_emb must be provided if fusion_mode is concat_shuffle"
        assert attr_emb.shape[0] == structure_emb.shape[0], "attr_emb and structure_emb must have the same number of rows"
        raise NotImplementedError("concat_power is not implemented")
    elif fusion_mode == "concat_shuffle":
        assert attr_emb is not None, "attr_emb must be provided if fusion_mode is concat_shuffle"
        assert attr_emb.shape[0] == structure_emb.shape[0], "attr_emb and structure_emb must have the same number of rows"
        structure_emb = np.random.permutation(structure_emb)
        return np.hstack([attr_emb, structure_emb])
    else:
        raise NotImplementedError(f"Fusion mode not implemented: {fusion_mode}")

def attribute_embedding(df: pd.DataFrame, attr_method: str = "passthrough", cols: Optional[list]=None, **kwargs) -> np.ndarray:
    attr_method = (attr_method or "passthrough").lower()
    if attr_method == "passthrough":
        # one-hot as float right away
        phenos = pd.get_dummies(df["phenotype"], prefix="ph", dtype=float)

        # keep only requested extras, coerce to numeric
        if cols:
            extra = df[cols].copy()
            for c in extra.columns:
                extra[c] = pd.to_numeric(extra[c], errors="coerce")
        else:
            extra = pd.DataFrame(index=df.index)

        X = pd.concat([phenos, extra], axis=1).fillna(0.0)
        return X.to_numpy(dtype=float)
    raise NotImplementedError(f"Attribute embedding attr_method not implemented: {attr_method}")

   

def structure_embedding(
    G: nx.Graph,
    method: str = "fastrp",
    **params,
) -> np.ndarray:
    """
    Compute structure embeddings for graph `G`.

    Parameters (via **params):
      Common:
        - dim: int = 128

      fastRP / fame (unchanged from your original):
        - q: int = 3
        - weights: Optional[Sequence[float]]
        - projection_method: {'gaussian','sparse'} = 'gaussian'
        - input_matrix: {'adj','trans'} = 'adj'
        - normalization: bool = False
        - alpha: Optional[float] = None
        - weight: str = 'weight'


      node2vec (PyG):
        - walk_length: int = 80
        - context_size: int = 10
        - walks_per_node: int = 10
        - num_negative_samples: int = 1
        - p: float = 1.0
        - q: float = 1.0
        - sparse: bool = True
        - batch_size: int = 256
        - epochs: int = 5
        - lr: float = 0.01
        - device: {'cpu','cuda'} or torch.device; default = 'cuda' if available
        - num_workers: int = 0
        - seed: int = 42
        - undirected: bool = True   # add both (u,v) and (v,u) to edge_index if True

      metapath2vec (PyG):
        # Two ways to use:
        # (A) Homogeneous fallback (single node type)
        - node_type: str = 'n'
        - relation: str = 'to'
        - metapaths: Optional[List[Tuple[str,str,str]]] = None
            # Default -> [(node_type, relation, node_type)]
        - walk_length/context_size/walks_per_node/num_negative_samples/sparse/batch_size/epochs/lr/device/num_workers/seed: same meaning as node2vec

        # (B) Fully-typed heterogeneous graph:
        - edge_index_dict: Dict[Tuple[str,str,str], torch.LongTensor]  # required
        - num_nodes_dict: Dict[str, int]                               # required
        - metapaths: List[Tuple[str,str,str]]                          # required
        - return_node_type: str                                        # which type to return embeddings for

    Returns:
      np.ndarray of shape (|V|, dim) for the chosen method.
      For node2vec and homogeneous metapath2vec, rows align with list(G.nodes()) in that order.
      For heterogeneous metapath2vec, rows align with [0..num_nodes(return_node_type)-1] of that node type.
    """
    import numpy as np
    method = (method or "fastrp").lower()

    # -------------------- helpers --------------------
    def _nx_to_edge_index(graph: nx.Graph, nodelist=None, undirected: bool = True):
        import torch
        if nodelist is None:
            nodelist = list(graph.nodes())
        idx = {n: i for i, n in enumerate(nodelist)}
        edges = []
        for u, v in graph.edges():
            ui, vi = idx[u], idx[v]
            edges.append((ui, vi))
            if undirected:
                edges.append((vi, ui))
        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index, nodelist

    def _pick_device(dev=None):
        import torch
        if dev is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(dev) if isinstance(dev, str) else dev

    # -------------------- random  --------------------
    if method == "random":
        n = G.number_of_nodes()
        dim = int(params.get("dim", 32))
        rs = int(params.get("random_state", 0))
        rng = np.random.default_rng(rs)
        return rng.normal(size=(n, dim)).astype(float)

    # -------------------- fastRP / fame (unchanged core) --------------------
    elif method == "fastrp":
        from src.fastrp import fastrp_projection, fastrp_wrapper
        A, conf = prepare_fastrp(G, params)
        U = fastrp_wrapper(A, conf)
        return U
    elif method == 'fastrp-het':
        from src.fastrp import fastrp_projection, fastrp_wrapper
        
        A, conf = prepare_fastrp(G, params)
        U = fastrp_wrapper(A, conf)
        return U
    elif method == 'fame':
        from src.fame import fastrp_wrapper, fastrp_projection
        A, conf = prepare_fastrp(G, params)
        U = fastrp_wrapper(A, conf)
        return U

    # -------------------- node2vec (PyG) --------------------
    elif method in {"node2vec", "node2vector", "n2v"}:
        try:
            import torch
            from torch_geometric.nn import Node2Vec
        except Exception as e:
            raise ImportError(
                "PyTorch Geometric is required for method='node2vec'. "
                "Install PyG and torch-scatter/torch-sparse per your CUDA/PyTorch setup."
            ) from e

        dim = int(params.get("dim", 128))
        walk_length = int(params.get("walk_length", 80))
        context_size = int(params.get("context_size", 10))
        walks_per_node = int(params.get("walks_per_node", 10))
        num_negative_samples = int(params.get("num_negative_samples", 1))
        p = float(params.get("p", 1.0))
        q = float(params.get("q", 1.0))
        sparse = bool(params.get("sparse", True))
        batch_size = int(params.get("batch_size", 256))
        epochs = int(params.get("epochs", 5))
        lr = float(params.get("lr", 0.01))
        num_workers = int(params.get("num_workers", 0))
        seed = int(params.get("seed", 42))
        undirected = bool(params.get("undirected", True))
        device = _pick_device(params.get("device", None))

        torch.manual_seed(seed)
        edge_index, nodelist = _nx_to_edge_index(G, None, undirected=undirected)

        model = Node2Vec(
            edge_index=edge_index,
            embedding_dim=dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            sparse=sparse,
        ).to(device)

        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
        optimizer_cls = torch.optim.SparseAdam if sparse else torch.optim.Adam
        optimizer = optimizer_cls(list(model.parameters()), lr=lr)

        model.train()
        for _ in range(epochs):
            for pos_rw, neg_rw in loader:
                pos_rw = pos_rw.to(device)
                neg_rw = neg_rw.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            Z = model.embedding.weight.detach().cpu().numpy()

        # Align to list(G.nodes()) order:
        return Z

    # -------------------- metapath2vec (PyG) --------------------
    elif method in {
        "metapath2vec", "metapath2vector", "metpath2vec", "metpath2vector", "mp2v"
    }:
        try:
            import torch
            from torch_geometric.nn import MetaPath2Vec
        except Exception as e:
            raise ImportError(
                "PyTorch Geometric is required for method='metapath2vec'. "
            ) from e

        dim = int(params.get("dim", 128))
        walk_length = int(params.get("walk_length", 80))
        context_size = int(params.get("context_size", 10))
        walks_per_node = int(params.get("walks_per_node", 10))
        num_negative_samples = int(params.get("num_negative_samples", 1))
        sparse = bool(params.get("sparse", True))
        batch_size = int(params.get("batch_size", 256))
        epochs = int(params.get("epochs", 5))
        lr = float(params.get("lr", 0.01))
        num_workers = int(params.get("num_workers", 0))
        seed = int(params.get("seed", 42))
        device = _pick_device(params.get("device", None))

        torch.manual_seed(seed)

        # Two modes:
        edge_index_dict = params.get("edge_index_dict", None)
        num_nodes_dict = params.get("num_nodes_dict", None)
        metapaths = params.get("metapaths", None)
        return_node_type = params.get("return_node_type", None)

        if edge_index_dict is not None and num_nodes_dict is not None and metapaths is not None:
            # Heterogeneous mode (user-provided)
            model = MetaPath2Vec(
                edge_index_dict=edge_index_dict,
                embedding_dim=dim,
                metapaths=metapaths,
                walk_length=walk_length,
                context_size=context_size,
                walks_per_node=walks_per_node,
                num_negative_samples=num_negative_samples,
                sparse=sparse,
            ).to(device)

            loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
            optimizer_cls = torch.optim.SparseAdam if sparse else torch.optim.Adam
            optimizer = optimizer_cls(list(model.parameters()), lr=lr)

            model.train()
            for _ in range(epochs):
                for pos_rw, neg_rw in loader:
                    pos_rw = pos_rw.to(device)
                    neg_rw = neg_rw.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.loss(pos_rw, neg_rw)
                    loss.backward()
                    optimizer.step()

            # Which node type to return?
            if return_node_type is None:
                # Heuristic: return the first node type seen in metapaths
                if len(metapaths) == 0:
                    raise ValueError("metapaths is empty; cannot infer return_node_type.")
                return_node_type = metapaths[0][0]  # ('paper','cites','paper') -> 'paper'

            with torch.no_grad():
                Z = model(return_node_type).detach().cpu().numpy()
            return Z

        # Homogeneous fallback: wrap G as a single-type heterograph
        node_type = params.get("node_type", "n")
        relation = params.get("relation", "to")
        # Build edge_index for (node_type, relation, node_type)
        undirected = bool(params.get("undirected", True))
        edge_index, nodelist = _nx_to_edge_index(G, None, undirected=undirected)
        etype = (node_type, relation, node_type)

        if metapaths is None:
            # Default to simple 1-hop relation path
            metapaths = [etype]

        model = MetaPath2Vec(
            edge_index_dict={etype: edge_index},
            embedding_dim=dim,
            metapaths=metapaths,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            sparse=sparse,
        ).to(device)

        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
        optimizer_cls = torch.optim.SparseAdam if sparse else torch.optim.Adam
        optimizer = optimizer_cls(list(model.parameters()), lr=lr)

        model.train()
        for _ in range(epochs):
            for pos_rw, neg_rw in loader:
                pos_rw = pos_rw.to(device)
                neg_rw = neg_rw.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            Z = model(node_type).detach().cpu().numpy()
        return Z

    else:
        raise NotImplementedError(f"Unsupported method: {method}")


# -- fastrp helpers --

def prepare_fastrp(G,params):
    nodelist = G.nodes()
    try:
        A_nx = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=None, format="csr")
    except Exception:
        # For older NetworkX
        A_nx = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=None, format="csr")
    A = csr_matrix(A_nx)  # ensure scipy.sparse.csr_matrix

    conf = {
        'projection_method': params.get("projection_method", "gaussian"),
        'input_matrix': params.get("input_matrix", "adj"),
        'weights': params.get("weights", None), # weights for each power of A, hyper-parameter
        'normalization': params.get("normalization", False),
        'dim': params.get("dim", 512),
        'alpha': params.get("alpha", None), # Tukey hyper-parameter
        'C': params.get("C", 1.0), # Not sure what this is?
        'X_attr': params.get('X_attr', None),
    }
    
    return A, conf

def get_emb_filename(prefix, conf):
    return prefix + '-dim=' + str(conf['dim']) + ',projection_method=' + conf['projection_method'] \
        + ',input_matrix=' + conf['input_matrix'] + ',normalization=' + str(conf['normalization']) \
        + ',weights=' + (','.join(map(str, conf['weights'])) if conf['weights'] is not None else 'None') \
        + ',alpha=' + (str(conf['alpha']) if 'alpha' in conf else '') \
        + ',C=' + (str(conf['C']) if 'alpha' in conf else '1.0') \
        + '.mat'


