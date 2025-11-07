#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graphlet_miner.py
-----------------
Fast(ish) colored-graphlet feature extractor for ROI graphs.

It builds a sparse ROI × motif matrix by counting connected, induced
subgraphs of sizes k=3..4 (configurable), where motifs are canonicalized
by (structure + node-color multiset). Designed to be run outside notebooks.

Typical usage:
    python graphlet_miner.py \
        --graphs-pkl /path/to/graphs/graph_dict_gabriel.pkl \
        --df /path/to/dataframes/df.csv \
        --roi-col roi_id --subject-col subject_id --label-col roi_label \
        --type-col phenotype \
        --k 3 4 \
        --max-samples-per-k 6000 \
        --mode combos \
        --normalize by_total \
        --n-jobs 8 \
        --outdir /path/to/output/evaluate/graphlet_mining

Outputs in OUTDIR:
    - X_graphlets.npz      : scipy.sparse CSR matrix (n_rois × n_motifs)
    - roi_list.json        : list of ROI ids in row order
    - vocab.json           : list of motif-hash tokens in column order
    - token_meta.json      : dict: token -> {"k": int, "adj": str, "colors": [..]}
    - roi_meta.csv         : ROI metadata (roi, y, subject)
"""

import argparse, json, os, sys, math, random, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix, save_npz
from joblib import Parallel, delayed

# --- Add near the top of graphlet_miner.py ---
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

try:
    from tqdm import tqdm
except Exception:
    # minimal fallback
    def tqdm(x, **kwargs): return x


# -----------------------------
# Canonical motif hashing
# -----------------------------

def _adj_signature(G: nx.Graph, nodes: List) -> str:
    """Upper-tri adjacency string for induced subgraph over 'nodes' order."""
    idx = {n:i for i,n in enumerate(nodes)}
    bits = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            bits.append('1' if G.has_edge(nodes[i], nodes[j]) else '0')
    return ''.join(bits)


def _color_multiset(G: nx.Graph, nodes: List, attr: str = "label") -> Tuple[str, ...]:
    """Sorted multiset (tuple) of node labels for given nodes."""
    return tuple(sorted([str(G.nodes[n].get(attr, "UNK")) for n in nodes]))


def canonical_token(G: nx.Graph, nodes: List, attr: str = "label") -> Tuple[str, Dict]:
    """
    Returns (token, meta) where token is SHA1 hash for (k|adj|colors)
    and meta contains the unpacked pieces for later interpretability.
    """
    k = len(nodes)
    adj = _adj_signature(G, nodes)
    cols = _color_multiset(G, nodes, attr=attr)
    payload = f"{k}|{adj}|{','.join(cols)}"
    tok = hashlib.sha1(payload.encode()).hexdigest()
    meta = {"k": k, "adj": adj, "colors": list(cols)}
    return tok, meta


# -----------------------------
# Subgraph samplers
# -----------------------------

def _connected_combos(G: nx.Graph, k: int, max_samples: Optional[int] = None) -> Iterable[List]:
    """
    Enumerate (or sample) k-combinations of nodes that are connected when induced.
    If max_samples is given, we sample uniformly from all combinations (approx).
    WARNING: combinations explode for large graphs; use with care.
    """
    import itertools as it
    nodes = list(G.nodes())
    if len(nodes) < k:
        return []

    combos = it.combinations(nodes, k)
    if max_samples is not None:
        combos = list(it.combinations(nodes, k))
        if len(combos) > max_samples:
            combos = random.sample(combos, max_samples)

    out = []
    for tup in combos:
        H = G.subgraph(tup)
        if nx.is_connected(H):
            out.append(list(tup))
    return out


def _ego_sampler(G: nx.Graph, k: int, samples: int = 5000, seed: int = 0) -> Iterable[List]:
    """
    Faster sampler that builds connected k-node sets by random BFS expansions.
    Good default when graphs are large (thousands of nodes).
    """
    if len(G) < k:
        return []

    rng = random.Random(seed)
    nodes = list(G.nodes())
    out = []
    for _ in range(samples):
        start = rng.choice(nodes)
        sub = [start]
        frontier = set(G.neighbors(start))
        while len(sub) < k and frontier:
            nxt = rng.choice(list(frontier))
            sub.append(nxt)
            frontier.update(G.neighbors(nxt))
            frontier.difference_update(sub)
        if len(sub) == k:
            # ensure induced connectivity (should be, but check)
            H = G.subgraph(sub)
            if nx.is_connected(H):
                out.append(sub)
    return out


# -----------------------------
# Counting per-ROI
# -----------------------------

def count_graphlets_for_roi(roi: str,
                            G: nx.Graph,
                            k_list: List[int],
                            mode: str = "ego",
                            max_samples_per_k: Optional[int] = 6000,
                            label_attr: str = "label",
                            seed: int = 0) -> Tuple[str, Counter, Dict[str, Dict]]:
    """
    Returns (roi, Counter[token->count], token_meta_map)
    token_meta_map collects meta for tokens seen in this ROI.
    """
    rng = random.Random(seed + hash(roi) % (10**6))
    counts = Counter()
    token_meta: Dict[str, Dict] = {}

    for k in k_list:
        if mode == "combos":
            subsets = _connected_combos(G, k, max_samples=max_samples_per_k)
        else:
            # ego-sampler default
            subsets = _ego_sampler(G, k, samples=(max_samples_per_k or 0) or 6000, seed=rng.randint(0, 10**9))

        for nodes in subsets:
            tok, meta = canonical_token(G, nodes, attr=label_attr)
            counts[tok] += 1
            if tok not in token_meta:
                token_meta[tok] = meta

    return roi, counts, token_meta


# -----------------------------
# I/O helpers
# -----------------------------

def load_graph_dict(graphs_pkl: Path) -> Dict[str, nx.Graph]:
    with open(graphs_pkl, "rb") as f:
        graph_dict = __import__("pickle").load(f)
    if not isinstance(graph_dict, dict):
        raise ValueError("Expected a dict[roi] -> nx.Graph in graphs_pkl")
    return graph_dict


def attach_node_labels(graph_dict: Dict[str, nx.Graph],
                       df: pd.DataFrame,
                       roi_col: str,
                       type_col: str) -> None:
    # Build per-ROI map of cell_id -> type
    needed = {roi_col, "cell_id", type_col}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"df is missing columns: {miss}")

    by_roi = df[[roi_col, "cell_id", type_col]].dropna()
    for roi, sub in by_roi.groupby(roi_col):
        cmap = dict(zip(sub["cell_id"].astype(int).values, sub[type_col].astype(str).values))
        G = graph_dict.get(str(roi))
        if G is None:
            continue
        nx.set_node_attributes(G, {n: cmap.get(int(n), 'UNK') for n in G.nodes()}, name="label")


def build_roi_targets(df: pd.DataFrame,
                      roi_col: str,
                      subject_col: str,
                      label_col: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    roi_meta = (
        df[[roi_col, subject_col, label_col]]
        .dropna()
        .drop_duplicates(subset=[roi_col])
    )
    rois = roi_meta[roi_col].astype(str).tolist()
    y = roi_meta[label_col].astype(int).values
    groups = roi_meta[subject_col].astype(str).values
    return rois, y, groups


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Colored-graphlet miner for ROI graphs")
    ap.add_argument("--graphs-pkl", type=Path, required=True, help="Pickle of dict[roi]->nx.Graph")
    ap.add_argument("--df", type=Path, required=True, help="CSV with ROI/cell/node-type and labels")
    ap.add_argument("--roi-col", type=str, default="roi_id")
    ap.add_argument("--subject-col", type=str, default="subject_id")
    ap.add_argument("--label-col", type=str, default="roi_label")
    ap.add_argument("--type-col", type=str, default="phenotype")
    ap.add_argument("--k", type=int, nargs="+", default=[3,4], help="Subgraph sizes to mine")
    ap.add_argument("--mode", type=str, choices=["ego","combos"], default="ego",
                    help="Sampler: 'ego' (fast) or 'combos' (uniform combos; slow)")
    ap.add_argument("--max-samples-per-k", type=int, default=6000,
                    help="Samples per k (per ROI). For 'combos', caps combinations; for 'ego', num ego-samples.")
    ap.add_argument("--normalize", type=str, choices=["by_total","by_nodes","none"], default="by_total",
                    help="Row-wise normalization: counts/total, counts/|V|, or none")
    ap.add_argument("--n-jobs", type=int, default=8, help="Parallel jobs across ROIs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info(f"[*] Loading graphs: {args.graphs_pkl}")
    graph_dict = load_graph_dict(args.graphs_pkl)

    logging.info(f"[*] Loading df: {args.df}")
    df = pd.read_csv(args.df)

    # Minimal checks
    for col in [args.roi_col, args.subject_col, args.label_col, "cell_id", args.type_col]:
        if col not in df.columns:
            raise ValueError(f"df is missing required column: {col}")

    # Attach node labels
    logging.info("[*] Attaching node labels to graphs...")
    attach_node_labels(graph_dict, df, roi_col=args.roi_col, type_col=args.type_col)

    # Build ROI targets & filter to ROIs that exist in graph_dict
    logging.info("[*] Building ROI targets...")
    roi_list, y, groups = build_roi_targets(df, args.roi_col, args.subject_col, args.label_col)
    roi_list = [r for r in roi_list if str(r) in graph_dict]
    y = np.array([y[i] for i, r in enumerate(roi_list)], dtype=int) if len(y)==len(roi_list) else y
    groups = np.array([groups[i] for i, r in enumerate(roi_list)], dtype=str) if len(groups)==len(roi_list) else groups
    logging.info(f"[*] ROIs to process: {len(roi_list)}")

    # Count graphlets in parallel
    logging.info(f"[*] Mining motifs with mode={args.mode}, k={args.k}, max_samples_per_k={args.max_samples_per_k} ...")
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(count_graphlets_for_roi)(
            roi=str(roi),
            G=graph_dict[str(roi)],
            k_list=args.k,
            mode=args.mode,
            max_samples_per_k=args.max_samples_per_k,
            label_attr="label",
            seed=args.seed
        )
        for roi in roi_list
    )

    # Aggregate counts and token meta
    vocab_set = set()
    roi_to_counts: Dict[str, Counter] = {}
    token_meta: Dict[str, Dict] = {}
    for roi, C, meta in results:
        roi_to_counts[roi] = C
        vocab_set.update(C.keys())
        for t, m in meta.items():
            if t not in token_meta:
                token_meta[t] = m

    vocab = sorted(vocab_set)
    tok2idx = {t:i for i,t in enumerate(vocab)}
    logging.info(f"[*] Unique motifs: {len(vocab)}")

    # Build sparse matrix
    rows, cols, vals = [], [], []
    node_counts = {roi: graph_dict[str(roi)].number_of_nodes() for roi in roi_list}
    for r_idx, roi in enumerate(roi_list):
        C = roi_to_counts.get(roi, {})
        total = float(sum(C.values())) if C else 1.0
        denom = 1.0
        if args.normalize == "by_total":
            denom = total
        elif args.normalize == "by_nodes":
            denom = float(node_counts[roi]) if node_counts[roi] > 0 else 1.0
        else:
            denom = 1.0

        for t, c in C.items():
            rows.append(r_idx)
            cols.append(tok2idx[t])
            vals.append(c / (denom if denom > 0 else 1.0))

    X = csr_matrix((vals, (rows, cols)), shape=(len(roi_list), len(vocab)))
    logging.info(f"[*] Feature matrix: {X.shape}  nnz={X.nnz}  density={X.nnz/(X.shape[0]*X.shape[1]+1e-9):.3e}")

    # Save artifacts
    save_npz(outdir / "X_graphlets.npz", X)
    (outdir / "roi_list.json").write_text(json.dumps(roi_list))
    (outdir / "vocab.json").write_text(json.dumps(vocab))
    (outdir / "token_meta.json").write_text(json.dumps(token_meta, indent=2))
    roi_meta = pd.DataFrame({"roi": roi_list, "y": y, "group": groups})
    roi_meta.to_csv(outdir / "roi_meta.csv", index=False)

    logging.info("[*] Saved:")
    logging.info(f"   - {outdir / 'X_graphlets.npz'}")
    logging.info(f"   - {outdir / 'roi_list.json'}")
    logging.info(f"   - {outdir / 'vocab.json'}")
    logging.info(f"   - {outdir / 'token_meta.json'}")
    logging.info(f"   - {outdir / 'roi_meta.csv'}")
    logging.info("[*] Done.")

if __name__ == "__main__":
    main()
