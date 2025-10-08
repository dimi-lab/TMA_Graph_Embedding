"""
# at the root of the project, example command:
python main.py --config config/demo.yaml --outdir /projects/wangc/m344313/OVTMA_project/output/demo
python main.py --config config/OVTMA_fov297_fastrp.yaml --outdir /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp &
python main.py --config config/OVTMA_fov297_fastrp_het.yaml --outdir /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp_het &
CUDA_VISIBLE_DEVICES=1 python main.py --config config/OVTMA_fov216_fastrp.yaml --outdir /projects/wangc/m344313/OVTMA_project/output/fov216_fastrp &
CUDA_VISIBLE_DEVICES=2 python main.py --config config/OVTMA_fov216_fastrp_het.yaml --outdir /projects/wangc/m344313/OVTMA_project/output/fov216_fastrp_het &

main.py — Orchestrates BMS mxIF structure-embedding search and exports.

Steps:
1) Load config & data; attach ROI/subject labels.
2) Build per-ROI graphs and a disconnected-union graph G_all.
3) SPECIAL CASE: compute basis embeddings for alpha=-0.5 with weights=[1]*10,
   save basis_i .mat and their UMAP(2D) as .mat.
4) Optuna search over alpha in [-1.0, 0.0] and per-order weights (log scale),
   maximizing structure_score.
5) Save: df, graph_dict, G_all, trial logs, best params, best embedding.

Outputs (under --outdir):
- data/df.csv
- graphs/graph_dict_<type>.pkl, graphs/G_all_<type>.pkl
- basis_embedding/alpha_-0.5/basis_{i}.mat and .../umap/basis_{i}.mat
- logs/run.log, logs/optuna_trials.csv, logs/best_params.yaml
- embeddings/best/struct_embedding.mat, embeddings/best/basis_list.pkl
- config/resolved_config.yaml (copy of the used config)
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import networkx as nx
from scipy.io import savemat
from scipy.sparse import issparse  
import scipy.sparse as sp
import pdb

from src.utils import align_df_with_G_all
from src.supervised_fit import SupervisedSearchConfig, supervised_search
from src.aggregation import aggregate

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# ---- Project imports (allow user to point to project root for src/)
def add_to_syspath(path: Path):
    import sys
    p = str(path.resolve())
    if p not in sys.path:
        sys.path.append(p)


def import_project_modules():
    from src.data_io import (
        read_cells_rds, read_roi_labels_csv, read_subject_labels_csv, attach_labels
    )
    from src.graph_builder import build_graph
    from src.node_embeddings import node_embedding
    from src.stats import basic_graph_metrics
    # Visualization imports not used in main run:
    # from src.viz import plot_cells, plot_graph
    return {
        "read_cells_rds": read_cells_rds,
        "read_roi_labels_csv": read_roi_labels_csv,
        "read_subject_labels_csv": read_subject_labels_csv,
        "attach_labels": attach_labels,
        "build_graph": build_graph,
        "node_embedding": node_embedding,
        "basic_graph_metrics": basic_graph_metrics,
    }

# ---- UMAP helper
def compute_umap(
    X,
    *,
    n_components: int = 2,
    random_state: int = 42,
    # Speed knobs (all optional)
    pca_dim: int | None = None,          # e.g., 50
    sample_fit_size: int | None = None,  # e.g., 120_000
    y: np.ndarray | None = None,         # optional labels for stratified sampling
    metric: str = "euclidean",
    n_neighbors: int = 15,
    n_epochs: int | None = None,         # e.g., 150–200 for speed; None = library default
    negative_sample_rate: int = 5,       # 2–3 is faster; 5 is default
    init: str = "spectral",              # "random" is faster
    backend: str = "auto",               # "auto" | "cpu" | "gpu"
    verbose: bool = False,
):
    """
    Accelerated UMAP with optional PCA pre-reduction, subsample fit + full transform,
    sparse-aware handling, and optional GPU via RAPIDS cuML.
    """
    import numpy as _np
    from scipy.sparse import issparse as _issparse

    # 0) Optional PCA pre-reduction (keeps CSR sparse until PCA)
    X_input = X
    if pca_dim is not None:
        # PCA requires dense; do a light randomized SVD PCA on a subsample to get components, then transform all
        from sklearn.decomposition import PCA
        # Small subsample just to fit PCA quickly (if huge); adjust as you like
        _fit_rows = min(200_000, X_input.shape[0])
        _idx = _np.random.default_rng(random_state).choice(X_input.shape[0], _fit_rows, replace=False)
        X_fit = X_input[_idx]
        if _issparse(X_fit):
            X_fit = X_fit.toarray()
        pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=random_state)
        pca.fit(X_fit)
        if _issparse(X_input):
            X_pca = pca.transform(X_input.toarray())
        else:
            X_pca = pca.transform(X_input)
        X_use = X_pca
    else:
        X_use = X_input  # sparse or dense OK

    # 1) GPU path (optional)
    if backend in ("auto", "gpu"):
        try:
            import cupy as cp  # noqa: F401
            from cuml.manifold import UMAP as cuUMAP
            # cuML expects dense (cupy) arrays; convert if feasible
            if _issparse(X_use):
                X_dense = X_use.toarray()
            else:
                X_dense = _np.asarray(X_use)
            X_gpu = cp.asarray(X_dense)

            um = cuUMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                n_epochs=n_epochs,
                init=init,
                metric=metric,
                random_state=random_state,
                negative_sample_rate=negative_sample_rate,
                verbose=verbose,
            )

            if sample_fit_size is not None and sample_fit_size < X_gpu.shape[0]:
                rng = _np.random.default_rng(random_state)
                idx_fit = rng.choice(X_gpu.shape[0], sample_fit_size, replace=False)
                um.fit(X_gpu[idx_fit])
                U = um.transform(X_gpu)  # map all 290k
            else:
                U = um.fit_transform(X_gpu)

            return cp.asnumpy(U)

        except Exception:
            if backend == "gpu":
                raise  # user explicitly requested GPU; surface the error
            # else fall through to CPU
            pass

    # 2) CPU path (umap-learn); supports sparse input
    import umap
    # Optional: control numba threads
    try:
        import numba
        # Respect env NUMBA_NUM_THREADS if set; otherwise leave default
        # numba.set_num_threads(k)  # uncomment to force threads programmatically
    except Exception:
        pass

    um = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        n_epochs=n_epochs,
        init=init,
        metric=metric,
        random_state=random_state,
        negative_sample_rate=negative_sample_rate,
        densmap=False,
        verbose=verbose,
        low_memory=True,  # reduces peak RAM
    )

    # Subsample fit, then transform everything (fast, memory-friendly)
    if sample_fit_size is not None and sample_fit_size < X_use.shape[0]:
        rng = _np.random.default_rng(random_state)
        if y is None:
            idx_fit = rng.choice(X_use.shape[0], sample_fit_size, replace=False)
        else:
            # Stratified sampling by labels y
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_fit_size, random_state=random_state)
            idx_fit, _ = next(sss.split(_np.zeros(len(y)), y))
        X_fit = X_use[idx_fit]
        um.fit(X_fit)            # build embedding on the sample
        U = um.transform(X_use)  # map all points into that space
    else:
        U = um.fit_transform(X_use)

    return U


# ---- IO helpers
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_mat(path: Path, data: Dict[str, np.ndarray]):
    ensure_dir(path.parent)
    savemat(str(path), data)

def save_pickle(path: Path, obj):
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_yaml(path: Path, obj):
    ensure_dir(path.parent)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

# ---- Sparse/dense helpers (NEW)
def _to_dense(x):
    """Convert SciPy sparse -> dense ndarray; keep ndarray as-is."""
    return x.toarray() if issparse(x) else np.asarray(x)

def save_mat_dense_or_sparse(path: Path, arr, key: str = "Z", force_dense: bool = False):
    """
    Save SciPy sparse as MATLAB sparse (CSC) when possible; otherwise save dense.
    SciPy's savemat understands CSR/CSC/COO; CSC is a safe choice.
    """
    ensure_dir(path.parent)
    if issparse(arr) and not force_dense:
        savemat(str(path), {key: arr.tocsc()})
    else:
        savemat(str(path), {key: _to_dense(arr)})

# ---- Config dataclass (minimal schema used here)
@dataclass
class RunConfig:
    cfg_path: Path
    outdir: Path
    n_trials: int
    base: Path | None = None  # where `src/` lives; default: project root (parent of main.py)
    seed: int = 42
    override: bool = False  # recompute node embeddings even if cached files exist
# ---- Logging setup
def setup_logging(log_dir: Path):
    ensure_dir(log_dir)
    #log_file = log_dir / "run.log"
    # log should be unique name with timestamp
    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")
    return log_file

# ---- Build disconnected union graph (no node-order tracking)  (UPDATED)
def build_union_graph(graph_dict: Dict[str, nx.Graph]) -> nx.Graph:
    G_all = nx.Graph()
    # Relabel each ROI's graph with (roi, node_id) to keep components disjoint
    for roi, G in graph_dict.items():
        H = nx.relabel_nodes(G, lambda n, r=roi: (r, n))
        G_all.update(H)
    return G_all


# ---- Main run
def main():
    parser = argparse.ArgumentParser(description="Run structure-embedding workflow.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--base", type=Path, default=None,
                        help="Project root containing src/. If omitted, uses script parent.")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--override", action="store_true", default=False,
                    help="Recompute node embeddings even if cached files exist.")
    args = parser.parse_args()

    run_cfg = RunConfig(
        cfg_path=args.config, outdir=args.outdir, n_trials=args.n_trials,
        base=args.base, seed=args.seed, override=args.override
    )

    # Prepare output directories
    ensure_dir(run_cfg.outdir)
    _ = setup_logging(run_cfg.outdir / "logs")
    set_seeds(run_cfg.seed)

    # Make sure we can import project modules
    project_root = run_cfg.base if run_cfg.base is not None else Path(__file__).resolve().parent
    add_to_syspath(project_root)
    modules = import_project_modules()

    # Load config
    with open(run_cfg.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Persist a resolved copy of the config used for the run
    save_yaml(run_cfg.outdir / "config" / "resolved_config.yaml", cfg)


    # === Load data ===
    if os.path.exists(run_cfg.outdir / "data" / "df.csv") and not run_cfg.override:
        logging.info("Loading cached data...")
        df = pd.read_csv(run_cfg.outdir / "data" / "df.csv")
        logging.info(f"Loaded df.csv with shape {df.shape}")
    else:   
        logging.info("Loading data...")
        paths = cfg["paths"]
        cols_cell = cfg["cell_columns"]
        cols_roi = cfg["roi_label_columns"]
        cols_subj = cfg.get("subject_label_columns", None)

        BASE = Path(paths["data_dir"])
        cells = modules["read_cells_rds"](BASE / paths["cell_rds"], cols_cell)
        roi_labels = modules["read_roi_labels_csv"](BASE / paths["roi_labels_csv"], cols_roi)
        if paths["subject_labels_csv"] is not None:
            assert cols_subj is not None, "subject_labels_csv is provided but subject_label_columns is not provided"
            subject_labels = modules["read_subject_labels_csv"](BASE / paths["subject_labels_csv"], cols_subj)
        else:
            subject_labels = None
        df = modules["attach_labels"](cells, roi_labels, subject_labels)

        # Save df
        ensure_dir(run_cfg.outdir / "dataframes")
        df.to_csv(run_cfg.outdir / "dataframes" / "df.csv", index=False)
        logging.info(f"Saved df.csv with shape {df.shape}")

    # === Build per-ROI graphs ===
    gcfg = cfg["graph"]
    if os.path.exists(run_cfg.outdir / "graphs" / f"graph_dict_{gcfg['type']}.pkl") and os.path.exists(run_cfg.outdir / "graphs" / f"G_all_{gcfg['type']}.pkl") and not run_cfg.override:
        logging.info("Loading cached graphs...")
        with open(run_cfg.outdir / "graphs" / f"graph_dict_{gcfg['type']}.pkl", "rb") as f:
            graph_dict = pickle.load(f)
        logging.info(f"Loaded graph_dict_{gcfg['type']}.pkl with {len(graph_dict)} ROI graphs")
        with open(run_cfg.outdir / "graphs" / f"G_all_{gcfg['type']}.pkl", "rb") as f:
            G_all = pickle.load(f)
        logging.info(f"Loaded G_all_{gcfg['type']}.pkl with {G_all.number_of_nodes()} nodes and {G_all.number_of_edges()} edges")
    else:
        logging.info("Building per-ROI graphs...")
        gcfg = cfg["graph"]
        graph_dict = {}
        for roi, df_roi in df.groupby("ROI"):
            G = modules["build_graph"](
                df_roi,
                kind=gcfg["type"],
                k=gcfg.get("knn_k", 8),
                radius=gcfg.get("radius", 25.0),
            )
            # Optionally compute & log basic metrics
            try:
                metrics = modules["basic_graph_metrics"](G)
                logging.info(f"[{roi}] Graph metrics: {metrics}")
            except Exception:
                pass
            graph_dict[roi] = G

        # Save graphs
        graphs_dir = run_cfg.outdir / "graphs"
        ensure_dir(graphs_dir)
        with open(graphs_dir / f"graph_dict_{gcfg['type']}.pkl", "wb") as f:
            pickle.dump(graph_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Saved graph_dict_{gcfg['type']}.pkl with {len(graph_dict)} ROI graphs")

        # === Build disconnected union graph (no node-order tracking) ===
        G_all = build_union_graph(graph_dict)
        with open(graphs_dir / f"G_all_{gcfg['type']}.pkl", "wb") as f:
            pickle.dump(G_all, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Saved G_all_{gcfg['type']}.pkl with {G_all.number_of_nodes()} nodes and {G_all.number_of_edges()} edges")

    df_aligned = align_df_with_G_all(df, G_all) # reordered df with the same order as G_all
    
    # === Supervised ROI-level search over node embeddings & aggregation ===
    sup_cfg = cfg.get("roi_supervision", None)
    if sup_cfg and sup_cfg.get("enabled", False):
        logging.info("Starting supervised ROI-level search (node2vec/metapath2vec + aggregation)...")

        # Build config object
        ss_cfg = SupervisedSearchConfig(
            params_fixed=sup_cfg.get("params_fixed", {}) or {},
            params_search_space=sup_cfg.get("params_search_space", {}) or {},
            aggregation_choices=sup_cfg.get("aggregation_choices", ["mean_pool"]),
            label_col='roi_label',
            group_col=sup_cfg.get("group_col", None),
            n_splits=int(sup_cfg.get("n_splits", 5)),
            random_state=int(sup_cfg.get("random_state", 42)),
            cache_dir=run_cfg.outdir / "cache" / "node_embeddings",
            override=run_cfg.override,
        )

        best_score, best_node_params, best_meta, best_clf_list = supervised_search(G_all, df_aligned, ss_cfg)
        logging.info(f"[ROI supervision] best_score={best_score:.6f}")
        logging.info(f"[ROI supervision] best_node_params={best_node_params}")
        logging.info(f"[ROI supervision] best_meta={best_meta}")

        # Refit once on full data with best params and export ROI embeddings
        Z_nodes = modules["node_embedding"](G_all, df_aligned, best_node_params)


        E_roi, group_ids = aggregate(Z_nodes, G=G_all, method=best_meta["aggr_method"], return_group_ids=True)

        # Save outputs
        roi_emb_dir = run_cfg.outdir / "evaluate" / "roi_supervised_best"
        ensure_dir(roi_emb_dir)
        savemat(str(roi_emb_dir / "roi_embedding.mat"), {"E": E_roi})
        save_pickle(roi_emb_dir / "group_ids.pkl", group_ids)
        save_pickle(roi_emb_dir / "best_clf_list.pkl", best_clf_list)
        # sanitize params (drop internals and convert types)
        best_node_params_yaml = {}
        for k, v in best_node_params.items():
            if str(k).startswith("_"):                 # drops _cache_dir, _override, _grid_keys
                continue
            if k in {"edge_index_dict", "num_nodes_dict", "metapaths", "X_attr"}:
                continue
            if isinstance(v, (sp.csr_matrix,np.ndarray)):
                continue
            if isinstance(v, Path):
                v = str(v)
            elif isinstance(v, (np.floating, np.integer)):
                v = v.item()
            best_node_params_yaml[str(k)] = v

        diagnostics_yaml = {}
        for k, v in best_meta.items():
            if k == "aggr_method":    # you already store it separately
                continue
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            diagnostics_yaml[str(k)] = v

        save_yaml(roi_emb_dir / "best_roi_supervision.yaml", {
            "best_score": float(best_score),
            "structure_method": best_node_params["structure_method"],
            "attr_method": best_node_params["attr_method"],
            "fusion_mode": best_node_params["fusion_mode"],
            "aggregation": best_meta["aggr_method"],
            "best_node_params": best_node_params_yaml,
            "diagnostics": diagnostics_yaml,
        })

        logging.info(f"Saved ROI embeddings and best config to {roi_emb_dir}")

    logging.info("Done.")

if __name__ == "__main__":
    main()
