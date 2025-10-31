from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Namespace:
    """Small helper to access dict keys as attributes."""
    __dict__: Dict[str, Any]

def dict_to_ns(d: dict) -> Namespace:
    return Namespace(**{k: dict_to_ns(v) if isinstance(v, dict) else v for k,v in d.items()})

import pandas as pd
def align_df_with_G_all(df, G_all):
    roi_col = "roi_id"
    id_col  = "cell_id"  
    # 1) Build the target index from the graph's nodes (keeps the graph's order)
    node_index = pd.MultiIndex.from_tuples(list(G_all.nodes()),
                                        names=[roi_col, id_col])
    # 2) Set a matching MultiIndex on df
    #    (fails loudly if duplicates — better than silently picking a row)
    df_idxed = df.set_index([roi_col, id_col])
    if df_idxed.index.has_duplicates:
        raise ValueError("Duplicates found in df")
        # pick the first occurrence per (roi_id, cell_id); or use a different reducer
        #  = df_idxed[~df_idxed.index.duplicated(keep="first")]

    # 3) Reindex to align order exactly to G_all.nodes()
    df_aligned = df_idxed.reindex(node_index)
    # restore roi_col and id_col as columns
    df_aligned = df_aligned.reset_index()
    return df_aligned


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
