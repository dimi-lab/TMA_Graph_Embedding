from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Project import helpers
# -----------------------------
import sys

def add_to_syspath(path: Path):
    p = str(path.resolve())
    if p not in sys.path:
        sys.path.append(p)


def import_project_modules(base: Path):
    add_to_syspath(base)
    from src.node_embeddings import structure_embedding, node_attribute_embedding
    from src.aggregation import aggregate
    from src.utils import align_df_with_G_all
    return {
        "structure_embedding": structure_embedding,
        "node_attribute_embedding": node_attribute_embedding,
        "aggregate": aggregate,
        "align_df_with_G_all": align_df_with_G_all,
    }

# -----------------------------
# IO helpers
# -----------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _roi_labels_from_df(df_aligned: pd.DataFrame, label_col: str) -> pd.DataFrame:
    lab = df_aligned[["ROI", label_col]].drop_duplicates()
    if lab["ROI"].duplicated().any():
        raise ValueError(f"Label column '{label_col}' is not constant within ROI.")
    return lab

# -----------------------------
# Logistic regression traceback
# -----------------------------

def fit_lr_and_get_weights(E: np.ndarray, y_raw: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Fit StandardScaler+LogReg and return coefficients in *original E space*.
    Returns (W, b, class_names) where W has shape (n_classes, d_out).
    """
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(map(str, le.classes_))

    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=None,
            random_state=random_state,
            multi_class="auto",
        ),
    )
    pipe.fit(E, y)

    scaler: StandardScaler = pipe.named_steps["standardscaler"]
    lr: LogisticRegression = pipe.named_steps["logisticregression"]

    # De-standardize: model uses z = (E - mean)/scale; w_z -> w_E = w_z / scale
    W = lr.coef_.astype(float, copy=True)
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)  # guard divide-by-zero
    W = W / scale[None, :]
    b = lr.intercept_.astype(float, copy=True) - (W * scaler.mean_[None, :] / scale[None, :]).sum(axis=1)
    return W, b, classes


def combine_class_weights(W: np.ndarray, mode: str = "l2") -> np.ndarray:
    """Combine OvR class-wise weights -> a single importance vector per feature."""
    if mode == "l1":
        imp = np.mean(np.abs(W), axis=0)
    elif mode == "max":
        imp = np.max(np.abs(W), axis=0)
    else:  # l2 (default)
        imp = np.sqrt(np.mean(W ** 2, axis=0))
    return imp

# -----------------------------
# Aggregation traceback
# -----------------------------

def map_roi_weights_to_node_dims(
    W_roi: np.ndarray,
    aggr_method: str,
    emb_dim_nodes: int,
) -> np.ndarray:
    """
    Map ROI-space weights back to node-embedding dimensions.

    Parameters
    ----------
    W_roi : (n_classes, d_out) ROI-level weights in original (unstandardized) feature space
    aggr_method : str used in aggregation
    emb_dim_nodes : int = dimensionality d of node embeddings Z

    Returns
    -------
    imp_nodes : (d,) global importance per node-embedding dimension
    """
    m = (aggr_method or "mean_pool").lower()
    d_out = W_roi.shape[1]

    # Methods that preserve dimensionality one-to-one
    same_dim = {"mean_pool", "attention_stub", "add_pool", "sum_pool", "max_pool",
                "global_attention", "softmax_pool", "powermean_pool", "mil_mean", "mil_max", "mil_attention"}

    if m in same_dim:
        if d_out != emb_dim_nodes:
            raise ValueError(f"Aggregation '{m}' produced d_out={d_out}, but node emb dim is {emb_dim_nodes}.")
        return combine_class_weights(W_roi, mode="l2")

    if m == "set2set":
        if d_out % 2 != 0:
            raise ValueError("Set2Set output dim should be 2*d; got d_out=%d" % d_out)
        d = d_out // 2
        if d != emb_dim_nodes:
            raise ValueError(f"Set2Set 2*d={d_out} implies d={d}, but node emb dim is {emb_dim_nodes}.")
        # Pair the two halves for each original dim: use L2 of the pair per class, then combine across classes
        W1 = W_roi[:, :d]
        W2 = W_roi[:, d:]
        W_pair = np.sqrt(W1**2 + W2**2)  # (n_classes, d)
        return combine_class_weights(W_pair, mode="l2")

    if m == "lstm_pool":
        # LSTM aggregation can change dimensionality arbitrarily, not invertible without the trained aggregator.
        # We fall back to a proportional mapping by truncation/padding.
        d = emb_dim_nodes
        M = min(d_out, d)
        W_trim = W_roi[:, :M]
        imp = np.zeros(d, dtype=float)
        imp[:M] = combine_class_weights(W_trim, mode="l2")
        return imp

    raise NotImplementedError(f"Traceback not implemented for aggregation '{aggr_method}'.")

# -----------------------------
# Main routine
# -----------------------------

def run(outdir: Path, base: Optional[Path] = None, import_base: Optional[Path] = None, save_csv: bool = True, random_state: int = 42):
    """
    Load best ROI-supervised config from a run directory and attribute classifier importance
    back to node-embedding dimensions.
    """
    outdir = outdir.resolve()
    base = base.resolve() if base else Path(__file__).resolve().parent
    mods = import_project_modules(import_base or base)

    # 1) Load config + logs
    cfg = load_yaml(outdir / "config" / "resolved_config.yaml")
    best = load_yaml(outdir / "logs" / "best_roi_supervision.yaml")
    aggr_method = best["aggregation"]
    node_method = best["method"]
    node_params = best.get("best_node_params", {})

    # 2) Load data and graph
    df = pd.read_csv(outdir / "dataframes" / "df.csv")
    # Align to graph node order
    with open(outdir / "graphs" / f"G_all_{cfg['graph']['type']}.pkl", "rb") as f:
        G_all = pickle.load(f)
    df_aligned = mods["align_df_with_G_all"](df, G_all)

    # 3) Recompute node embeddings with best params (handles attr_mode='concat' internally in supervised flow, here we recompute similarly)
    Z = mods["structure_embedding"](G_all, method=node_method, **node_params)
    Z = np.asarray(Z, dtype=float)
    d_nodes = Z.shape[1]

    # If the run used attribute concatenation within supervised_fit, reflect that here
    attr_mode = str(node_params.get("attr_mode", "none")).lower()
    if attr_mode == "concat":
        attr_method = str(node_params.get("attr_method", "passthrough")).lower()
        attr_cols = node_params.get("attr_cols", None)
        X_attr = mods["node_attribute_embedding"](df_aligned, method=attr_method, cols=attr_cols)
        if X_attr.shape[0] != Z.shape[0]:
            raise ValueError("Attribute rows != embedding rows. Ensure df_aligned matches G_all.")
        Z = np.hstack([Z, X_attr.astype(float, copy=False)])
        d_nodes = Z.shape[1]

    # 4) Aggregate to ROI embeddings using the best aggregation
    E, group_ids = mods["aggregate"](Z, G=G_all, method=aggr_method, return_group_ids=True)

    # 5) Prepare labels at ROI level (same ordering as aggregation output)
    label_col = cfg.get("roi_supervision", {}).get("label_col", "roi_label")
    lab_df = _roi_labels_from_df(df_aligned, label_col)
    roi_df = pd.DataFrame({"ROI": pd.Series(group_ids)})
    y_raw = roi_df.merge(lab_df, on="ROI", how="left")[label_col].values
    if pd.isna(y_raw).any():
        missing = roi_df[pd.isna(y_raw)].ROI.tolist()
        raise ValueError(f"Missing ROI labels for: {missing[:10]} ...")

    # 6) Fit LR and extract ROI-space weights (de-standardized)
    W_roi, b, classes = fit_lr_and_get_weights(E, y_raw, random_state=random_state)

    # 7) Map ROI weights -> node-embedding dimensions
    imp_nodes = map_roi_weights_to_node_dims(W_roi, aggr_method, emb_dim_nodes=d_nodes)

    # 8) Save report
    out_dir = outdir / "evaluate" / "roi_supervised_best"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_imp = pd.DataFrame({
        "node_dim": np.arange(d_nodes, dtype=int),
        "importance": imp_nodes,
    }).sort_values("importance", ascending=False, kind="mergesort")

    # Also include per-class signed weights in ROI space for completeness
    df_w = pd.DataFrame(W_roi.T, columns=[f"w_class={c}" for c in classes])
    df_out = pd.concat([df_imp.reset_index(drop=True), df_w], axis=1)

    if save_csv:
        csv_path = out_dir / "importance_node_dims.csv"
        df_out.to_csv(csv_path, index=False)
        logging.info(f"Saved node-dimension importance: {csv_path}")

    # Return key artifacts for programmatic use
    return {
        "aggregation": aggr_method,
        "node_method": node_method,
        "classes": classes,
        "W_roi": W_roi,  # (n_classes, d_out)
        "importance": imp_nodes,  # (d_nodes,)
        "node_dim_order": df_out["node_dim"].tolist(),
        "report_path": str((out_dir / "importance_node_dims.csv").resolve()) if save_csv else None,
    }


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Trace classifier importance back to node-embedding dimensions.")
    ap.add_argument("--outdir", type=Path, required=True, help="Run output directory (same used by main.py)")
    ap.add_argument("--base", type=Path, default="./", help="Project root containing src/")
    ap.add_argument("--import-base", type=Path, default=None, help="Optional different import base for src/")
    ap.add_argument("--no-save", action="store_true", help="Do not save CSV; just print a head preview")
    ap.add_argument("--topk", type=int, default=20, help="Show top-k dimensions in console")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for LR fitting")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    artifacts = run(
        outdir=args.outdir,
        base=args.base,
        import_base=args.import_base,
        save_csv=not args.no_save,
        random_state=args.seed,
    )

    imp = artifacts["importance"]
    order = np.argsort(-imp)
    topk = min(args.topk, imp.shape[0])
    print("\nTop-{} node-embedding dimensions by importance (L2 across classes):".format(topk))
    for rank, j in enumerate(order[:topk], start=1):
        print(f"#{rank:2d}  dim={j:<4d}  importance={imp[j]:.6f}")


if __name__ == "__main__":
    main()

# example run:
# python interprete.py --outdir <your_run_outdir> --topk 30
