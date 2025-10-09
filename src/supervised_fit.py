# src/supervised_fit.py
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from itertools import product
import re

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

from src.node_embeddings import node_embedding
from src.aggregation import aggregate

from pathlib import Path
from scipy.io import savemat, loadmat
import scipy.sparse as sp
# ---- Human-readable filename helpers ----
def _grid_params_only(node_params: Dict) -> Dict:
    """
    Return only the params that came from the grid (i.e., those that vary).
    We pass these keys in via node_params['_grid_keys'] inside supervised_search.
    """
    keys = node_params.get("_grid_keys")
    if not keys:
        # fallback: all non-internal keys
        keys = [k for k in node_params.keys() if not k.startswith("_")]
    blacklist = ["l1_ratio"]
    def _keep(k: str) -> bool:
        return (k in node_params) and (k not in blacklist) and (not k.startswith("clf__"))
    return {k: node_params[k] for k in keys if _keep(k)}

def _filename_from_grid(method: str, grid_params: Dict) -> str:
    kvs = []
    for k in sorted(grid_params.keys(), key=str):
        v = grid_params[k]
        if isinstance(v, (list, tuple)):
            val = _fmt_list(v)
        elif isinstance(v, dict):
            # skip nested dicts to keep name compact
            continue
        else:
            val = _fmt_scalar(v)
        kvs.append(f"{_sanitize_token(str(k))}-{_sanitize_token(val)}")
    left = _sanitize_token((method or "").lower())
    mid = "__".join(kvs) if kvs else "default"
    return f"{left}_{mid}"

def _cache_path_from_grid(cache_dir: Path, method: str, grid_params: Dict) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = _filename_from_grid(method, grid_params) + ".mat"
    return cache_dir / fname

def _fmt_scalar(v):
    if v is None: return "none"
    if isinstance(v, bool): return "1" if v else "0"
    if isinstance(v, (int, np.integer)): return str(int(v))
    if isinstance(v, (float, np.floating)):
        # compact float, no trailing zeros
        s = f"{float(v):g}"
        # guard against scientific notation getting too long
        return s.replace("+", "")
    return str(v)

def _sanitize_token(s: str) -> str:
    # keep alnum, dot, plus, minus; collapse others to '-'
    s = re.sub(r"[^A-Za-z0-9.\-+]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "x"

def _fmt_list(val, max_elems=6):
    try:
        it = list(val)
    except Exception:
        return "list"
    parts = [_sanitize_token(_fmt_scalar(x)) for x in it[:max_elems]]
    if len(it) > max_elems:
        parts.append(f"more{len(it)-max_elems}")
    return "(" + "-".join(parts) + ")"


def _node_embedding_with_cache(
    G_all: nx.Graph,
    df_aligned: pd.DataFrame,
    node_params: Dict,
    cache_dir: Optional[Path],
    override: bool,
) -> np.ndarray:
    structure_method = str(node_params.get("structure_method", "random")).lower()
    if structure_method not in {"node2vec", "metapath2vec"}: # only need to cache for these time_consuming methods
        cache_dir = None
    out_path = _cache_path_from_grid(cache_dir, structure_method, _grid_params_only(node_params)) if cache_dir else None

    # Try load
    if out_path is not None and out_path.exists() and not override:
        try:
            Z = loadmat(str(out_path))["Z"]
            # lightweight sanity check
            if Z.shape[0] != G_all.number_of_nodes():
                logging.warning(
                    f"Cached embedding node count {Z.shape[0]} != graph nodes {G_all.number_of_nodes()}; recomputing."
                )
            else:
                logging.info(f"Loaded cached embedding: {out_path.name}")
                return Z
        except Exception as e:
            logging.warning(f"Failed to read cached .mat '{out_path.name}': {e}; recomputing.")

    # Compute fresh
    Z = node_embedding(G_all, df_aligned, node_params)

    # Save the embedding in .mat
    if out_path is not None:
        try:
            savemat(str(out_path), {"Z": np.asarray(Z, dtype=np.float64)})
            logging.info(f"Saved embedding cache: {out_path.name}")
        except Exception as e:
            logging.warning(f"Failed to write .mat cache '{out_path}': {e}")

    return Z


@dataclass
class SupervisedSearchConfig:
    params_fixed: Dict                     # fixed params for the chosen method
    params_search_space: Dict             
    aggregation_choices: List[str]         # e.g. ["mean_pool","global_attention","set2set","mil_attention"]
    label_col: str                         # ROI-level label column name in df_aligned (same per-ROI)
    group_col: Optional[str] = None        # optional grouping column, e.g. "Subject" to group CV by patient
    n_splits: int = 5
    random_state: int = 42
    cache_dir: Optional[Path] = None
    override: bool = False
    n_jobs: int = 1 # parallel workers (processes). 1 = serial.


def _roi_labels_from_df(df_aligned: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Returns a 2-column DF: ROI, label
    Assumes df_aligned has columns ['ROI', label_col] and each ROI has exactly one label.
    """
    lab = df_aligned[['ROI', label_col]].drop_duplicates()
    if lab['ROI'].duplicated().any():
        raise ValueError(f"Label column '{label_col}' is not constant within ROI.")
    return lab

def _score_multiclass_robust(y_true, proba, n_classes,type="accuracy") -> float:
    """Macro ROC-AUC when possible; else balanced accuracy."""
    if type == "accuracy":
        y_pred = np.argmax(proba, axis=1)
        return float(accuracy_score(y_true, y_pred))
    if type == "average_acc":
        y_pred = np.argmax(proba, axis=1)
        return float(balanced_accuracy_score(y_true, y_pred))
    elif type == "roc_auc":
        if len(np.unique(y_true)) < 2:
            raise ValueError("single-class fold")
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        else:
            return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    else:
        raise ValueError(f"Invalid type: {type}")

def _evaluate_once(
    G_all: nx.Graph,
    df_aligned: pd.DataFrame,
    node_params: Dict,
    aggr_method: str,
    label_col: str,
    group_col: Optional[str],
    n_splits: int,
    random_state: int,
) -> Tuple[float, Dict, List]:
    """
    Train/val CV on ROI embeddings produced by (node embedding -> aggregation).
    Returns (mean_cv_score, diagnostics_dict).
    """
    # 1) Node embedding
    Z = _node_embedding_with_cache(
        G_all=G_all,
        df_aligned=df_aligned,
        node_params=node_params,
        cache_dir=node_params.get("_cache_dir"),
        override=bool(node_params.get("_override", False)),
    )# (N_nodes, d)\
    
        
    # 2) Aggregate to ROI level
    E, group_ids = aggregate(
        Z, G=G_all, method=aggr_method, return_group_ids=True
    )  # E: (N_ROIs, d’), group_ids ~ ordered ROI ids as seen in G_all

    # 3) Align ROI labels to aggregation order
    lab_df = _roi_labels_from_df(df_aligned, label_col)
    # group_ids contain ROI identifiers; when G_all nodes are (ROI, local_id), group_ids are those ROI keys.
    # Convert to str for safe merge keys if needed:
    key_series = pd.Series(group_ids, name="ROI")
    roi_df = pd.DataFrame({"ROI": key_series})
    lab_df_merged = roi_df.merge(lab_df, on="ROI", how="left")
    if lab_df_merged[label_col].isna().any():
        missing = roi_df[lab_df_merged[label_col].isna()].ROI.tolist()
        raise ValueError(f"Missing ROI labels for: {missing[:10]} (and possibly more).")

    y_raw = lab_df_merged[label_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    # Optional grouping at ROI level (e.g., Subject)
    if group_col is not None and group_col in df_aligned.columns:
        grp_map = df_aligned[['ROI', group_col]].drop_duplicates()
        grp_vec = roi_df.merge(grp_map, on="ROI", how="left")[group_col].values
        use_groups = True
    else:
        grp_vec = None
        use_groups = False

    # 4) Choose an effective number of folds that the data can support
    #    - Must be <= min samples per class
    #    - If grouping is used, must also be <= number of unique groups
    class_counts = np.bincount(y, minlength=n_classes)
    min_class_count = int(class_counts[class_counts > 0].min()) if (class_counts > 0).any() else 0
    n_splits_eff = int(n_splits)
    n_splits_eff = min(n_splits_eff, max(2, min_class_count))  # at most min per-class, but keep intent for >=2
    if use_groups:
        n_unique_groups = int(len(pd.Series(grp_vec).dropna().unique()))
        n_splits_eff = min(n_splits_eff, n_unique_groups)

    if n_splits_eff < 2:
        raise ValueError(
            f"Not enough data to perform 2-fold CV "
            f"(requested n_splits={n_splits}, min_class_count={min_class_count}"
            + (f", n_unique_groups={n_unique_groups}" if use_groups else "") + ")."
        )
    if n_splits_eff != n_splits:
        logging.info(f"[CV] Adjusted n_splits from {n_splits} -> {n_splits_eff} "
                     f"(min_class_count={min_class_count}"
                     + (f", n_unique_groups={n_unique_groups}" if use_groups else "") + ").")

    # 5) Build the splitter
    if use_groups:
        splitter = GroupKFold(n_splits=n_splits_eff)
        split_iter = splitter.split(E, y, groups=grp_vec)
    else:
       splitter = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
       split_iter = splitter.split(E, y)


    clf_list = [] # list of clf objects
    scores = []
    per_class_fold_acc_int = {k: [] for k in range(n_classes)}
   
    for tr, va in split_iter:
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="saga",
                n_jobs=1, # avoid nested parallelization
                random_state=random_state,
                penalty="elasticnet",
                l1_ratio=float(node_params.get("l1_ratio", 1.0)),
            ),
        )
        clf.fit(E[tr], y[tr])
        proba = clf.predict_proba(E[va])
        fold = _score_multiclass_robust(y[va], proba, n_classes)
        scores.append(fold)
        # record detailed predictions by class
        y_pred = clf.predict(E[va])
        
        for k in range(n_classes):
            mask = (y[va] == k)
            if mask.sum() > 0:
                acc_k = float((y_pred[mask] == k).mean())
            else:
                acc_k = float("nan")
            per_class_fold_acc_int[k].append(acc_k)
        clf_list.append(clf)

    # summarize per-class mean ± std (ignore NaNs safely)
    classes_str = list(map(str, le.classes_))
    per_class_fold_acc = {classes_str[k]: per_class_fold_acc_int[k] for k in range(n_classes)}
    per_class_mean = {c: float(np.nanmean(per_class_fold_acc[c])) for c in classes_str}
    per_class_std  = {c: float(np.nanstd (per_class_fold_acc[c], ddof=1)) if len(per_class_fold_acc[c]) > 1 else 0.0
                      for c in classes_str}

    return clf_list,float(np.mean(scores)), {
        "emb_dim": Z.shape[1],
        "roi_count": E.shape[0],
        "aggr_out_dim": E.shape[1],
        "classes": classes_str,
        "fold_scores": [float(s) for s in scores],
        "per_class_fold_acc": per_class_fold_acc,  # dict[str] -> List[float]
        "per_class_mean": per_class_mean,          # dict[str] -> float
        "per_class_std": per_class_std,            # dict[str] -> float
        "n_splits_effective": int(n_splits_eff),
        "clf_l1_ratio": float(node_params.get("l1_ratio", 1.0)),
     }
# ---- picklable worker for parallel map ----
def _run_one_combo(args) -> Tuple[bool, Dict]:
    """
    Returns (ok, payload). If ok=True, payload has: score, node_params, aggr_method, diag, clf_list
    If ok=False, payload has: 'error'
    """
    try:
        import os
        (G_all, df_aligned, node_params, aggr_method, label_col, group_col, n_splits, random_state) = args

        # make sure each worker is single-threaded (helps with MKL/BLAS oversubscription)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        clf_list, score, diag = _evaluate_once(
            G_all, df_aligned, node_params, aggr_method, label_col, group_col, n_splits, random_state
        )
        return True, {
            "score": score,
            "node_params": node_params,
            "aggr_method": aggr_method,
            "diag": diag,
            "clf_list": clf_list,
        }
    except Exception as e:
        return False, {"error": str(e)}
def _iter_param_grid(space: Dict) -> List[Dict]:
    """
    Expand a discrete parameter grid into all combinations.
    Accepts:
      - key: [v1, v2, ...] or tuple(...)  -> treated as choices
      - key: {"choice": [v1, v2, ...]}    -> treated as choices
    Rejects continuous specs like {"uniform": ...}, {"log_uniform": ...}.
    """
    if not space:
        return [dict()]

    keys, values_lists = [], []
    for k, v in space.items():
        if isinstance(v, dict) and "choice" in v:
            vals = list(v["choice"])
        elif isinstance(v, (list, tuple, np.ndarray)):
            vals = list(v)
        else:
            raise ValueError(
                f"Grid search requires discrete choices for '{k}'. "
                f"Use a list/tuple or {{'choice': [...]}}; got: {v}"
            )
        if len(vals) == 0:
            raise ValueError(f"Empty grid for key '{k}'.")
        keys.append(k)
        values_lists.append(vals)

    return [dict(zip(keys, combo)) for combo in product(*values_lists)]
def supervised_search(
    G_all: nx.Graph,
    df_aligned: pd.DataFrame,
    cfg: SupervisedSearchConfig,
) -> Tuple[float, Dict, Dict, List]:
    if not cfg.aggregation_choices:
        raise ValueError("aggregation_choices must be a non-empty list.")

    grid = _iter_param_grid(cfg.params_search_space or {})
    tasks = []
    for combo in grid:
        node_params = dict(cfg.params_fixed or {})
        node_params.update(combo)
        node_params["_grid_keys"] = list(combo.keys())
        node_params["_cache_dir"] = cfg.cache_dir
        node_params["_override"]  = cfg.override
        for aggr_method in cfg.aggregation_choices:
            tasks.append((
                G_all, df_aligned, node_params, aggr_method,
                cfg.label_col, cfg.group_col, cfg.n_splits, cfg.random_state
            ))

    total = len(tasks)
    logging.info(f"[ROI supervision] Grid size: {len(grid)} node-param combos × {len(cfg.aggregation_choices)} aggregations = {total} runs")

    best_score, best_node_params, best_meta, best_clf_list = -1.0, None, None, None

    if cfg.n_jobs is None or int(cfg.n_jobs) <= 1:
        # Serial path (original behavior)
        for i, args in enumerate(tasks, start=1):
            ok, payload = _run_one_combo(args)
            if not ok:
                logging.warning(f"[skip] combo #{i} failed: {payload['error']}")
                continue
            score = payload["score"]
            node_params = payload["node_params"]
            aggr_method = payload["aggr_method"]
            diag = payload["diag"]
            clf_list = payload["clf_list"]

            logging.info(f"[{i}/{total}] {node_params['structure_method']}+{node_params['attr_method']}+{node_params['fusion_mode']}+{aggr_method} -> {score:.4f} | params={_grid_params_only(node_params)}")
            if score > best_score:
                best_score = score
                best_node_params = dict(node_params)
                best_meta = {"aggr_method": aggr_method, **diag}
                best_clf_list = clf_list
                if "per_class_mean" in diag and "per_class_std" in diag:
                    cls_order = list(diag.get("classes", diag["per_class_mean"].keys()))
                    cls_str = ", ".join(
                        f"{c}:{diag['per_class_mean'][c]:.3f}±{diag['per_class_std'][c]:.3f}"
                        for c in cls_order
                    )
                    logging.info(f"[best-so-far] {node_params['structure_method']}+{node_params['attr_method']}+{node_params['fusion_mode']}+{aggr_method} | by-class acc: {cls_str}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # Parallel path
        n_workers = int(cfg.n_jobs)
        logging.info(f"[parallel] Launching {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_run_one_combo, args): idx for idx, args in enumerate(tasks, start=1)}
            for fut in as_completed(futures):
                i = futures[fut]
                ok, payload = fut.result()
                if not ok:
                    logging.warning(f"[skip] combo #{i} failed: {payload['error']}")
                    continue
                score = payload["score"]
                node_params = payload["node_params"]
                aggr_method = payload["aggr_method"]
                diag = payload["diag"]
                clf_list = payload["clf_list"]

                logging.info(f"[{i}/{total}] {node_params['structure_method']}+{node_params['attr_method']}+{node_params['fusion_mode']}+{aggr_method} -> {score:.4f} | params={_grid_params_only(node_params)}")
                if score > best_score:
                    best_score = score
                    best_node_params = dict(node_params)
                    best_meta = {"aggr_method": aggr_method, **diag}
                    best_clf_list = clf_list
                    if "per_class_mean" in diag and "per_class_std" in diag:
                        cls_order = list(diag.get("classes", diag["per_class_mean"].keys()))
                        cls_str = ", ".join(
                            f"{c}:{diag['per_class_mean'][c]:.3f}±{diag['per_class_std'][c]:.3f}"
                            for c in cls_order
                        )
                        logging.info(f"[best-so-far] {node_params['structure_method']}+{node_params['attr_method']}+{node_params['fusion_mode']}+{aggr_method} | by-class acc: {cls_str}")

    if best_node_params is None:
        raise RuntimeError("All grid combinations failed; please check your config and data.")
    return best_score, best_node_params, best_meta, best_clf_list