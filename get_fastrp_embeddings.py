#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.io import savemat
from sklearn.neighbors import radius_neighbors_graph

from src.fastrp import fastrp_wrapper


# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------------
# IO helpers
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_node_order(out_prefix: Path, node_order: List[Tuple[str, int]]) -> None:
    ensure_dir(out_prefix.parent)

    json_path = out_prefix.with_suffix(".json")
    json_ready = [(str(r), str(cid)) for (r, cid) in node_order]
    json_path.write_text(json.dumps(json_ready, indent=2))

    pkl_path = out_prefix.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(node_order, f, protocol=pickle.HIGHEST_PROTOCOL)


def coerce_columns(df: pd.DataFrame, cell_columns: Dict[str, str] | None) -> pd.DataFrame:
    if not cell_columns:
        return df
    rename_map = {v: k for k, v in cell_columns.items() if v in df.columns and k != v}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ----------------------------
# Graph building (sklearn)
# ----------------------------
def radius_adjacency_sklearn(coords: np.ndarray, radius: float, n_jobs: int) -> sp.csr_matrix:
    logger.info(f"  Building radius graph (n={coords.shape[0]}, radius={radius})")

    t0 = time.time()
    A = radius_neighbors_graph(
        coords,
        radius=radius,
        mode="connectivity",
        include_self=False,
        n_jobs=n_jobs,
    ).tocsr()

    # Symmetrize
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    A.sum_duplicates()
    A.eliminate_zeros()
    A = A.astype(np.float32)

    elapsed = time.time() - t0
    logger.info(
        f"  Graph built: nnz={A.nnz:,}, avg_degree={A.nnz / A.shape[0]:.2f}, "
        f"time={elapsed:.2f}s"
    )

    approx_mem_mb = (A.nnz * 8 + A.shape[0] * 4) / (1024 ** 2)
    logger.info(f"  Approx CSR memory ≈ {approx_mem_mb:.2f} MB")

    return A


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    total_start = time.time()

    ap = argparse.ArgumentParser(
        description="Build block-diagonal radius graph adjacency and run FastRP."
    )
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--df", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    logger.info("Starting pipeline")
    logger.info(f"Config: {args.config}")
    logger.info(f"Dataframe: {args.df}")
    logger.info(f"Output dir: {args.outdir}")

    cfg = yaml.safe_load(args.config.read_text())
    outdir = args.outdir
    ensure_dir(outdir)

    logger.info("Loading dataframe...")
    df = pd.read_csv(args.df)
    df = coerce_columns(df, cfg.get("cell_columns", None))
    logger.info(f"Dataframe loaded: shape={df.shape}")

    required = ["roi_id", "cell_id", "x", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    gcfg = cfg["graph"]
    if gcfg.get("type", "radius") != "radius":
        raise ValueError("Only radius graph supported in this script.")

    radius = float(gcfg["radius"])
    logger.info(f"Using radius={radius}")

    A_blocks: List[sp.csr_matrix] = []
    node_order: List[Tuple[str, int]] = []

    logger.info("Building graphs per ROI...")

    for roi, df_roi in df.groupby("roi_id", sort=True):
        logger.info(f"Processing ROI={roi}")
        roi_start = time.time()

        df_roi = df_roi.reset_index(drop=True)
        coords = df_roi[["x", "y"]].to_numpy(dtype=np.float32, copy=False)

        A_roi = radius_adjacency_sklearn(coords, radius=radius, n_jobs=args.n_jobs)
        A_blocks.append(A_roi)

        cids = df_roi["cell_id"].to_numpy()
        for cid in cids:
            node_order.append((str(roi), int(cid)))

        roi_time = time.time() - roi_start
        logger.info(f"Finished ROI={roi} in {roi_time:.2f}s")

    logger.info("Building block-diagonal adjacency matrix...")
    t0 = time.time()
    A_all = sp.block_diag(A_blocks, format="csr", dtype=np.float32)
    A_all.sum_duplicates()
    A_all.eliminate_zeros()
    logger.info(
        f"Block-diagonal built: shape={A_all.shape}, nnz={A_all.nnz:,}, "
        f"time={time.time() - t0:.2f}s"
    )

    # Save adjacency
    ensure_dir(outdir / "adjacency")
    logger.info("Saving adjacency matrix...")
    sp.save_npz(outdir / "adjacency" / "A_csr.npz", A_all)
    save_node_order(outdir / "adjacency" / "node_order", node_order)

    # Save dataframe snapshot
    ensure_dir(outdir / "data")
    df.to_csv(outdir / "data" / "df.csv", index=False)

    # Save config
    ensure_dir(outdir / "config")
    (outdir / "config" / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False)
    )

    # ---------------- FastRP ----------------
    logger.info("Running FastRP embedding...")
    t0 = time.time()
    fcfg = cfg["fastrp"]
    Z = fastrp_wrapper(A_all, fcfg)
    logger.info(f"FastRP done in {time.time() - t0:.2f}s, shape={Z.shape}")

    ensure_dir(outdir / "embeddings")
    np.save(outdir / "embeddings" / "Z.npy", Z)
    savemat(str(outdir / "embeddings" / "Z.mat"), {"Z": Z})

    total_time = time.time() - total_start
    logger.info(f"Pipeline completed successfully in {total_time:.2f}s")
    logger.info(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
