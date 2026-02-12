#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("dummy_example"))
    ap.add_argument("--n_rois", type=int, default=3)
    ap.add_argument("--cells_per_roi", type=int, default=1_000_000)
    ap.add_argument("--box_size", type=float, default=10_000.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    dfs = []

    for r in range(args.n_rois):
        roi_id = f"ROI_{r:02d}"
        n = args.cells_per_roi
        box = args.box_size

        # Uniform points in 10k x 10k box
        xy = rng.uniform(0, box, size=(n, 2)).astype(np.float32)

        # Simple integer cell IDs
        cell_ids = (r * 10_000_000 + np.arange(n, dtype=np.int64))

        df_roi = pd.DataFrame(
            {
                "roi_id": roi_id,
                "cell_id": cell_ids,
                "x": xy[:, 0],
                "y": xy[:, 1],
            }
        )

        dfs.append(df_roi)

        print(f"[ROI {roi_id}] generated {n} cells in {box} x {box} box")

    df = pd.concat(dfs, ignore_index=True)

    df_path = outdir / "dummy_df.csv"
    df.to_csv(df_path, index=False)

    print(f"[OK] wrote {df_path} with shape {df.shape}")


if __name__ == "__main__":
    main()


'''
# 1) make dummy df
python make_dummy_example.py --outdir dummy_example

# 2) run FastRP pipeline
# IMPORTANT: --base should point to your project root that contains src/
python get_fastrp_embeddings.py \
  --config config/dummy_config.yaml \
  --df dummy_example/dummy_df.csv \
  --outdir dummy_example/out &> dummy_example/pipeline.log
'''