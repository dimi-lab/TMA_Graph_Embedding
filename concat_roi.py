#!/usr/bin/env python3
# concat_roi.py
# Concatenate specific columns from all CSVs in a folder and save as RDS.
# python concat_roi.py --by_roi_dir /projects/wangc/m344313/OVTMA_project/data/OVTMA_fov297/by_ROI --phenotype_column metacluster --roi_id_column region --out /projects/wangc/m344313/OVTMA_project/data/OVTMA_fov297/combined.rds --recursive

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
import time
def find_csvs(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted([p for p in root.glob(pattern) if p.is_file()])

def normalize_maybe_none(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = s.strip().lower()
    return None if s2 in {"", "none", "null", "na"} else s

def read_needed_cols(csv_path: Path, phen_col: str, roi_col: str, cell_col: Optional[str], x_col: str, y_col: str) -> pd.DataFrame:
    cols = [x_col, y_col, phen_col, roi_col] + ([cell_col] if cell_col else [])
    try:
        df = pd.read_csv(csv_path, usecols=cols)
    except ValueError as e:
        # Usually happens when a required column is missing
        try:
            header = pd.read_csv(csv_path, nrows=0)
            present = set(header.columns)
            missing = [c for c in cols if c not in present]
        except Exception:
            missing = cols
        raise ValueError(f"{csv_path}: missing required columns: {missing}. Original error: {e}")

    # If no cell_id column was provided, create one from the row index
    if cell_col is None:
        df = df.reset_index(drop=True)
        df.insert(0, "cell_id", df.index.astype("int64"))
    missing = [k for k in cols if k not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cells RDS after renaming: {missing}")
    return df

def write_rds(df: pd.DataFrame, out_path: Path):
    # Try pyreadr first
    try:
        import pyreadr  # type: ignore
        pyreadr.write_rds(str(out_path), df)
        print(f"Saved RDS via pyreadr: {out_path}")
        return
    except Exception as e1:
        print(f"pyreadr write failed or not available: {e1}", file=sys.stderr)

    # Fallback to rpy2 (requires R installed)
    try:
        import rpy2.robjects as ro  # type: ignore
        from rpy2.robjects import pandas2ri  # type: ignore
        pandas2ri.activate()
        ro.r["saveRDS"](pandas2ri.py2rpy(df), str(out_path))
        print(f"Saved RDS via rpy2: {out_path}")
        return
    except Exception as e2:
        print(f"rpy2 write failed or not available: {e2}", file=sys.stderr)

    raise RuntimeError(
        "Could not write .rds file. Install one of:\n"
        "  - pyreadr  (pip install pyreadr)\n"
        "  - rpy2 + R (pip install rpy2; ensure R is installed)\n"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Read specified columns from all CSVs in a directory and save as a single .rds"
    )
    parser.add_argument(
        "--by_roi_dir", required=True,
        help="Directory containing CSV files."
    )
    parser.add_argument(
        "--cell_id_column", default="none",
        help="Column name for cell ID (default: none). If none, a 'cell_id' column is created from row index."
    )
    parser.add_argument(
        "--x_column", required=True,
        help="Column name for x coordinate."
    )
    parser.add_argument(
        "--y_column", required=True,
        help="Column name for y coordinate."
    )
    parser.add_argument(
        "--phenotype_column", required=True,
        help="Column name for phenotype."
    )
    parser.add_argument(
        "--roi_id_column", required=True,
        help="Column name for ROI ID."
    )
    parser.add_argument(
        "--out", default="combined.rds",
        help="Output .rds path (default: combined.rds)"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Also read CSVs in subdirectories."
    )
    args = parser.parse_args()

    root = Path(args.by_roi_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Error: {root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    csvs = find_csvs(root, args.recursive)
    if not csvs:
        print(f"No CSV files found in {root} (recursive={args.recursive}).", file=sys.stderr)
        sys.exit(1)

    cell_col = normalize_maybe_none(args.cell_id_column)
    x_col = args.x_column
    y_col = args.y_column
    phen_col = args.phenotype_column
    roi_col = args.roi_id_column

    dfs = []
    errors = 0
    for p in csvs:
        try:
            df = read_needed_cols(p, phen_col, roi_col, cell_col, x_col, y_col)
            # Reorder columns for consistency: put cell id first if it exists/was created.
            desired_order = []
            if cell_col is None:
                desired_order = ["cell_id", phen_col, roi_col, x_col, y_col]
            else:
                desired_order = [cell_col, phen_col, roi_col, x_col, y_col]
            df = df[desired_order]
            dfs.append(df)
            print(f"Read {p} -> {len(df)} rows")
        except Exception as e:
            errors += 1
            print(f"Skipping {p}: {e}", file=sys.stderr)

    if not dfs:
        print("No data read (all files missing required columns?).", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"Concatenated rows: {len(combined)} (from {len(dfs)} files; {errors} file(s) skipped)")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    write_rds(combined, out_path)
    t1 = time.time()
    print(f"Time taken: {t1-t0} seconds")

if __name__ == "__main__":
    main()
