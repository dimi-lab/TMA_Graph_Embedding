#!/usr/bin/env python3
# logs_to_csv.py
from __future__ import annotations

import argparse, ast, json, re, sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

#  2025-10-09 15:25:57,217 | INFO | [12/128] method=fastrp | attr=passthrough | fusion=concat | aggregation=set2set | score[roc_auc]=0.812345 | params={"dim":256,"weights":[1,2,4]}
LINE_RE = re.compile(
    r"""
    ^(?:\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+\|\s+INFO\s+\|\s+)?  # optional ts + INFO +
    \[\s*(?P<i>\d+)\s*/\s*(?P<n>\d+)\s*\]\s*
    method=(?P<method>[^|]+?)\s*\|\s*
    (?:attr=(?P<attr>[^|]+?)\s*\|\s*)?
    (?:fusion=(?P<fusion>[^|]+?)\s*\|\s*)?
    aggregation=(?P<aggregation>[^|]+?)\s*\|\s*
    score\[(?P<score_type>[^\]]+)\]\s*=\s*(?P<score>-?\d+(?:\.\d+)?)\s*\|\s*
    params=(?P<params>\{.*\})
    \.?\s*$  # optional trailing period
    """,
    re.VERBOSE,
)
def parse_params(text: str) -> Dict[str, Any]:
    # tolerant dict parsing
    try:
        val = ast.literal_eval(text)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    try:
        val = json.loads(
            text.replace("'", '"')
                .replace("None", "null")
                .replace("True", "true")
                .replace("False", "false")
        )
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    return {}

def canonical_weights(w: Any) -> str | None:
    # Turn list-like weights into a compact canonical string to use as a *categorical key*
    if w is None:
        return None
    if isinstance(w, (list, tuple)):
        try:
            return json.dumps(list(w), separators=(",", ":"))
        except Exception:
            return str(w)
    return str(w)

def parse_line(line: str) -> Dict[str, Any] | None:
    m = LINE_RE.search(line.strip())
    if not m:
        return None
    g = m.groupdict()
    params = parse_params(g["params"])

    weights_key = canonical_weights(params.get("weights"))
    dim = params.get("dim", None)
    attr_mode = params.get("attr_mode", g.get("attr"))
    fusion_mode = params.get("fusion_mode", g.get("fusion"))

    return {
            "timestamp": 'none',
            "idx": int(g["i"]),
            "n_total": int(g["n"]),
            "method": g["method"].strip(),
            "aggregation": g["aggregation"].strip(),
            "score": float(g["score"]),
            "score_type": g["score_type"].strip(),
            "l1_ratio": float(params.get("l1_ratio", 1.0)),
            "param.dim": pd.to_numeric(dim, errors="coerce"),
            "param.attr_mode": attr_mode,
            "param.fusion_mode": fusion_mode,
            "param.weights_vec": weights_key or 'none',
            "param.q": params.get("q",1),
            "raw.params": g["params"],
    }

def rows_from_path(p: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            r = parse_line(ln)
            if r:
                rows.append(r)
    return rows

def main():
    ap = argparse.ArgumentParser(description="Parse run logs into a tidy CSV.")
    ap.add_argument("inputs", nargs="+", help="Log files, directories, or glob patterns.")
    ap.add_argument("--out", type=Path, default=Path("analysis/parsed_runs.csv"))
    args = ap.parse_args()

    # collect files
    files: List[Path] = []
    for s in args.inputs:
        p = Path(s)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.log")))
        else:
            files.extend([Path(x) for x in sorted(Path().glob(s)) if Path(x).is_file()])

    if not files:
        print("No input log files found.", file=sys.stderr)
        sys.exit(2)

    all_rows: List[Dict[str, Any]] = []
    for f in files:
        all_rows.extend(rows_from_path(f))

    if not all_rows:
        print("No matching lines found in provided logs.", file=sys.stderr)
        sys.exit(3)

    df = pd.DataFrame(all_rows)

    # pivot to wide format and merge duplicates
    index_cols = [c for c in df.columns if c not in ("score", "score_type")]
    df_wide = (
        df.pivot_table(
            index=index_cols,          # "everything else the same"
            columns="score_type",      # unique score types -> columns
            values="score",            # fill with the numeric scores
            aggfunc="first"            # or "mean"/"max" if you expect true duplicates
        )
        .reset_index()
    )
    df_wide.columns.name = None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out.resolve()}")

if __name__ == "__main__":
    main()

# python logs_to_csv.py --out analysis/fastrp_het.csv /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp_het/logs/run_20250929_145007.log
# python logs_to_csv.py --out analysis/fastrp.csv /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp/logs/run_20250929_150335.log