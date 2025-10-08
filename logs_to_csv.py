#!/usr/bin/env python3
# logs_to_csv.py
from __future__ import annotations

import argparse, ast, json, re, sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

LINE_RE = re.compile(
    r"""
    ^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})
    \s+\|\s+INFO\s+\|\s+
    \[(?P<i>\d+)\/(?P<n>\d+)\]\s+
    (?P<method>[^\s]+)                # structure embedding method, e.g. fastrp
    \s*\+\s*
    (?P<aggregation>[^\s]+)           # aggregation method, e.g. set2set
    \s*->\s*
    (?P<score>-?\d+(\.\d+)?)
    \s*\|\s*params=(?P<params>\{.*\})\s*$
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
    attr_mode = params.get("attr_mode", None)

    return {
        "timestamp": g["ts"],
        "idx": int(g["i"]),
        "n_total": int(g["n"]),
        "method": g["method"],             # categorical
        "aggregation": g["aggregation"],   # categorical
        "score": float(g["score"]),
        "l1_ratio": float(params.get("l1_ratio", 1.0)),     # numeric
        "param.dim": pd.to_numeric(dim, errors="coerce"),  # numeric
        "param.attr_mode": attr_mode,                      # categorical (may be None)
        "param.weights_vec": weights_key,                  # categorical (unique vector token)
        "raw.params": g["params"],                         # keep original text for traceability
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out.resolve()}")

if __name__ == "__main__":
    main()

# python logs_to_csv.py --out analysis/fastrp_het.csv /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp_het/logs/run_20250929_145007.log
# python logs_to_csv.py --out analysis/fastrp.csv /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp/logs/run_20250929_150335.log