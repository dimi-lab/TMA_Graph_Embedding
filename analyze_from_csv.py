#!/usr/bin/env python3
# analyze_from_csv.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pearson_r(x: pd.Series, y: pd.Series) -> float | np.nan:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])

def summarize_impacts(
    df: pd.DataFrame,
    score_col: str = "score",
    numeric_cols: list[str] = ["param.dim"],
    categorical_cols: list[str] = ["method","aggregation","param.attr_mode","param.weights_vec"],
) -> pd.DataFrame:
    out_rows = []
    global_mean = df[score_col].mean()

    # numeric: Pearson r
    for col in numeric_cols:
        if col in df.columns:
            r = pearson_r(df[col], df[score_col])
            out_rows.append({
                "feature": col,
                "kind": "numeric",
                "metric": "pearson_r",
                "value": r,
                "n": int(df[[col, score_col]].dropna().shape[0]),
                "note": "",
            })

    # categorical: (top mean − global mean) and spread
    for col in categorical_cols:
        if col in df.columns:
            cats = df[col].astype("category")
            by = df.groupby(cats, dropna=False)[score_col].agg(["mean","count"]).sort_values("mean", ascending=False)
            if not by.empty:
                top_level = by.index[0]
                out_rows.append({
                    "feature": col,
                    "kind": "categorical",
                    "metric": "top_mean_minus_global",
                    "value": float(by.iloc[0]["mean"] - global_mean),
                    "n": int(by["count"].sum()),
                    "note": f"top={top_level}",
                })
    return pd.DataFrame(out_rows).sort_values(["kind","value"], ascending=[True, False])

def boxplot_by_category(df: pd.DataFrame, cat_col: str, score_col: str, out: Path):
    fig = plt.figure(figsize=(10, 6))
    cats = df[cat_col].astype("category")
    cat_levels = list(cats.cat.categories)
    groups = [df.loc[cats == c, score_col].dropna().values for c in cat_levels]
    plt.boxplot(groups, labels=[str(c) for c in cat_levels], showfliers=False)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(score_col)
    plt.xlabel(cat_col)
    plt.title(f"{score_col} by {cat_col}")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)

def bar_means_by_category(df: pd.DataFrame, cat_col: str, score_col: str, out: Path, top_n: int = 20):
    stats = (
        df.groupby(df[cat_col].astype("category"), dropna=False)[score_col]
          .agg(["mean","count"])
          .sort_values("mean", ascending=False)
          .head(top_n)
          .reset_index()
    )
    fig = plt.figure(figsize=(10, 6))
    y = np.arange(len(stats))
    plt.barh(y, stats["mean"].values)
    plt.yticks(y, [str(x) for x in stats[cat_col]])
    plt.gca().invert_yaxis()
    plt.xlabel("mean score")
    plt.title(f"Top {min(top_n,len(stats))} {cat_col} levels by mean score")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)

def scatter_vs_dim(df: pd.DataFrame, dim_col: str, score_col: str, out: Path):
    # simple scatter (dim is numeric)
    x = pd.to_numeric(df[dim_col], errors="coerce")
    y = pd.to_numeric(df[score_col], errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(x[m], y[m], alpha=0.6)
    r = np.corrcoef(x[m], y[m])[0,1]
    plt.xlabel(dim_col)
    plt.ylabel(score_col)
    plt.title(f"{score_col} vs {dim_col} (r={r:.3f})")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Analyze hyperparameter impacts from a parsed CSV.")
    ap.add_argument("csv", nargs="+", help="CSV file(s), directory(ies), or glob(s) produced by logs_to_csv.py")
    ap.add_argument("--outdir", type=Path, default=Path("analysis/plots"))
    ap.add_argument("--score-col", default="score")
    # choose which columns are treated as numeric / categorical
    ap.add_argument("--numeric", nargs="*", default=["param.dim"])
    ap.add_argument("--categorical", nargs="*", default=["method","aggregation","param.attr_mode","param.weights_vec"])
    ap.add_argument("--topn-cats", type=int, default=20, help="Top-N categories to show in bar plots")
    args = ap.parse_args()

    csv_files = args.csv
    if not csv_files:
        raise SystemExit("No CSV inputs found.")

    # Load & concat
    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        df = df.assign(source_csv=str(f))
        frames.append(df)
    df = pd.concat(frames, axis=0, ignore_index=True)


    # Impacts table
    impacts = summarize_impacts(
        df,
        score_col=args.score_col,
        numeric_cols=args.numeric,
        categorical_cols=args.categorical,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    impacts_path = args.outdir / "impacts_summary.csv"
    impacts.to_csv(impacts_path, index=False)

    # Plots
    # 1) scatter for dim
    if "param.dim" in df.columns:
        scatter_vs_dim(df, "param.dim", args.score_col, args.outdir / "scatter_score_vs_dim.png")

    # 2) categorical boxplots + bar(means)
    for cat_col in args.categorical:
        if cat_col in df.columns and df[cat_col].notna().any():
            # Only boxplot if not too many levels (<= 30), else skip to keep readable
            if df[cat_col].astype("category").nunique() <= 30:
                boxplot_by_category(df, cat_col, args.score_col, args.outdir / f"box_{cat_col}.png")
            bar_means_by_category(df, cat_col, args.score_col, args.outdir / f"bar_{cat_col}.png", top_n=args.topn_cats)

    print(f"Saved impacts to: {impacts_path.resolve()}")
    print(f"Figures saved to: {args.outdir.resolve()}")

if __name__ == "__main__":
    main()

# python analyze_from_csv.py analysis/fastrp_het.csv --outdir analysis/plots/fastrp_het
# python analyze_from_csv.py analysis/fastrp.csv --outdir analysis/plots/fastrp
# python analyze_from_csv.py analysis/fastrp_het.csv analysis/fastrp.csv --outdir analysis/plots/fastrp_het_fastrp