from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx

def global_celltype_counts(df: pd.DataFrame) -> pd.DataFrame:
    c = df.groupby(["ROI","phenotype"]).size().rename("n").reset_index()
    totals = df.groupby("ROI").size().rename("N").reset_index()
    out = c.merge(totals, on="ROI", how="left")
    out["prop"] = out["n"] / out["N"]
    return out

def basic_graph_metrics(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degs = [d for _,d in G.degree()]
    return {
        "n_nodes": n,
        "n_edges": m,
        "avg_degree": float(np.mean(degs)) if degs else 0.0,
        "avg_clustering": float(nx.average_clustering(G)) if n else 0.0,
        "n_components": nx.number_connected_components(G),
    }
