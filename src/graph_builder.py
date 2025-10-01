from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial import Voronoi

def build_knn(df: pd.DataFrame, k: int = 8) -> nx.Graph:
    
    node_ids = [int(i) for i in df["cell_id"].values]
    coords = df[["x","y"]].values
    tree = cKDTree(coords)
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for i, pt in enumerate(coords):
        dists, idxs = tree.query(pt, k=k+1)
        for j in idxs[1:]:
            G.add_edge(node_ids[i], node_ids[j])
    return G

def build_radius(df: pd.DataFrame, radius: float) -> nx.Graph:
    
    node_ids = [int(i) for i in df["cell_id"].values]
    coords = df[["x","y"]].values
    tree = cKDTree(coords)
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for i, pt in enumerate(coords):
        idxs = tree.query_ball_point(pt, radius)
        for j in idxs:
            if i < j:
                G.add_edge(node_ids[i], node_ids[j])
    return G

def build_gabriel(df: pd.DataFrame) -> nx.Graph:
    
    node_ids = [int(i) for i in df["cell_id"].values]
    coords = df[["x","y"]].values

    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for a, b in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
            u, v = sorted((int(a), int(b)))
            edges.add((u, v))

    tree = cKDTree(coords)
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for (u, v) in edges:
        mid = 0.5 * (coords[u] + coords[v])
        r = 0.5 * np.linalg.norm(coords[u] - coords[v]) + 1e-12
        if r <= 0:  # degenerate edge (identical points)
            continue
        idxs = tree.query_ball_point(mid, r - 1e-12)
        # If any other point lies inside the Gabriel circle, skip
        if set(idxs) - {u, v}:
            continue
        # Map local indices back to global df.index labels
        G.add_edge(node_ids[u], node_ids[v])

    return G

def build_voronoi(df: pd.DataFrame) -> nx.Graph:
    
    node_ids = [int(i) for i in df["cell_id"].values]
    coords = df[["x","y"]].values
    vor = Voronoi(coords)
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    # ridge_points are index pairs (i, j) whose Voronoi cells share an edge ⇒ Delaunay edge
    for i, j in vor.ridge_points:
        if i == j:
            continue
        u, v = node_ids[i], node_ids[j]
        if u != v:
            G.add_edge(u, v)
    return G

def build_graph(df: pd.DataFrame, kind: str = "gabriel", **kwargs) -> nx.Graph:
    kind = (kind or "gabriel").lower()
    if kind == "knn":
        return build_knn(df, k=int(kwargs.get("k", 8)))
    if kind == "radius":
        return build_radius(df, radius=float(kwargs.get("radius", 25.0)))
    if kind == "voronoi":
        return build_voronoi(df)
    if kind == "gabriel":
        return build_gabriel(df)
    raise ValueError(f"Unknown graph type: {kind}")
