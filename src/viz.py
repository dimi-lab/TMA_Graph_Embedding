from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

def plot_cells(df: pd.DataFrame, ax=None, s=4, alpha=0.8):
    if ax is None:
        fig, ax = plt.subplots()
    for ct, d in df.groupby("phenotype"):
        ax.scatter(d["x"], d["y"], s=s, alpha=alpha, label=str(ct))
    #ax.set_aspect("equal")
    #ax.legend(title="Phenotype", fontsize=8)
    #ax.set_xlabel("x"); ax.set_ylabel("y")
    return ax

def plot_graph(G: nx.Graph, df: pd.DataFrame, ax=None, node_size=6, alpha=0.7):
    if ax is None:
        fig, ax = plt.subplots()
    pos = {i:(row.x, row.y) for i, row in df[["x","y"]].iterrows()}
    nx.draw_networkx_edges(G, pos=pos, ax=ax, width=0.5, alpha=alpha)
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=node_size, alpha=alpha)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return ax
