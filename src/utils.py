from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Namespace:
    """Small helper to access dict keys as attributes."""
    __dict__: Dict[str, Any]

def dict_to_ns(d: dict) -> Namespace:
    return Namespace(**{k: dict_to_ns(v) if isinstance(v, dict) else v for k,v in d.items()})

import pandas as pd
def align_df_with_G_all(df, G_all):
    roi_col = "ROI"
    id_col  = "cell_id"  
    # 1) Build the target index from the graph's nodes (keeps the graph's order)
    node_index = pd.MultiIndex.from_tuples(list(G_all.nodes()),
                                        names=[roi_col, id_col])
    # 2) Set a matching MultiIndex on df
    #    (fails loudly if duplicates — better than silently picking a row)
    df_idxed = df.set_index([roi_col, id_col])
    if df_idxed.index.has_duplicates:
        raise ValueError("Duplicates found in df")
        # pick the first occurrence per (ROI, cell_id); or use a different reducer
        #  = df_idxed[~df_idxed.index.duplicated(keep="first")]

    # 3) Reindex to align order exactly to G_all.nodes()
    df_aligned = df_idxed.reindex(node_index)
    # restore roi_col and id_col as columns
    df_aligned = df_aligned.reset_index()
    return df_aligned