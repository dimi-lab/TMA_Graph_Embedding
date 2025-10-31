from __future__ import annotations
import pandas as pd
import pyreadr
from typing import Dict, List
import logging
import pdb

def read_cells_rds(path: str, colmap: Dict[str,str]) -> pd.DataFrame:
    res = pyreadr.read_r(path)
    df = next(iter(res.values()))
    rename = {
        colmap.get("x","x"): "x",
        colmap.get("y","y"): "y",
        colmap.get("phenotype","phenotype"): "phenotype",
        colmap.get("roi_id","roi_id"): "roi_id",
        colmap.get("cell_id","cell_id"): "cell_id",
    }
    df = df.rename(columns=rename)
    missing = [k for k in ["x","y","phenotype","roi_id","cell_id"] if k not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cells RDS after renaming: {missing}")
    return df

def read_roi_labels_csv(path: str, colmap: Dict[str,str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if colmap.get('patient_id') is not None:
        rename = {
            colmap.get("roi_id","roi_id"): "roi_id",
            colmap.get("patient_id","patient_id"): "patient_id",
        }
        df = df.rename(columns=rename)

        if not set(["roi_id","patient_id"]).issubset(df.columns):
            raise ValueError("roi_id labels must contain roi_id, and patient_id")
        return df
    else:
        rename = {
            colmap.get("roi_id","roi_id"): "roi_id",
            colmap.get("roi_label","roi_label"): "roi_label",
        }
        df = df.rename(columns=rename)
        # remove rows with roi_label is null
        out = df[df["roi_label"].notna()]
        logging.info(f"Removed {df[df['roi_label'].isna()].shape[0]} rows with roi_label is null")
        if not set(["roi_id","roi_label"]).issubset(df.columns):
            raise ValueError("roi_id labels must contain roi_id, and roi_label")
   
        return out
def read_subject_labels_csv(path: str, colmap: Dict[str,str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {
        colmap.get("patient_id","patient_id"): "patient_id",
        colmap.get("patient_label","patient_label"): "patient_label",
    }
    df = df.rename(columns=rename)
    # remove rows with patient_id is null
    df = df[df["patient_label"].notna()]
    logging.info(f"Removed {df[df['patient_label'].isna()].shape[0]} rows with patient_label is null")
    if not set(["patient_id","patient_label"]).issubset(df.columns):
        raise ValueError("Subject labels must contain patient_id and patient_label")
    return df

def attach_labels(df_cells: pd.DataFrame, df_roi: pd.DataFrame, df_subj: pd.DataFrame | None = None) -> pd.DataFrame:
    # keep only rows whose roi_id exists in both df_cells and df_roi
    out = df_cells.merge(df_roi, on="roi_id", how="inner")
    if df_subj is not None:
        # keep only rows whose patient_id exists in both out and df_subj
        out = out.merge(df_subj, on="patient_id", how="inner")
        out['roi_label'] = out['patient_label'] # roi_id level label is the same as patient level label
    return out

def validate_df(df: pd.DataFrame, required_cols: List[str] =["x","y","phenotype","roi_id","cell_id","roi_label"]) -> bool:

    missing = [k for k in required_cols if k not in df.columns]
    if missing:
        print(f"Missing required columns in df: {missing}")
        return False
    return True