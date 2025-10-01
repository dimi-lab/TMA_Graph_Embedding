from __future__ import annotations
import pandas as pd
import pyreadr
from typing import Dict
import logging
import pdb

def read_cells_rds(path: str, colmap: Dict[str,str]) -> pd.DataFrame:
    res = pyreadr.read_r(path)
    df = next(iter(res.values()))
    rename = {
        colmap.get("x","x"): "x",
        colmap.get("y","y"): "y",
        colmap.get("phenotype","phenotype"): "phenotype",
        colmap.get("roi_id","ROI"): "ROI",
        colmap.get("cell_id","cell_id"): "cell_id",
    }
    df = df.rename(columns=rename)
    missing = [k for k in ["x","y","phenotype","ROI","cell_id"] if k not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cells RDS after renaming: {missing}")
    return df

def read_roi_labels_csv(path: str, colmap: Dict[str,str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if colmap.get('patient_id') is not None:
        rename = {
            colmap.get("roi_id","ROI"): "ROI",
            colmap.get("roi_label","roi_label"): "roi_label",
            colmap.get("patient_id","patient_id"): "patient_id",
        }
        df = df.rename(columns=rename)
        # remove rows with roi_label is null
        out = df[df["roi_label"].notna()]
        logging.info(f"Removed {df[df['roi_label'].isna()].shape[0]} rows with roi_label is null")
        if not set(["ROI","patient_id","roi_label"]).issubset(df.columns):
            raise ValueError("ROI labels must contain ROI, roi_label, and patient_id")
    else:
        rename = {
            colmap.get("roi_id","ROI"): "ROI",
            colmap.get("roi_label","roi_label"): "roi_label",
        }
        df = df.rename(columns=rename)
        # remove rows with roi_label is null
        out = df[df["roi_label"].notna()]
        logging.info(f"Removed {df[df['roi_label'].isna()].shape[0]} rows with roi_label is null")
        if not set(["ROI","roi_label"]).issubset(df.columns):
            raise ValueError("ROI labels must contain ROI, and roi_label")
   
    return out
def read_subject_labels_csv(path: str, colmap: Dict[str,str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {
        colmap.get("patient_id","patient_id"): "patient_id",
    }
    df = df.rename(columns=rename)
    # remove rows with patient_id is null
    df = df[df["patient_id"].notna()]
    logging.info(f"Removed {df[df['patient_id'].isna()].shape[0]} rows with patient_id is null")
    if "patient_id" not in df.columns:
        raise ValueError("Subject labels must contain patient_id")
    return df

def attach_labels(df_cells: pd.DataFrame, df_roi: pd.DataFrame, df_subj: pd.DataFrame | None = None) -> pd.DataFrame:
    # keep only rows whose ROI exists in both df_cells and df_roi
    out = df_cells.merge(df_roi, on="ROI", how="inner")
    if df_subj is not None:
        # keep only rows whose patient_id exists in both out and df_subj
        out = out.merge(df_subj, on="patient_id", how="inner")
    return out
