# TMA22 Skeleton (Config-Driven)

A **lean, extensible** code skeleton that wires:
- **Data I/O**: read `.rds` cell table (x, y, phenotype, ROI) with configurable column names; ROI-level and subject-level labels.
- **Graphs**: kNN / radius / Delaunay→Gabriel.
- **Embeddings**: *skeleton* interface for structure (e.g., node2vec) and node-attribute embeddings.
- **Fusion**: choose strategy by argument (e.g., concat) or pass a custom callable.
- **Aggregation (MIL)**: placeholder choices (mean/attention-stub) passed as arguments.
- **Auxiliary losses**: argument-driven hook (returns 0 by default).

> placeholder
> placeholder

## Quickstart
```bash
conda env create -f environment.yml
conda activate tma22-skeleton
jupyter lab notebooks/00_wireup_demo.ipynb
```

## Config
Edit `config/config.yaml` to point to your files and name your columns.
