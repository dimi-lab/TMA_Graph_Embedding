# README

## Purpose
`main.py` executes the full **embedding → aggregation → ROI-level classification** pipeline for multiplexed imaging (mxIF) datasets.  
It loads cell tables and ROI labels, builds per-ROI graphs, computes node embeddings, aggregates them into ROI-level embeddings, and trains/evaluates ROI classifiers.

## Inputs
The workflow is driven by a YAML config file. Main input components include:

### 1. Cell-level data
- `.rds` or `.csv` file containing per-cell `x`, `y`, `phenotype`, `roi_id`, `cell_id`.
- Column names are mapped through `cell_columns`.

### 2. ROI-level labels
- CSV mapping `roi_id` → `roi_label`.  
- Optionally supports `patient_id` and subject-level labels.

### 3. Graph configuration
- Graph type: `knn` or `radius`
- Parameters such as `knn_k` or `radius`.

### 4. ROI-supervision configuration
- Embedding parameter search space  
- Aggregation method (mean / attention / pooling)  
- ROI classifier and CV settings.

## Outputs
All results are written under the directory specified by `--outdir`. Key outputs:

### Data
- `dataframes/df.csv` — merged and validated cell table with ROI/subject labels.

### Graphs
- `graphs/graph_dict_<type>.pkl` — per-ROI graphs.  
- `graphs/G_all_<type>.pkl` — disconnected union graph.

### Embeddings & Classification
- Cached node embeddings.  
- ROI embeddings stored under:
  ```
  evaluate/roi_supervised_best/<metric>/roi_embedding.mat
  ```
- Best hyperparameters (`best_roi_supervision.yaml`).  
- Trained classifier objects.

### Logs & Config
- `logs/run_*.log` — timestamped logs.  
- `config/resolved_config.yaml` — exact config used in the run.

## Pipeline Summary
1. Load and validate cell + ROI/subject labels  
2. Build per-ROI graphs  
3. Construct global union graph  
4. Run supervised search over embedding + aggregation hyperparameters  
5. Compute best node embeddings  
6. Aggregate node embeddings into ROI-level vectors  
7. Train and export ROI-level classifiers  
8. Save embeddings, parameters, and diagnostics  

