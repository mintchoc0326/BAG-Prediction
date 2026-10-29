# Connectome Brain Age

Research pipelines for brain-age prediction and brain-age gap (BAG) analysis using diffusion MRI (DWI) and functional MRI (fMRI) connectomes from ADNI, HABS, ADRC, and ADDecode.

The current workflow includes NeuroCombat harmonization, Ridge regression, graph neural networks (GNNs), multimodal fusion, and checkpoint-based attribution.

## Repository layout

| Path | Purpose |
| --- | --- |
| `code/final/` | Current harmonization, brain-age, fusion, and attribution Python scripts |
| `code/final/slides/` | Presentation-generation scripts and their dependencies |
| `code/tools/` | PCA plots, figure maintenance, and local validation utilities |
| `data/README.md` | Local data layout and access expectations |
| `results/README.md` | Local output layout and provenance notes |
| `requirements.txt` | Direct Python dependencies for the analysis scripts |

Historical local notebooks and scripts remain excluded from this repository, except for the three scripts previously published on GitHub, which are preserved in `code/explore/github_legacy/`. These archived scripts are not current entry points. Current Python scripts are authoritative. Local compatibility symlinks are also excluded.

## Environment

Use a separate Python environment. Install PyTorch and PyTorch Geometric builds appropriate for your CPU/CUDA environment, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

This dependency list is not a locked environment and does not guarantee numerical reproduction. The source has been syntax-checked; a clean installation and complete training run have not been validated. Presentation tooling has additional dependencies in `code/final/slides/requirements.txt` and may require external applications.

## Data and paths

Datasets are not distributed with this repository. Obtain the necessary permissions separately and place local inputs under `data/final/{ADNI,HABS,ADRC,ADDecode}/`. Connectome, metadata, and atlas paths must match the configuration blocks in the scripts.

Several scripts contain the original absolute project path `/mnt/newStor/paros/paros_WORK/avery`. Before running on another machine, update their `ROOT`, `PROJECT_DIR`, and other path settings. Run commands from the repository root. There is no unified command-line interface; some scripts execute analysis when invoked and overwrite their configured outputs.

## Workflow

Run the stages in order after configuring inputs and paths:

| Stage | Script in `code/final/` | Local output |
| --- | --- | --- |
| DWI harmonization | `run_DWI_harmonization.py` | `results/final/DWI/` |
| fMRI harmonization | `run_fMRI_harmonization.py` | `results/final/fMRI/` |
| DWI Ridge / GNN | `Ridge_GNN_BAG.py`, `GNN_DWI_BAG.py` | DWI `Ridge_BAG/`, `GNN_BAG/` |
| fMRI Ridge / GNN | `Ridge_fMRI_BAG.py`, `GNN_fMRI_BAG.py` | fMRI `Ridge_BAG/`, `GNN_BAG/` |
| Pairing and metadata audit | `prepare_DWI_fMRI_fusion_metadata.py` | `results/final/Fusion/metadata/` |
| Ridge fusion | `Ridge_DWI_fMRI_fusion.py` | Fusion `Ridge_baseline/` |
| Weighted Ridge fusion | `Ridge_DWI_fMRI_weighted_fusion.py` | Fusion `Weighted_Ridge/` |
| GNN fusion | `GNN_DWI_fMRI_fusion.py` | Fusion `GNN_fusion/` |
| Attribution | `GNN_BAG_interpretability.py` | Fusion `Attribution/` |
| ROI order audit | `audit_connectome_roi_order.py` | Fusion `roi_order_audit/` |

For example, these commands recompute the primary harmonization outputs:

```bash
python code/final/run_DWI_harmonization.py
python code/final/run_fMRI_harmonization.py
python code/tools/plot_harmonization_pca_by_dataset.py
```

Attribution requires original outer-fold checkpoints and matching saved predictions. Those checkpoints were absent from the local project when this repository was prepared. Archived reconstructed-model attribution is not output from the current verified-checkpoint workflow.

`fmri_bag_cv.py` and `fusion_transformers.py` provide shared model components. Some tools, including `validate_organization.py` and slide utilities, require local historical backups or presentation artifacts excluded from Git.

## Publication scope

Git tracks current source code and documentation only. Raw/derived data, participant metadata, predictions, checkpoints, presentations, caches, environment folders, historical notebooks, and audit backups remain local. `.gitignore` uses explicit allowlists for `code/`, `data/`, and `results/` to keep generated or participant-level files out of future commits.

No dataset redistribution rights or software license are granted by this README. Add an appropriate license once the project owner has chosen one.

## Reproduce the English tables and performance figure

A single script, [`code/reporting/build_results_report.py`](code/reporting/build_results_report.py), computes MAE, RMSE, R², and Pearson r from existing raw CN out-of-fold predictions. It generates the six-model matched-cohort figure, aggregate tables, full-population unimodal supplements, and every native table from both reviewed slide decks.

```bash
python code/reporting/build_results_report.py
# Reproduce the 46 presentation tables without private prediction files:
python code/reporting/build_results_report.py --tables-only --output /tmp/bag-slide-tables
```

See [`reports/brain_age/README.md`](reports/brain_age/README.md) for data sources, confidence intervals, model definitions, and limitations. Only aggregate reports are published; individual predictions remain local. Presentation tables are embedded display-value snapshots, not automatic recomputations of every historical analysis.
