# Current workflow

Run from the project root after installing dependencies and configuring the paths described in the root README. Only Python sources are authoritative; historical notebooks are excluded from Git.

| Stage | Entry point | Output |
|---|---|---|
| DWI harmonization | `run_DWI_harmonization.py` | `results/final/DWI` |
| fMRI harmonization | `run_fMRI_harmonization.py` | `results/final/fMRI` |
| DWI Ridge / GNN | `Ridge_GNN_BAG.py`, `GNN_DWI_BAG.py` | modality `Ridge_BAG`, `GNN_BAG` |
| fMRI Ridge / GNN | `Ridge_fMRI_BAG.py`, `GNN_fMRI_BAG.py` | modality `Ridge_BAG`, `GNN_BAG` |
| Pairing | `prepare_DWI_fMRI_fusion_metadata.py` | `results/final/Fusion/metadata` |
| Ridge fusion | `Ridge_DWI_fMRI_fusion.py` | fusion `Ridge_baseline` |
| Weighted Ridge | `Ridge_DWI_fMRI_weighted_fusion.py` | fusion `Weighted_Ridge` |
| GNN fusion | `GNN_DWI_fMRI_fusion.py` | fusion `GNN_fusion` and original fold checkpoints |
| Attribution | `GNN_BAG_interpretability.py` | fusion `Attribution` (requires original fold checkpoints) |
| ROI audit | `audit_connectome_roi_order.py` | fusion `roi_order_audit` |

`fmri_bag_cv.py` and `fusion_transformers.py` are importable model dependencies and remain alongside entry points. Paths changed; model algorithms and selection policies did not.

Example:

```bash
.cluster_storage/conda_envs/harmonization/bin/python code/final/run_DWI_harmonization.py
.cluster_storage/conda_envs/harmonization/bin/python code/final/run_fMRI_harmonization.py
.cluster_storage/conda_envs/harmonization/bin/python code/tools/plot_harmonization_pca_by_dataset.py
```

The two harmonization commands recompute primary results. The PCA utility reads the current `results/final/{DWI,fMRI}` outputs and writes each modality’s plots under `harmonization_evaluation/`. Do not rerun legacy scripts to regenerate current results.

Slide builder history is in `slides/`. The reviewed presentation is the PPTX/PDF in `results/presentation`; the reconciliation tool in `code/tools/reconcile_slides.py` applies the current code audit and refreshed PCA figures.
