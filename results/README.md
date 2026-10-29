# Local results (not distributed)

- `final/DWI/`: DWI harmonization and unimodal Ridge/GNN outputs.
- `final/fMRI/`: fMRI harmonization and unimodal Ridge/GNN outputs.
- `final/Fusion/`: pairing metadata, Ridge/GNN fusion, ROI audits, and attribution.
- `explore/`: historical analyses and reconstructed-model attribution.
- `checks/`: local verification runs and organization backups.
- `presentation/`: generated slides, PDFs, and exported figures.

These outputs are excluded from Git because they can include participant-level information, model artifacts, and large generated files. Current code may be newer than available local outputs. Attribution requires original outer-fold checkpoints and matching predictions; archived reconstructed-model attribution does not establish successful execution of the current workflow.
