# Current participant-nested pipeline

Use [the repository quick start](../../README.md) for installation and exact prepare/train/audit/report commands. Read [private input requirements](../../docs/data_format.md) before preparing data. The single entry point is `python code/run_pipeline.py --help`.

## Modules

| Module | Responsibility |
| --- | --- |
| `data.py` | Raw matrix loading, clinical linkage and conservative modality pairing |
| `splits.py` | Frozen participant outer/inner/stopping splits and batch support |
| `preprocessing.py` | Shared frozen training-only modality transforms |
| `engine.py` | Nested Ridge/GINE evaluation, refits, provenance and resume checks |
| `gnn.py` | Signed graph construction, GINE, stopping and fixed-epoch refit |
| `calibration.py` | CN inner-OOF global/cohort age-bias correction |
| `audit.py` | Independent split, fit-index, saved-state and calibration checks |
| `reporting.py`, `slides.py` | Aggregate reports, diagnostics and editable methods slides |
| `diagnostics.py`, `downstream.py` | Training-fit PCA, fold-specific embeddings, original-checkpoint sensitivity and size curves |
| `clinical.py` | Verified-source clinical analysis interface |
| `tests.py` | Synthetic boundary and end-to-end tests |

Required helper modules remain in `../final/`. Their import names are retained for existing saved-model compatibility. The current pipeline does not depend on the deleted historical global-ComBat/GNN/fusion scripts.

## Evaluation scope

Five outer participant folds and three inner folds are used in full configurations. All learned preprocessing and selection stays within its training partition. GINE uses a separate stopping split within inner training, refits before inner validation, and fixes outer epochs from inner choices. Held-out age never enters the feature transform.

DWI uses log1p edges; signed fMRI uses Fisher-z edges. Training masks preserve original edge indices. GINE early fusion concatenates node profiles, uses two edge channels and the union of modality topologies. Ridge early fusion concatenates standardized modality features.

Calibration is fitted on selected-candidate CN inner OOF predictions from outer training. Global and cohort-aware outputs are retained; cohorts with fewer than ten reference scans or no age variation fall back to the global trend. Held-out age is used for scoring and BAG arithmetic.

The archived matched Ridge run and four fixed-capacity GINE arms completed and passed their artifact audits. Supported 75%/90% Ridge/GINE training-size runs completed; 25%/50%/60% runs failed batch-support preflight. `validation.json` is short-budget software validation only. The broader `default.json` joint search must be classified by its own run manifest rather than inferred from the fixed-capacity experiments.

Current cohort figures and selected six-model aggregates are in [reports/current](../../reports/current/README.md). Clinical/protocol/visit and ROI verification remains unresolved; no clinical or anatomical validity follows from implementation tests.
