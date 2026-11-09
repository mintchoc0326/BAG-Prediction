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

## GINE capacity diagnostics and clinical inventory

`GINE_capacity_study.py` compares two smaller GINE models (2 layers × 32 channels and 3 × 32) against the current 3 × 64 architecture. It uses identical participant-grouped splits, a fixed 50% positive-edge topology, training-only target scaling, and validation-MAE early stopping. It records evaluation-mode training/validation MAE and MSE each epoch, fold metrics, model parameter counts, runtime, checkpoints, and private split/prediction records in an isolated output directory.

```bash
python code/final/GINE_capacity_study.py --output results/final/DWI/GINE_capacity_study_new
# Rebuild MAE/loss plots and the HTML report after training:
python code/final/GINE_capacity_study.py --output results/final/DWI/GINE_capacity_study_new --render-only
```

This is an exploratory capacity diagnostic: existing global ComBat and a previously selected graph threshold are reused. It does not fix the harmonization leakage or establish a winning architecture. The existing HABS all-CN override conflicts with harmonized source diagnosis labels; diagnosis provenance and clinical visit alignment require review. One seed is used; across-fold SD is not a confidence interval. Existing GNN_BAG outputs are preserved. An output directory with a run manifest is not overwritten.

The 4 November 2026 run is under `results/final/DWI/GINE_capacity_study_20261104/`. Its `study_source_snapshot.py` matches the training-source SHA-256 in the manifest; presentation-only report rendering was added while training was running.

The clinical inventory generator requires pandas and reads the four canonical local metadata CSVs:

```bash
python code/reporting/build_clinical_inventory.py
```

Outputs are local under `results/final/clinical_inventory_20261104/`: full field inventory, candidate cross-cohort measures, exact-key imaging linkage coverage, source diagnosis counts, HTML report, and a preliminary cBAG plan. Counts describe source rows or explicitly labeled unique exact-key matches; duplicated keys are excluded rather than resolved by row order. Clinical assessment timing, scale definitions, units, sentinel codes, and composite derivation remain unverified. The inventory exports aggregate counts and field names, not participant records. It does not fit clinical associations. Alternative raw/Excel sources are not yet inventoried.

## Fold-contained DWI Ridge implementation (6 November 2026)

This new entry point does not overwrite the historical harmonization or Ridge scripts. It reads original raw connectome files using the archived DWI eligibility manifest, matches canonical source participant IDs and diagnoses by unique connectome key, and excludes ambiguous joins. The HABS all-CN override is not used. Labels and visit matching are still provisional; independent source verification remains necessary.

```bash
# Use the project environment containing neuroCombat and scikit-learn.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/final/test_foldwise_dwi.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/final/run_foldwise_DWI_Ridge.py --output results/final/DWI/foldwise_Ridge_new
python code/final/slides/build_foldwise_validation_slides.py --run results/final/DWI/foldwise_Ridge_new
```

The runner evaluates three prespecified arms on identical participant folds: no ComBat; batch-only location/scale ComBat; and mean-only ComBat estimated using training ages. In the additive mode, the transformation is `x_harmonized = x - training_residual_SD * batch_offset`; age terms cancel because batch scales are one. Training/apply equivalence is checked against the installed neuroCombat output. Held-out age never enters feature transformations. Sex, diagnosis, genotype and metabolic values are not predictor features. The method is an age-informed additive correction, not a full location/scale correction preserving arbitrary unseen covariates. Batch-only ComBat is a baseline, not a promise that biological variation survives.

Outer CV is participant-grouped and batch-stratified. Inner grouped CV fits edge masks, ComBat and StandardScaler independently before alpha selection. Selected inner-OOF CN predictions estimate the BAG-age trend; the chosen pipeline is refit on outer-training CN and applied to all held-out participants. Raw age predictions provide MAE/RMSE/R²/r; cBAG additionally uses the held-out age only in the downstream BAG/correction arithmetic. Unknown batches are unsupported for harmonization and raise errors. A fixed feasibility rule restricts eligibility to batches with at least five source-labeled CN participants; the splitter selects the first seed with training support based only on groups, batches and reference eligibility, never ages/errors.

Current restricted population: 919 eligible scans / 670 participants, including 723 CN scans / 546 CN participants. Excluded from the archived 1,234-scan universe: 127 ADNI scans with insufficient batch reference support and 188 ADRC scans with ambiguous clinical keys. These are not matched to the old published benchmark; comparisons across those populations must not be treated as changes due solely to leakage correction.

Local run: `results/final/DWI/foldwise_Ridge_validated_20261106/`. It includes an HTML report, all outer OOF predictions, cBAG, inner search scores, calibration provenance, preprocessing checks, eligibility counts, scanner/sex probes, residual-age correlations, model artifacts and source snapshots/hashes. Participant files remain local under ignored `results/`. Fixed-regularization scanner and sex probes are descriptive; scanner predictability can reflect age/cohort confounding. Clinical labels, acquisition/assessment dates, canonical identity accuracy, sparse-site exclusions and ADRC protocol consistency remain limitations. GINE, fMRI and fusion have not yet been migrated to this entry point.

Tests use synthetic data to check exact ComBat fit/apply equivalence, independence from companion held-out samples, invariance to altered outer-test ages, group separation, complete OOF coverage, and training-only bias-calibration provenance. These implementation checks do not validate the clinical source data or establish biological preservation.

To independently check completed fold artifacts, run `python code/final/audit_foldwise_run.py --run results/final/DWI/foldwise_Ridge_validated_20261106`. It verifies OOF coverage, participant separation, shared masks, CN-only fitting, calibration provenance, cBAG arithmetic and training-source snapshot hashes, then writes an aggregate `validation_checks.json`. This is a runtime-artifact audit, complementary to the original manual source audit.

## fMRI fold-contained Ridge (6 November 2026)

`run_foldwise_fMRI_Ridge.py` now uses the same nested-CV controller as DWI, with `FmriPreprocessor` in `foldwise_fmri.py`. Signed raw Pearson correlations are preserved; a fixed clipped Fisher-z transform replaces DWI log1p. The shared ComBat/scaling core is fitted separately within each training split. Negative correlations are never set to zero. The archived 847-row fMRI eligibility table supplies raw file paths and acquisition batches; no harmonized feature array is read. Canonical clinical labels and participant IDs are matched through unique keys, without an all-HABS CN override.

```bash
OPENBLAS_NUM_THREADS=1 python code/final/test_foldwise_fmri.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/final/run_foldwise_fMRI_Ridge.py --output results/final/fMRI/foldwise_Ridge_new
python code/final/audit_foldwise_run.py --run results/final/fMRI/foldwise_Ridge_new
python code/final/slides/build_foldwise_validation_slides.py --run results/final/fMRI/foldwise_Ridge_new --output reports/pipeline_audit/fmri_foldwise_validation_new
```

Completed local run: `results/final/fMRI/foldwise_Ridge_validated_20261106/`. Eligible: 575 scans / 447 participants; CN evaluation: 441 scans / 353 participants. Exclusions: 89 ADNI scans with insufficient CN batch support; 183 ADRC scans with ambiguous clinical keys. The three arms share identical fMRI folds and feature masks. Modality-specific participant populations/folds differ from the DWI run; these results are not a matched cross-modality comparison and cannot be used directly for fusion. A shared paired-sample evaluation remains a separate step.

Pooled CN OOF MAE: no ComBat 7.428 years; batch-only ComBat 8.780; training-age mean-only ComBat 7.488. These are descriptive comparisons without model-refitting uncertainty or a significance claim. Held-out ages never enter feature transformations; they are used for scoring and downstream BAG/cBAG arithmetic. Clinical source validity, scan/assessment matching, independent identity review and ADRC protocol verification remain unresolved. fMRI GINE and fusion have not yet been migrated.

The three fMRI synthetic tests check signed Fisher-z/domain behavior, fit/apply equivalence and age isolation, and an end-to-end run audited independently. The four DWI tests also passed after the shared-core refactor. Completed real-run artifacts passed OOF coverage, participant separation, shared-mask, training-only calibration, cBAG arithmetic and snapshot-hash checks. Editable fMRI slides are local under `reports/pipeline_audit/fmri_foldwise_validation_20261106/`.

## Unified project entry point

The current implementation is now `python code/run_pipeline.py`; see [`../brainage_pipeline/README.md`](../brainage_pipeline/README.md). Shared ComBat lives in `harmonization.py`, with compatibility classes retained for saved foldwise DWI/fMRI models. Common paired folds now support Ridge, GINE and early fusion. The original foldwise Ridge scripts above remain reproducible modality-wide development baselines. The new matched population is smaller because cross-modality age conflicts are excluded explicitly. Current aggregate reports and slides are under `reports/new_pipeline/`; old benchmarks remain historical.
