# Connectome Brain Age

Current participant-nested brain-age prediction from DWI, signed fMRI and paired early fusion. Ridge is the primary baseline; GINE capacity and residual arms use the same paired population and participant folds.

This branch contains the current implementation, its required support modules, configuration, tests and the current cohort-comparison figure. Historical model scripts, presentations and reports have been removed from this branch. Runtime data and earlier local work are not published.

## Repository map

| Path | Purpose |
| --- | --- |
| `code/run_pipeline.py` | Main entry point for all current stages |
| `code/brainage_pipeline/` | Preparation, participant splits, preprocessing, Ridge/GINE evaluation, calibration, audits and downstream analysis |
| `code/brainage_pipeline/configs/` | Ridge, GINE capacities, short validation and clinical-specification configurations |
| `code/final/` | Required fit/apply harmonization, input-loader and grouped-split support; retained module names preserve saved-checkpoint compatibility |
| `code/reporting/plot_model_comparison.py` | Rebuild the cohort four-panel figure using published aggregate metrics |
| `code/reporting/export_paired_metrics.py` | Export current aggregates from audited original Ridge/GINE OOF runs |
| `code/reporting/` | Current study orchestration, summary and clinical-inventory utilities |
| `docs/data_format.md` | Required private inputs, column names and matrix format |
| `reports/current/` | Current six-model aggregate metrics, figure exports and provenance |

## 1. Install

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python code/run_pipeline.py --help
```

Select a PyTorch build appropriate for your CPU/CUDA environment before installing the remaining dependencies if GPU support is required. CUDA is optional; the GINE implementation selects CUDA when available. Dependencies are not locked, so a fresh installation is not a guarantee of bitwise GPU reproduction. PPTX generation uses python-pptx; optional PDF conversion additionally requires LibreOffice.

## 2. Prepare authorized private inputs

Read [the input specification](docs/data_format.md) first. Raw imaging, participant metadata and the eligible-scan manifests are not distributed. The loader reads raw 84-by-84 connectome CSV files listed in private manifests; it does not use globally harmonized arrays as model inputs.

The preparation stage expects an input root containing `data/final/` and the two manifest paths documented below. It does not process DICOM/NIfTI or generate those manifests. Their archived eligibility restrictions remain a limitation.

```bash
python code/run_pipeline.py prepare \
  --root /absolute/path/to/private-input-root \
  --output results/current/prepared
```

If inputs are placed in the documented paths inside this checkout, omit `--root`. Matrix paths may be absolute or relative to the private input root. Choose a new output directory when inputs or preparation rules change; an existing prepared directory is rejected.

Preparation performs unique clinical-key matching, CN/batch eligibility checks, raw matrix loading and conservative DWI/fMRI pairing. The default age agreement tolerance is 0.01 years. Conflicting metadata are excluded and recorded, rather than averaged or resolved by selecting the first duplicate row. Matching keys do not verify acquisition dates.

## 3. Run the nested Ridge baseline first

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/run_pipeline.py train \
  --prepared results/current/prepared \
  --output results/current/ridge \
  --config code/brainage_pipeline/configs/ridge.json

python code/run_pipeline.py audit --run results/current/ridge
python code/run_pipeline.py report \
  --run results/current/ridge --output results/current/ridge_report
```

This runs five outer participant folds and three inner folds for DWI, fMRI and early fusion under no-ComBat and training-age mean-only ComBat. Ridge alpha is selected inside inner validation. Learned feature QC, harmonization and scaling are fitted on the corresponding training partition. All visits from a participant stay together.

The default population is paired. `--population DWI` or `--population fMRI` provides a modality-wide supplement; those populations must not be presented as matched multimodal comparisons.

## 4. Optional GINE evaluation

Use the fixed 2-by-32 arm for the six-model reference figure:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/run_pipeline.py train \
  --prepared results/current/prepared \
  --output results/current/gine_2x32 \
  --config code/brainage_pipeline/configs/gine_2x32.json

python code/run_pipeline.py audit --run results/current/gine_2x32
python code/run_pipeline.py report \
  --run results/current/gine_2x32 --output results/current/gine_report
```

The fixed-capacity configurations are `gine_1x16.json`, `gine_2x32.json`, `gine_3x64.json` and `gine_2x32_skip.json`. They select graph sparsity inside inner validation. `default.json` instead allows an inner search over capacities and residual settings. Completion of separate fixed-capacity runs does not establish completion of that broader combined search.

GINE stopping uses a separate participant split within inner training, followed by refitting before inner validation. Outer refitting uses a fixed epoch count derived from inner training. The maximum budget is 250 epochs with patience 25. `validation.json` uses a short budget to exercise software paths; its scores are not trained-model benchmarks.

## 5. Rebuild the cohort figure

The published aggregate table is sufficient; private inputs and model training are not required:

```bash
python code/reporting/plot_model_comparison.py
python code/reporting/plot_model_comparison.py \
  --mode none --output reports/current/paired_model_performance_no_combat
# Optional display without confidence intervals:
python code/reporting/plot_model_comparison.py \
  --no-ci --output results/current/paired_model_performance_no_ci
```

Outputs are PNG, PDF and editable-text SVG. Panels show MAE, RMSE, R-squared and Pearson r separately for ADNI, ADRC and HABS. The default is training-age mean-only ComBat. GINE is the prespecified 2-by-32 reference, not the best arm selected using outer-test scores. Fusion is current early fusion, not historical weighted fusion.

To regenerate the aggregate table from your completed original runs:

```bash
python code/reporting/export_paired_metrics.py \
  --ridge-run results/current/ridge \
  --gine-run results/current/gine_2x32 \
  --output reports/current
python code/reporting/plot_model_comparison.py
```

The exporter requires complete, audited, nonpilot five-fold paired runs with matching data, folds and evaluated observations. It exports aggregates and file fingerprints, not individual predictions. The current figure contains 237 CN scans from 189 participants: ADNI 125/77, ADRC 106/106 and HABS 6/6 (scans/participants). Bootstrap intervals use 3,000 cohort-stratified participant draws of fixed OOF predictions; they omit model-refitting and selection uncertainty.

## 6. Reports, slides and downstream stages

```bash
python code/run_pipeline.py slides \
  --run results/current/ridge --prepared results/current/prepared \
  --output results/current/slides

python code/run_pipeline.py diagnostics \
  --run results/current/ridge --prepared results/current/prepared \
  --output results/current/diagnostics

python code/run_pipeline.py embeddings \
  --run results/current/gine_2x32 --output results/current/embeddings
```

`attribute --help` describes original-checkpoint gradient-times-input sensitivity. Anatomical names require a verified 84-ROI ordering. Embedding projections are fitted within each fold's training space; raw coordinates from independently fitted models are not pooled.

`clinical --help` requires a reviewed source table and verified coding, units and visit alignment. `configs/clinical_spec.json` is intentionally unverified and cannot authorize clinical inference by itself. Diagnosis defines the CN training reference; sex, APOE and metabolic variables are not age-prediction features.

For training-size experiments, rerun `train` with `--fraction`, a new output directory and the same prepared data/configuration. Use `learning-curves --help` to summarize completed runs. Fractions refer to participant groups. Unsupported small-batch subsets fail explicitly. Learning curves report mean outer-fold MAE, which differs from pooled OOF MAE.

The additional study-driver and portal scripts retain the original experiment directory conventions; use the commands above for a new analysis in a new checkout.

## 7. Verification and resuming

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/test_pipeline.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/final/test_foldwise_dwi.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/final/test_foldwise_fmri.py
```

Tests cover age-free transforms, fit/apply equivalence, participant boundaries, signed graphs, early fusion, calibration provenance, input paths and clinical gates. To resume an unchanged run, repeat its `train` command with `--resume`. Changed code, inputs, settings or populations require a new run directory. This cleanup adds an explicit preparation input-root option; do not resume an older source fingerprint under changed code.

## Interpretation limits

Current results are development reanalyses, not untouched external validation. Training-age mean-only ComBat does not prove complete biological preservation. Evaluation requires known batches with sufficient CN support. Source diagnoses, independent identity verification, acquisition/clinical dates, ADRC protocol consistency and ROI order remain unresolved. HABS has only six matched CN scans. A passed implementation audit does not establish clinical or anatomical validity.

The repository does not distribute participant data or grant dataset redistribution permissions. Earlier files remain recoverable from Git history; local data, results and cleanup backups remain ignored.
