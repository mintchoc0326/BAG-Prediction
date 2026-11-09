# Current participant-nested brain-age pipeline

Use `python code/run_pipeline.py --help` from the repository root. This is the new predictive evaluation entry point. Original standalone scripts remain historical reproduction tools. Current aggregate results and editable slides are linked from `reports/new_pipeline/index.html`.

## Implemented scope

- DWI, signed fMRI and paired DWI–fMRI early fusion.
- Ridge and GINE with identical paired observations and frozen participant folds.
- No-ComBat control, batch-only sensitivity mode and training-age mean-only ComBat. The primary candidate is additive age-informed ComBat; this is not proof of complete biological preservation.
- Inner-only feature QC, harmonization, scaling and configuration selection. GINE capacity, residual skips and per-scan sparsity are selected inside inner validation.
- GINE early stopping uses a separate participant split inside inner training. Preprocessing is fitted on the early-training partition; then preprocessing and model are refitted on all inner training for the selected epoch count before inner validation. Outer refit uses the median selected inner epoch, without outer early stopping.
- Training-only CN inner-OOF reference calibration, with both global and cohort-aware cBAG outputs. Sparse cohort reference samples explicitly fall back to the global trend. No held-out age enters feature transformations. Held-out age is used for scoring and downstream BAG/cBAG arithmetic.
- Original checkpoints, fold-specific held-out embeddings, gradient-times-input node/edge sensitivity and optional verified clinical overlays.
- Source/data fingerprints, resumable tasks and independent artifact audits. Runtime private data stay under ignored `results/`.

All models use imaging features. Batch metadata is needed to choose trained acquisition parameters; clinical diagnosis defines the normative training reference but is not a predictor. Age is a supervised target and may inform training-only additive offsets. Sex, APOE and metabolic variables do not enter age-prediction features.

## Data restrictions that remain unresolved

The current archived input universes contain 1,234 DWI and 847 fMRI scans. Unique clinical matching and known-batch CN support retain 919 DWI / 575 fMRI scans. There are 575 common scan keys, but 250 have age differences exceeding the prespecified 0.01-year tolerance. They are excluded without averaging or overwriting ages. Another seven lack sufficient paired CN batch support. The matched analysis therefore has 318 scans / 252 participants, including 237 CN scans. HABS contributes very few retained pairs. Dates, label provenance, independent identity confirmation, ROI order and ADRC protocols remain unverified. Matching keys and agreeing metadata do not establish contemporaneous acquisition.

Changing the agreement tolerance, imputing ages or resolving repeated clinical rows requires documented source evidence. The software does not silently make those decisions. These restricted matched results cannot be directly compared with the old six-model benchmark or the larger modality-wide results.

## Environment

Use the project environment containing NumPy, pandas, SciPy, scikit-learn, joblib, neuroCombat, PyTorch and PyTorch Geometric. Root `requirements.txt` lists direct dependencies; optional clinical inference needs statsmodels. Editable slide generation needs `python-pptx` from `code/final/slides/requirements.txt`. PDF export needs LibreOffice. Environments are not locked; run manifests and source snapshots provide provenance but do not guarantee bitwise GPU reproduction.

## Prepare, train, audit and report

Use a new output directory for each changed configuration. The examples below use fresh generic names rather than overwrite completed runs.

```bash
python code/run_pipeline.py prepare --output results/final/new_pipeline/prepared_new

# Full-budget matched Ridge controls and additive harmonization.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/run_pipeline.py train \
  --prepared results/final/new_pipeline/prepared_new \
  --output results/final/new_pipeline/ridge_new \
  --config code/brainage_pipeline/configs/ridge.json

python code/run_pipeline.py audit --run results/final/new_pipeline/ridge_new
python code/run_pipeline.py report --run results/final/new_pipeline/ridge_new --output reports/new_pipeline/ridge_new
python code/run_pipeline.py slides --run results/final/new_pipeline/ridge_new --prepared results/final/new_pipeline/prepared_new --output reports/new_pipeline/slides_new
```

`default.json` evaluates Ridge and GINE for DWI, fMRI and early fusion under no-ComBat and age-informed mean-only preprocessing: five outer folds, three inner folds, up to 250 epochs, patience 25. The GINE grid includes 1×16, 2×32 and 3×64 architectures plus a 2×32 residual ablation at two sparsity levels. This is a substantial long-training workload. It is code-ready; the current report does not claim that it has completed.

`validation.json` exercises GINE/fusion with two epochs. Its reports and figures explicitly say software validation. These metrics must not be used to compare trained models. `ridge.json` runs only Ridge, with the full nested Ridge search. Batch-only mode can be added as a declared sensitivity arm; preprocessing mode is not selected using outer-test scores.

To resume exactly the same run, repeat `train` with `--resume`. Changed settings, prepared inputs, source code or population are rejected rather than combined. Failed tasks without a completed task record are recomputed; completed tasks are retained. Changing source requires a new run. Source snapshots in old runs remain usable evidence even if current source later changes.

For modality-wide supplements, pass `--population DWI` or `--population fMRI`. These are not matched multimodal comparisons.

## Graph representation and early fusion

DWI uses fixed log1p edge features; fMRI uses fixed Fisher-z of signed correlations. Global feature statistics are not fitted before CV. Constant/invalid residual-variance masks and StandardScaler are learned on the corresponding training subset. Standardized feature vectors map back to their original ROI pairs; filtered entries are scaled zero and excluded from graph topology. Sparsity is a per-subject magnitude rank, with its fraction chosen by inner validation. DWI topology uses positive raw connections; fMRI keeps both signs. GINE node features are full standardized connectivity profiles; edge attributes retain signed transformed values. Fusion concatenates the two node profiles, uses two edge channels and the union of selected modality topologies. Ridge fusion concatenates independently standardized modalities. This baseline is early fusion, not the historical weighted or late-fusion implementation.

## Increasing training-set size

Repeat `train` with the same prepared data, configuration and outer folds, changing `--fraction` to 0.25, 0.5, 0.75 and 1. Training subsets are participant-based with minimum batch support. Fractions that cannot support the design fail explicitly. Each fraction redoes inner selection and training-reference calibration using only its subset.

```bash
python code/run_pipeline.py learning-curves --runs RUN_FRACTION_025 RUN_FRACTION_050 RUN_FRACTION_075 RUN_FRACTION_100 --output reports/new_pipeline/training_size
```

The summarizer rejects different held-out folds, data populations or configurations. Plotted x values are actual mean CN training scan counts; the specified fractions refer to participant groups.

## Embeddings and original-checkpoint interpretation

```bash
python code/run_pipeline.py embeddings --run GINE_RUN --output LOCAL_EMBEDDING_OUTPUT
python code/run_pipeline.py attribute --run GINE_RUN --task GINE_RUN/fold1/age_mean/DWI/gine --prepared PREPARED_DATA --output LOCAL_ATTRIBUTION_OUTPUT
```

Embedding PCA is fitted to training embeddings separately for each outer model, then applied to its held-out embeddings. Raw coordinates are never pooled across folds. Age, cohort, diagnosis and sex are validation overlays, not input features. Optional `--clinical-spec` enables verified genotype/metabolic overlays without using cBAG as an embedding input.

Attribution loads the original outer checkpoint and exact preprocessing state; it does not reconstruct/retrain a replacement model. Output is local absolute gradient-times-input sensitivity, not a causal explanation. Without verified ROI ordering, labels remain numeric. `--roi-labels` accepts a JSON object with `roi_order_verified: true` and exactly 84 `labels`. A pilot attribution test is technical validation only, not an anatomical finding.

## Clinical analysis and target inventory

The refreshed inventory is generated by:

```bash
python code/reporting/build_clinical_inventory.py --output results/final/new_pipeline/clinical_inventory_new
```

It includes candidate age, sex, cognitive status, APOE, diabetes type and HbA1c fields. Counts describe source rows or explicit exact-key subsets, not final clinically verified participants. Existing composite definitions and sentinel codes remain unverified.

`configs/clinical_spec.json` is deliberately unverified. Supply a reviewed imaging-linked source CSV, categorical code mappings, HbA1c units, T2D definition and clinical-visit alignment before enabling inference.

```bash
python code/run_pipeline.py clinical --run COMPLETED_NONPILOT_RUN --spec VERIFIED_SPEC.json --output LOCAL_CLINICAL_OUTPUT
```

The implemented analysis uses complete cases, cohort/age/sex covariates, participant-cluster uncertainty, group plots, continuous HbA1c/age checks and BH correction across the exploratory tests in that invocation. The code supports additional declared targets such as APOE genotype. Education/other confounders and longitudinal analyses require a prespecified extension. It does not infer T2D from generic diabetes, invent HbA1c units, impute clinical outcomes or run inference on pilot predictions.

## Tests and completed evidence

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/test_pipeline.py
python code/final/test_foldwise_dwi.py
python code/final/test_foldwise_fmri.py
```

Synthetic tests cover input/age isolation, train/apply equivalence, split boundaries, residual/signed graphs, early-fusion end-to-end execution, calibration provenance, source-safe resume and verified/gated clinical interfaces. The real completed Ridge run has 30 tasks; the precision-corrected GINE software validation has 15. Each is independently audited. Full-budget GINE, training-size experiments, clinical biological conclusions and anatomical claims remain pending rather than inferred from short-budget tests.

Current report hub: `reports/new_pipeline/index.html`. Rebuild its aggregate copies/status tables with `python code/reporting/build_pipeline_portal.py`. The earlier historical report links here without altering its original metrics or figures.

## Fixed-capacity and skip ablations

`gine_1x16.json`, `gine_2x32.json`, `gine_3x64.json` and `gine_2x32_skip.json` hold capacity fixed while choosing sparsity inside inner validation. Run them on the same prepared paired data and seed to obtain outer-OOF capacity comparisons. The 2×32 versus 2×32-skip presets isolate the residual ablation. The broader default grid instead selects capacity internally; its selected inner-validation values are not independent outer ablation results. Full-budget preset runs remain pending.

For a training-only shared-coordinate PCA illustration of held-out pre/post-harmonization features, run `python code/run_pipeline.py diagnostics --run MATCHED_RIDGE_RUN --prepared PREPARED_DATA --output reports/new_pipeline/diagnostics`. It fits one common PCA basis to the two training representations and projects held-out data; visual mixing is descriptive, not a proof of biological preservation.

A verified ROI JSON may optionally include 84 `networks` entries alongside `labels`, enabling network-node and network-edge sensitivity summaries. No network assignments are guessed.
