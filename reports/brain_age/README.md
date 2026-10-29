# English performance report and presentation tables

Open [performance_report.html](performance_report.html) after downloading this folder, or view [performance_tables.md](performance_tables.md) on GitHub. The four-panel figure is [paired_model_performance.png](paired_model_performance.png), also available as PDF and SVG. [performance_tables.pdf](performance_tables.pdf) contains one table per cohort and a pooled table.

## Models and evaluation population

The main figure compares six models on identical raw CN OOF observations: Ridge DWI-only, Ridge fMRI-only, Ridge weighted early fusion, GNN DWI-only, GNN fMRI-only, and GNN fusion. This is the existing matched benchmark, not the larger independently trained unimodal analyses. ADNI has 131 scans / 74 participants; ADRC 243 / 243; HABS 13 / 13; pooled 387 / 330. Models share participant-grouped outer folds. Ridge fusion explicitly means weighted early fusion; unweighted early fusion and late mean are included in `paired_metrics_all_models.csv`.

`paired_metrics_six_models.csv` contains the main results. `full_unimodal_metrics.csv` contains DWI/fMRI full-population results for ADNI, ADRC, and HABS. These use different modality populations and are not a matched fusion comparison. ADDecode appears in DWI harmonization inputs but has no saved CN OOF predictions in any of these model files; its metrics cannot be computed. `model_availability.csv` explicitly records all four datasets and missing model results. No estimates are imputed.

## Confidence intervals and metrics

Point estimates are computed across scans, not averaged fold metrics. MAE and RMSE are in years; R² = 1 − SSE/SST; Pearson r measures linear association and is not the square root of R². Negative R² values are retained.

Intervals use 3,000 participant-cluster bootstrap replicates, keeping all visits together; the same resampling is used across models within a cohort. Pooled resampling is stratified by dataset. Percentile 95% CIs condition on the existing fitted OOF models. They do not include model refitting or selection uncertainty, and do not establish significant model differences. Undefined R²/r replicates are excluded and valid counts are recorded. HABS uncertainty is based on only 13 people.

Upstream harmonization used all eligible scans while preserving age, so these results retain the previously documented leakage risk. These are pooled-training OOF estimates, not external-cohort or leave-one-dataset-out validation.

## All slide tables in one Python file

`code/reporting/build_results_report.py` embeds all 46 native table occurrences across the two reviewed PPTX decks (23 per deck), preserving cell text, slide number, deck name, and context. Run `--tables-only` without private data; use `--slides path/to/deck.pptx` to extract tables from a newer deck. It rebuilds:

- `all_slide_tables.html`: exact Unicode cell text and provenance.
- `all_slide_tables.pdf`: one table per page; mathematical font support depends on the local fonts.
- `slide_tables/*.csv`: editable individual tables.
- `slide_table_manifest.json`: full embedded table snapshot and slide references.

The existing slides are preserved. Their tables are reproduced from displayed snapshots, not recomputed from every underlying analysis; historical IG values remain explicitly attributed to reconstructed models. Main performance metrics are newly calculated from saved prediction CSVs. `provenance.json` records source hashes, bootstrap settings, and limitations. No participant-level prediction records are published.

## Run

From the repository root, with numpy, pandas, and matplotlib installed:

```bash
python code/reporting/build_results_report.py --bootstrap 3000 --seed 20261029
```

The full run requires the private OOF CSVs at the paths documented in `provenance.json`. It does not retrain or modify model outputs. Use `--output /some/directory` to choose a different destination.
