# Current paired cohort comparison

[Four-panel PDF](paired_model_performance.pdf) · [PNG](paired_model_performance.png) · [SVG](paired_model_performance.svg)

[No-ComBat PDF](paired_model_performance_no_combat.pdf) · [PNG](paired_model_performance_no_combat.png) · [SVG](paired_model_performance_no_combat.svg)

Panels show MAE, RMSE, R-squared and Pearson r. Six models share the same outer-held-out CN observations: Ridge and fixed GINE 2x32, each with DWI, fMRI and early fusion. GINE 2x32 is a prespecified reference arm rather than a winner selected on outer scores. The default figure uses training-age mean-only ComBat; no-ComBat is a separate sensitivity figure.

| Cohort | CN scans | Participants |
| --- | ---: | ---: |
| ADNI | 125 | 77 |
| ADRC | 106 | 106 |
| HABS | 6 | 6 |
| Total | 237 | 189 |

The archived paired registry contains 318 scans / 252 participants including non-CN scans. The figure uses CN only. HABS estimates are especially uncertain. This is the current restricted development evaluation, not the historical 387-CN-scan benchmark.

`paired_metrics_six_models.csv` contains 48 aggregate rows: two modes x six models x pooled/three cohort summaries. `provenance.json` records original-run fingerprints and export checks. No participant-level predictions are published.

Intervals are 95% percentile intervals from 3,000 cohort-stratified participant bootstrap draws, conditional on fixed OOF models. They exclude refitting and selection uncertainty. Metrics are raw predicted-age performance, not recalculated after BAG correction. Source labels, visit dates, acquisition protocols and ROI order remain unverified.

Rebuild from the committed aggregate table:

```bash
python code/reporting/plot_model_comparison.py
python code/reporting/plot_model_comparison.py --mode none --output reports/current/paired_model_performance_no_combat
```

To export aggregates from original completed runs, use `code/reporting/export_paired_metrics.py --help`. See the [root README](../../README.md) for the full workflow.
