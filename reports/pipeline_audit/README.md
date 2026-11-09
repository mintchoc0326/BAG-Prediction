# DWI pipeline audit

Source-code review prepared 3 November 2026. This is an audit and proposed design, not an implemented leakage-free pipeline or a new training result. Participant IDs, clinical variables, and effects on prediction metrics still require validation. The main findings concern global ComBat before CV, threshold selection using reported OOF scores, and the prediction/calibration provenance of cBAG.

## Files

- `DWI_pipeline_audit_editable.pptx`: seven slides with editable native shapes and text.
- `DWI_pipeline_audit_editable.pdf`: meeting preview.
- `audit_findings.tsv`: findings, status, source-code locations, and next actions; no participant records.
- `build_slides.py`: reproducible presentation builder.

## Rebuild

From the repository root:

```bash
python -m pip install -r reports/pipeline_audit/requirements.txt
python reports/pipeline_audit/build_slides.py
libreoffice --headless --convert-to pdf --outdir reports/pipeline_audit reports/pipeline_audit/DWI_pipeline_audit_editable.pptx
```

PowerPoint generation requires python-pptx; PDF export additionally requires LibreOffice. The builder validates slide count and shape boundaries. Code-line references describe the audited source snapshot and may shift as implementation changes.

## Implementation follow-up: 6 November 2026

The original audit slides above remain a source-audit snapshot. A restricted known-site DWI Ridge implementation now lives in `code/final/foldwise_dwi.py` and `code/final/run_foldwise_DWI_Ridge.py`, with synthetic boundary checks in `code/final/test_foldwise_dwi.py`. See `code/final/README.md` for reproduction commands and scientific limits. This does not yet migrate GINE/fMRI/fusion or verify clinical labels and ADRC protocols.

`code/final/slides/build_foldwise_validation_slides.py` builds a separate native editable deck from the completed run; local output is `reports/pipeline_audit/foldwise_validation_20261106/`. Age-trained mean-only ComBat uses training ages only to estimate additive offsets; held-out transformations never accept chronological age. The slides distinguish implementation validation from unresolved scientific/data validation.

The matching fMRI Ridge implementation and independent audit were also completed on 6 November 2026. Its editable deck and PDF are under `fmri_foldwise_validation_20261106/`; rebuild with the shared slide builder using `--run results/final/fMRI/foldwise_Ridge_validated_20261106`. DWI and fMRI currently have modality-specific eligible populations/folds; a paired comparison and fusion evaluation remain pending. GINE has not yet been migrated.

## Current unified pipeline

The latest implementation and editable methods deck are linked from `../new_pipeline/index.html`. See `code/brainage_pipeline/README.md` for the common fold plan, separate early stopping, training-only cohort calibration and explicit pairing exclusions. The original audit files in this directory remain the historical source-audit record; they are not silently replaced by a claim that all clinical/protocol questions are resolved.
