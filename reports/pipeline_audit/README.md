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
