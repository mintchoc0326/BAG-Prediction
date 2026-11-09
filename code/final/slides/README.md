# Slide sources

## Comprehensive current presentation

Run `python code/final/slides/update_comprehensive_pipeline.py` to update the full
`Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx` presentation.
It reads aggregate current results, replaces historical methods with the
participant-nested implementation, and retains original numerical figures in a
clearly marked historical appendix. The original 54-slide deck is backed up once
under `results/presentation/archive_before_pipeline_update_20261109/`.

Then apply the shared style and export the PDF:

```bash
python code/tools/restyle_slides.py results/presentation/Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx
libreoffice --headless --convert-to pdf --outdir results/presentation results/presentation/Connectome_Brain_Age_Results_English_Editable_with_Pipelines.pptx
```

The updated PPTX/PDF copies are published in `reports/brain_age/slides/`.
The builder requires the local aggregate CSVs and figures under
`reports/new_pipeline/`, plus the archived original presentation. These aggregate
inputs are retained locally; they were not part of the code-and-slides GitHub
publication. The builder records source hashes and shape-bound checks in
`results/presentation/comprehensive_pipeline_update_validation.json`.

The historical builder chain below reproduces the earlier evaluation and should
not be run over the current presentation as an update step.

`build_slides.py`, `add_pipelines.py`, `equations_in_boxes.py` and `conference_pipelines.py` form the builder dependency chain; they are retained together as build support, rather than independent scientific analysis entry points.

To rebuild and apply the current code/output audit:

```bash
python code/final/slides/conference_pipelines.py
python code/tools/reconcile_slides.py
```

Install Python dependencies from requirements.txt if needed. Final PPTX/PDF exports are in results/presentation. After rebuilding PPTX, export PDF with LibreOffice. The separate-equation layout superseded by boxed equations is archived in code/explore/slides/.

## Presentation formatting

After building or reconciling decks, run `python code/tools/restyle_slides.py` from the repository root. This applies the shared formal style to all report/presentation PPTX files while verifying unchanged text, notes, chart values, embedded media and relationships. Originals are backed up under `.local/slide_format_backups/`; verification is recorded in `reports/slide_format_validation.json`. Re-export the matching PDFs after formatting.
