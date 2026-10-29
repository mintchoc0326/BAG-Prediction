# Slide sources

`build_slides.py`, `add_pipelines.py`, `equations_in_boxes.py` and `conference_pipelines.py` form the builder dependency chain; they are retained together as build support, rather than independent scientific analysis entry points.

To rebuild and apply the current code/output audit:

```bash
python code/final/slides/conference_pipelines.py
python code/tools/reconcile_slides.py
```

Install Python dependencies from requirements.txt if needed. Final PPTX/PDF exports are in results/presentation. After rebuilding PPTX, export PDF with LibreOffice. The separate-equation layout superseded by boxed equations is archived in code/explore/slides/.
