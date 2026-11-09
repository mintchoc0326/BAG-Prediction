"""Editable native-shape slides for the unified pipeline and current run status."""
import json
from pathlib import Path
import pandas as pd


def slides(run, prepared, output):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.dml.color import RGBColor

    run, prepared, output = Path(run), Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text())
    counts = pd.read_csv(prepared / "sample_counts.csv")
    pairs = pd.read_csv(prepared / "pairing_counts.csv").fillna("")
    r = Presentation()
    r.slide_width = Inches(13.333)
    r.slide_height = Inches(7.5)

    def box(s, x, y, w, h, text, size=20, color="FFFFFF"):
        sh = s.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
        )
        sh.fill.solid()
        sh.fill.fore_color.rgb = RGBColor.from_string(color)
        sh.line.color.rgb = RGBColor.from_string("E5E7EB")
        sh.line.width = Pt(1)
        sh.text_frame.word_wrap = True
        sh.text_frame.margin_left = Pt(10)
        sh.text_frame.margin_right = Pt(10)
        sh.text_frame.margin_top = Pt(6)
        sh.text_frame.margin_bottom = Pt(6)
        for i, line in enumerate(text.split("\n")):
            p = sh.text_frame.paragraphs[0] if i == 0 else sh.text_frame.add_paragraph()
            p.text = line
            p.font.name = "Arial"
            p.font.size = Pt(size)
            p.font.color.rgb = RGBColor.from_string("111827")
            p.line_spacing = 1.0
            p.space_after = Pt(0)

    def new(title):
        s = r.slides.add_slide(r.slide_layouts[6])
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = RGBColor.from_string("FFFFFF")
        box(s, 0.4, 0.25, 12.5, 0.7, title, 26, "FFFFFF")
        for p in s.shapes[-1].text_frame.paragraphs:
            p.font.color.rgb = RGBColor.from_string("111827")
            p.font.bold = True
        box(
            s,
            0.4,
            7.05,
            12.5,
            0.25,
            "Unified pipeline | Run status: "
            + manifest["status"]
            + " | Clinical / protocol verification pending",
            10,
            "FFFFFF",
        )
        for p in s.shapes[-1].text_frame.paragraphs:
            p.font.color.rgb = RGBColor.from_string("374151")
        return s

    def table(s, headers, rows):
        sh = s.shapes.add_table(
            len(rows) + 1,
            len(headers),
            Inches(0.6),
            Inches(1.5),
            Inches(12.1),
            Inches(3.3),
        )
        for i, row in enumerate([headers] + rows):
            for j, v in enumerate(row):
                cell = sh.table.cell(i, j)
                cell.text = str(v)
                for p in cell.text_frame.paragraphs:
                    p.font.name = "Arial"
                    p.font.size = Pt(15)

    s = new("Unified pipeline: implemented boundaries and current evidence")
    box(
        s,
        0.6,
        1.4,
        12.1,
        1.0,
        "One raw-data registry, one participant fold plan, shared train/apply ComBat.\nRidge and GINE use identical paired observations for DWI, fMRI and fusion.",
        22,
        "F8FAFC",
    )
    box(
        s,
        0.6,
        2.9,
        12.1,
        3.3,
        "Implemented\n• Training-only edge QC, harmonization and normalization\n• Inner model / sparsity selection and independent GINE stopping\n• Inner-OOF reference calibration; global and cohort-aware cBAG\n• Original outer checkpoints, private embeddings, gradient contributions\n• Runtime audits, resumable tasks and source snapshots\n\nRun classification: "
        + (
            "validation/pilot only — short epoch budget is not a performance result"
            if manifest["settings"].get("validation_only")
            else "development evaluation — external confirmation still required"
        ),
        20,
    )
    s = new("Outer / inner / stopping boundaries")
    box(
        s,
        0.6,
        1.25,
        12.1,
        0.65,
        "Raw DWI log1p / signed fMRI Fisher-z → fixed participant outer split",
        20,
    )
    box(
        s,
        0.6,
        2.2,
        5.85,
        3.9,
        "OUTER TRAINING CN\nInner train / validation split\n↓ GINE only: split inner training into fit / stopping\nFit preprocessing on fit only; choose epoch on stopping\n↓ Refit preprocessing/model on all inner training\nEvaluate inner validation; choose configuration\n↓ Calibrate using inner OOF predictions\nRefit outer model for inner-selected epoch budget",
        18,
        "F8FAFC",
    )
    box(
        s,
        6.75,
        2.2,
        5.95,
        3.9,
        "OUTER HELD-OUT PARTICIPANTS\nFrozen preprocessing → raw age prediction\n↓ Compute BAG using actual age\n↓ Apply frozen training-only calibration\n\nNo test age in feature transformations.\nNo test fold in hyperparameters / sparsity / epochs.\nNo clinical, genetic or metabolic predictor features.",
        19,
    )
    s = new("Harmonization decision follows age-preservation evidence")
    box(
        s,
        0.6,
        1.4,
        12.1,
        1.2,
        "Primary development candidate: training-age mean-only ComBat\nHeld-out transformation: x_harmonized = x − training residual SD × batch offset",
        22,
        "F8FAFC",
    )
    box(
        s,
        0.6,
        3,
        12.1,
        2.8,
        "Training age helps separate age and acquisition offsets; held-out age is never required.\nNo-ComBat remains a prespecified control. Batch-only location/scale remains a diagnostic sensitivity option.\nPrior modality-wide MAE: DWI none 6.02 / mean-only 6.11; fMRI none 7.43 / mean-only 7.49.\nThose historical development comparisons are not the new matched six-model benchmark.",
        21,
    )
    s = new("Conservative population: unresolved pairs are not averaged")
    table(
        s,
        ["Population", "Scans", "Participants", "CN scans"],
        counts[["population", "scans", "participants", "CN_scans"]].values.tolist(),
    )
    reason = "; ".join(
        (str(int(row.scans)) + " " + (row.exclusion_reason or "retained"))
        for row in pairs.itertuples()
    )
    box(
        s,
        0.6,
        5.2,
        12.1,
        1.2,
        "Pairing: "
        + reason
        + ".\nSame-key / age / sex / diagnosis agreement is checked; acquisition/clinical dates still require verification.",
        18,
        "F8FAFC",
    )
    s = new("GINE comparisons and scientific outputs")
    box(
        s,
        0.6,
        1.4,
        12.1,
        4.9,
        "Capacity: 1 × 16, 2 × 32, current 3 × 64; 2 × 32 residual ablation.\nGraph sparsity is selected inside inner validation. Signed fMRI edges are retained; fusion uses modality-union topology and concatenated node profiles.\n\nLearning curves: evaluation-mode training / stopping loss and MAE; selected epochs recorded. Separate runs vary training participant fractions with fixed outer test folds.\n\nBias correction: inspect global and cohort-aware OOF cBAG versus age within every cohort.\n\nInterpretation: original outer checkpoints, fold-specific embeddings and gradient-times-input. Anatomical naming requires verified ROI order.",
        20,
    )
    s = new("Results: completion status is part of the result")
    summary = (
        pd.read_csv(run / "task_summary.csv")
        if (run / "task_summary.csv").exists()
        else pd.DataFrame()
    )
    if len(summary):
        grouped = (
            summary.groupby(["mode", "modality", "learner"])
            .agg(completed_folds=("fold", "nunique"), mean_fold_MAE=("MAE", "mean"))
            .reset_index()
        )
        table(
            s,
            ["Mode", "Input", "Model", "Completed folds", "Mean fold MAE"],
            [
                [a, b, c, int(d), f"{e:.3f}"]
                for a, b, c, d, e in grouped.values.tolist()
            ][:12],
        )
    box(
        s,
        0.6,
        5.25,
        12.1,
        1.1,
        "Status: "
        + manifest["status"]
        + ". "
        + (
            "Short-budget validation: metrics must not be presented as trained-model performance."
            if manifest["settings"].get("validation_only")
            else "Mean-fold values are descriptive; use the report for pooled OOF metrics and cohort breakdowns."
        ),
        19,
        "F8FAFC",
    )
    s = new("Remaining data validation and clinical analysis gates")
    box(
        s,
        0.6,
        1.4,
        12.1,
        4.9,
        "Verify HABS diagnosis provenance, ADRC visit links/protocols, age discrepancies and ROI order.\n\nClinical inventory: CN/MCI/AD, APOE / APOE4, sex, T2D, HbA1c and continuous age. Source codes, units and clinical timing must be verified before testing.\n\nClinical analyses: group distributions and complete-case multivariable associations with cohort, age and sex; participant-cluster uncertainty; explicit exploratory multiplicity correction.\n\nEmbeddings: train-fit PCA per outer model, held-out overlays; never pool unaligned fold coordinates.\n\nNo claim of fully removed scanner effects, preserved biology, clinical biomarkers or unseen-site generalization.",
        20,
    )
    target = output / "Unified_pipeline_editable.pptx"
    r.save(target)
    for s in Presentation(target).slides:
        for sh in s.shapes:
            assert (
                sh.left >= 0
                and sh.top >= 0
                and sh.left + sh.width <= r.slide_width
                and sh.top + sh.height <= r.slide_height
            )
    return target
