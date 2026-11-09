#!/usr/bin/env python3
"""Assemble the current public aggregate report hub without private predictions."""
import html
import json
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/new_pipeline"
RUNTIME = ROOT / "results/final/new_pipeline"


def copy_aggregate(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if (
            file.is_file()
            and file.suffix in {".csv", ".html", ".json", ".png", ".svg", ".pdf"}
            and "private" not in file.name
        ):
            shutil.copy2(file, destination / file.name)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    copy_aggregate(RUNTIME / "clinical_inventory_20261106", OUT / "clinical_inventory")
    copy_aggregate(
        ROOT / "results/final/foldwise_analysis_20261106",
        OUT / "harmonization_evaluation",
    )
    prepared = RUNTIME / "prepared_20261106"
    counts = pd.read_csv(prepared / "sample_counts.csv")
    pairing = pd.read_csv(prepared / "pairing_counts.csv").fillna("retained")
    counts.to_csv(OUT / "sample_counts.csv", index=False)
    pairing.to_csv(OUT / "pairing_counts.csv", index=False)
    inventory = pd.read_csv(
        RUNTIME / "clinical_inventory_20261106/candidate_cross_cohort_measures.csv"
    )
    inventory.to_csv(OUT / "clinical_target_candidates.csv", index=False)
    stages = []
    for label, directory in [
        ("Matched Ridge", "matched_ridge_20261106"),
        ("GINE code validation", "gine_validation_v2_20261106"),
    ]:
        path = RUNTIME / directory
        manifest = (
            json.loads((path / "manifest.json").read_text())
            if (path / "manifest.json").exists()
            else {"status": "not_run"}
        )
        check = (
            json.loads((path / "audit.json").read_text())
            if (path / "audit.json").exists()
            else {"status": "not_audited"}
        )
        stages.append(
            dict(
                stage=label,
                status=manifest["status"],
                audit=check["status"],
                validation_only=manifest.get("settings", {}).get(
                    "validation_only", False
                ),
            )
        )
    stages += [
        dict(
            stage="Full-budget nested GINE capacity / residual / topology comparison",
            status="code_ready_not_run",
            audit="short_budget_path_tested",
            validation_only=False,
        ),
        dict(
            stage="Increasing-training-size study",
            status="code_ready_not_run",
            audit="shared_outer_test_enforced",
            validation_only=False,
        ),
        dict(
            stage="Clinical association analysis",
            status="awaiting_verified_source_codes_units_dates",
            audit="synthetic_interface_tested",
            validation_only=False,
        ),
        dict(
            stage="Anatomical claims",
            status="awaiting_verified_ROI_order_and_full_models",
            audit="original_checkpoint_path_tested",
            validation_only=False,
        ),
    ]
    study_status = OUT / "gine_study/study_status.csv"
    if study_status.exists():
        study = pd.read_csv(study_status)
        for stage_name, part in [
            ("Full-budget nested GINE capacity / residual / topology comparison", study[study.fraction.eq(1)]),
            ("Increasing-training-size study", study[study.fraction.lt(1)]),
        ]:
            for stage in stages:
                if stage["stage"] == stage_name:
                    stage["status"] = (
                        "complete" if part.status.eq("complete").all()
                        else "in_progress_see_live_study_status"
                    )
                    stage["audit"] = "each_completed_run_independently_audited"
    pd.DataFrame(stages).to_csv(OUT / "stage_status.csv", index=False)
    performance = pd.read_csv(OUT / "matched_ridge/performance.csv")
    pooled = performance[performance.cohort.eq("Pooled")]
    audit_map = pd.DataFrame(
        [
            [
                "Global ComBat before CV",
                "Training-only shared fit/apply module",
                "Synthetic age/companion isolation; train/apply equivalence",
            ],
            [
                "Repeated scans across evaluations",
                "One canonical participant plan across six models",
                "Outer, inner and stopping split audit",
            ],
            [
                "Threshold/model choices on outer OOF",
                "Inner selection; separate GINE stopping then refit",
                "Fit-index and prediction provenance audit",
            ],
            [
                "Training-in-sample cBAG / residual cohort age bias",
                "Inner OOF calibration; global and cohort-aware outputs",
                "Reconstruct coefficients and held-out cBAG",
            ],
            [
                "DWI/fMRI metadata conflicts",
                "Conservative unique-key pairing; explicit age/sex/diagnosis exclusions",
                "No averaging or first-row duplicate resolution",
            ],
            [
                "Reconstructed interpretation models",
                "Original outer checkpoints and fold-specific embeddings",
                "Checkpoint path exercised; ROI names gated",
            ],
        ],
        columns=["Audit finding", "Implementation", "Verification"],
    )
    audit_map.to_csv(OUT / "audit_to_implementation.csv", index=False)
    sections = [
        "<h1>Current brain-age pipeline</h1>",
        "<p>This report supersedes the historical full-sample-ComBat prediction workflow. Raw inputs, participant-grouped nested CV, training-only transforms and bias calibration are now implemented. Clinical/protocol/ROI source validity remains unresolved.</p>",
        "<h2>Current status</h2>",
        '<p><a href="gine_study/index.html">Full-budget GINE: live training status, completed architecture comparisons and learning curves</a></p>' if study_status.exists() else "",
        '<p><a href="gine_study/comparison_details.html">Audited GINE/Ridge paired comparisons, stopping diagnostics and editable comparison slides</a></p>' if (OUT / "gine_study/comparison_details.html").exists() else "",
        pd.DataFrame(stages).to_html(index=False),
        "<h2>Audit to implementation</h2>",
        audit_map.to_html(index=False),
        "<h2>Analysis populations</h2>",
        counts.to_html(index=False),
        "<h2>Paired-scan exclusions</h2>",
        pairing.to_html(index=False),
        "<p>250 of 575 common scan keys have age discrepancies above 0.01 years. These records are excluded, not averaged. Another 7 have insufficient paired CN batch support, leaving 318 scans / 252 participants (237 CN scans). Dates remain unverified; no claim of fully validated contemporaneous pairing is made.</p>",
        "<h2>Completed matched Ridge results</h2>",
        pooled.to_html(index=False, float_format=lambda x: f"{x:.3f}"),
        "<p>Three imaging inputs share observations and outer folds. These differ from the historical benchmark. Cohort calibration was specified during development after prior OOF diagnostics; external confirmation remains necessary. HABS has few retained paired observations.</p>",
        '<p><a href="matched_ridge/index.html">Full matched Ridge report and figures</a> · <a href="gine_validation/index.html">GINE short-budget technical validation — not scientific performance</a></p>',
        '<h2>Editable methods slides</h2><p><a href="slides/Unified_pipeline_editable.pptx">PowerPoint: native editable text, tables and shapes</a> · <a href="slides/Unified_pipeline_editable.pdf">PDF preview</a></p>',
        '<p><a href="diagnostics/harmonization_PCA.svg">Held-out harmonization PCA using shared training-only axes</a> · <a href="validation_summary.json">Test and runtime-audit summary</a></p>',
        '<h2>Harmonization evidence and clinical inventory</h2><p><a href="harmonization_evaluation/analysis_report.html">Earlier modality-wide harmonization diagnostics</a> · <a href="clinical_inventory/clinical_inventory.html">Cohort-by-cohort source inventory and preliminary cBAG plan</a> · <a href="clinical_target_candidates.csv">Clinical candidate field availability</a></p>',
        "<p>T2D definitions and HbA1c units cannot be inferred from column names. The clinical analysis command remains gated until source codes and visit alignment are verified. Sample counts in the source inventory are not automatically analysis-ready participant counts.</p>",
        "<h2>Code and reproduction</h2><p>Entry point: code/run_pipeline.py. See code/brainage_pipeline/README.md for commands, default/validation configurations, training-size studies and downstream analysis. No participant-level prediction table is included in this report directory.</p>",
    ]
    css = "<style>body{font:15px Arial,sans-serif;margin:32px;color:#222;max-width:1500px}table{border-collapse:collapse;margin-bottom:24px}td,th{border:1px solid #ddd;padding:8px;text-align:left}th{background:#f3f3f3}h2{margin-top:32px}</style>"
    (OUT / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">' + css + "".join(sections)
    )
    print(OUT / "index.html")


if __name__ == "__main__":
    build()
