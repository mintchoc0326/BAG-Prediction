#!/usr/bin/env python3
"""Aggregate audited GINE arms, matched Ridge deltas and editable meeting slides.

May watch the study driver; never treats incomplete folds as final results.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "results/final/new_pipeline"
OUT = ROOT / "reports/new_pipeline/gine_study"
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/final")]


def paired_delta(left, right, seed=42, replicates=3000):
    """Participant-cluster bootstrap of matched prediction loss differences."""
    import numpy as np
    import pandas as pd

    keys = ["row_index", "fold"]
    a = left[left.is_cn].sort_values(keys)
    b = right[right.is_cn].sort_values(keys)
    cols = keys + ["age", "participant_group", "cohort"]
    if not a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True)):
        raise ValueError("Paired comparison requires identical observations and folds")
    delta = abs(a.predicted_age.to_numpy() - a.age.to_numpy()) - abs(b.predicted_age.to_numpy() - b.age.to_numpy())
    d = a[["cohort", "participant_group"]].copy()
    d["delta"] = delta
    d["n"] = 1
    g = d.groupby(["cohort", "participant_group"])[["delta", "n"]].sum().reset_index()
    rng = np.random.default_rng(seed)
    totals = np.zeros((replicates, 2))
    for _, cohort in g.groupby("cohort"):
        weights = rng.multinomial(len(cohort), np.full(len(cohort), 1 / len(cohort)), size=replicates)
        totals += weights @ cohort[["delta", "n"]].to_numpy()
    lower, upper = np.percentile(totals[:, 0] / totals[:, 1], [2.5, 97.5])
    return dict(delta_MAE=float(delta.mean()), CI_low=float(lower), CI_high=float(upper), n_scans=len(d), n_participants=d.participant_group.nunique())


def predictions(run):
    import pandas as pd

    rows = []
    for path in Path(run).glob("fold*/*/*/*/predictions_private.csv"):
        d = pd.read_csv(path)
        d["fold"] = int(path.parts[-5].replace("fold", ""))
        d["mode"] = path.parts[-4]
        d["modality"] = path.parts[-3]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def summarize(state):
    import numpy as np
    import pandas as pd

    completed = [j for j in state["jobs"] if j["status"] == "complete"]
    if not completed:
        return
    ridge_run = RUNTIME / "matched_ridge_20261106"
    reference_manifest = json.loads((ridge_run / "manifest.json").read_text())
    reference_plan = json.loads((ridge_run / "fold_plan_private.json").read_text())
    base = predictions(ridge_run)
    paired = []
    gaps = []
    full_predictions = {}
    for job in completed:
        run = Path(job["run"])
        manifest = json.loads((run / "manifest.json").read_text())
        audit = json.loads((run / "audit.json").read_text())
        if manifest["status"] != "complete" or audit["status"] != "passed":
            raise ValueError("Only independently audited complete runs may be summarized")
        if manifest["prepared_manifest_sha256"] != reference_manifest["prepared_manifest_sha256"]:
            raise ValueError("Prepared data mismatch")
        plan = json.loads((run / "fold_plan_private.json").read_text())
        if [f["test"] for f in plan["folds"]] != [f["test"] for f in reference_plan["folds"]]:
            raise ValueError("Outer fold mismatch")
        if job["fraction"] == 1:
            if plan != reference_plan:
                raise ValueError("Full-training inner/stopping folds differ from Ridge")
            current = predictions(run)
            full_predictions[job["arm"]] = current
            for (mode, modality), part in current.groupby(["mode", "modality"]):
                ref = base[base["mode"].eq(mode) & base.modality.eq(modality)]
                paired.append(dict(arm=job["arm"], reference="ridge", mode=mode, modality=modality, **paired_delta(part, ref)))
        for task_path in run.glob("fold*/*/*/gine/task.json"):
            task = json.loads(task_path.read_text())
            history = pd.read_csv(task_path.with_name("learning_history.csv"))
            selected = history[(history.stage == "early_stopping") & history.candidate.eq(task["selected_candidate"])]
            for inner, part in selected.groupby("inner_fold"):
                best = int(part.selected_epoch.iloc[0])
                best_row = part[part.epoch.eq(best)].iloc[0]
                gaps.append(dict(arm=job["arm"], fraction=job["fraction"], fold=int(task_path.parts[-5].replace("fold", "")), mode=task_path.parts[-4], modality=task_path.parts[-3], inner_fold=int(inner), parameters=task["parameters"], selected_epoch=best, last_epoch=int(part.epoch.max()), reached_epoch_cap=int(part.epoch.max()) >= manifest["settings"]["training"]["max_epochs"], train_MAE_at_selected=best_row.train_mae, stopping_MAE_at_selected=best_row.validation_mae, stopping_minus_train_MAE=best_row.validation_mae-best_row.train_mae, outer_refit_epochs=task["epochs"]))
    for arm, reference in [("gine_1x16", "gine_3x64"), ("gine_2x32", "gine_3x64"), ("gine_2x32_skip", "gine_2x32")]:
        if arm in full_predictions and reference in full_predictions:
            for (mode, modality), part in full_predictions[arm].groupby(["mode", "modality"]):
                ref = full_predictions[reference]
                ref = ref[ref["mode"].eq(mode) & ref.modality.eq(modality)]
                paired.append(dict(arm=arm, reference=reference, mode=mode, modality=modality, **paired_delta(part, ref)))
    paired = pd.DataFrame(paired)
    gaps = pd.DataFrame(gaps)
    paired.to_csv(OUT / "paired_MAE_differences.csv", index=False)
    gaps.to_csv(OUT / "early_stopping_details.csv", index=False)
    if len(gaps):
        gap_summary = gaps.groupby(["arm", "fraction", "mode", "modality"]).agg(parameters=("parameters", "first"), selected_epoch_median=("selected_epoch", "median"), selected_epoch_min=("selected_epoch", "min"), selected_epoch_max=("selected_epoch", "max"), mean_training_MAE=("train_MAE_at_selected", "mean"), mean_stopping_MAE=("stopping_MAE_at_selected", "mean"), mean_stopping_train_gap=("stopping_minus_train_MAE", "mean"), epoch_cap_rate=("reached_epoch_cap", "mean")).reset_index()
        gap_summary.to_csv(OUT / "early_stopping_summary.csv", index=False)
    else:
        gap_summary = pd.DataFrame()
    performance = pd.read_csv(OUT / "ridge_gine_comparison.csv") if (OUT / "ridge_gine_comparison.csv").exists() else pd.DataFrame()
    learning_rows = []
    for job in completed:
        if job["arm"] not in {"gine_2x32", "ridge"}:
            continue
        d = pd.read_csv(Path(job["run"]) / "task_summary.csv")
        d["fraction"] = job["fraction"]
        d["arm"] = job["arm"]
        learning_rows.append(d)
    d = pd.read_csv(ridge_run / "task_summary.csv")
    d["fraction"] = 1.0
    d["arm"] = "ridge"
    learning_rows.append(d)
    learning = pd.concat(learning_rows, ignore_index=True)
    learning.to_csv(OUT / "learning_curve_fold_metrics.csv", index=False)
    plot_learning(learning)
    write_slides(performance, gap_summary, paired, learning)
    sections = ["<h1>GINE study: audited comparison details</h1>", "<p>Only completed audited arms are included. A negative paired MAE difference favours the named arm. Participant-cluster bootstrap intervals are stratified by cohort and conditional on fitted OOF models; they do not capture refitting or split uncertainty and are exploratory, without multiplicity adjustment.</p>", "<h2>Paired MAE differences</h2>", paired.to_html(index=False, float_format=lambda x: f"{x:.3f}"), "<h2>Early stopping and training–validation gap</h2>", "<p>Values are from the selected sparsity candidate in each outer fold. Stopping validation is inside inner training; it is not outer-test performance. Gap alone does not establish why one model performs better.</p>", gap_summary.to_html(index=False, float_format=lambda x: f"{x:.3f}"), '<p><a href="GINE_comparison_editable.pptx">Editable comparison slides</a> · <a href="index.html">Live study status and full reports</a></p>']
    for mode in ["none", "age_mean"]:
        for modality in ["DWI", "fMRI", "fusion"]:
            image = f"learning_curve_{mode}_{modality}.png"
            if (OUT / image).exists():
                sections.append(f'<p>{mode} / {modality}</p><img src="{image}" width="600">')
    temporary = OUT / "comparison_details.html.tmp"
    temporary.write_text('<!doctype html><meta charset="utf-8"><style>body{font:15px Arial;margin:35px}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd}</style>' + "\n".join(sections))
    temporary.replace(OUT / "comparison_details.html")


def plot_learning(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for (mode, modality), part in data.groupby(["mode", "modality"]):
        if part.fraction.nunique() < 2:
            continue
        with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            fig, ax = plt.subplots(figsize=(5.5, 4))
            for arm, group in part.groupby("arm"):
                agg = group.groupby("fraction").agg(n=("n_training_CN", "mean"), MAE=("MAE", "mean"))
                ax.plot(agg.n, agg.MAE, marker="o", color="#4878A8" if arm == "ridge" else "#BC5B58", label="Ridge" if arm == "ridge" else "GINE 2×32")
            ax.set(xlabel="CN training scans (mean across folds)", ylabel="Mean outer-fold MAE (years)")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False)
            fig.tight_layout()
            for ext in ["png", "pdf", "svg"]:
                fig.savefig(OUT / f"learning_curve_{mode}_{modality}.{ext}", dpi=200, bbox_inches="tight")
            plt.close(fig)


def write_slides(performance, gap, paired, learning):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
    from pptx.dml.color import RGBColor

    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)

    def slide(title, note):
        s = deck.slides.add_slide(deck.slide_layouts[6])
        for x, y, w, h, text, size in [(0.5, .25, 12.3, .8, title, 27), (.5, 6.65, 12.3, .65, note, 12)]:
            tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
            tf.word_wrap = True
            tf.text = text
            for p in tf.paragraphs:
                p.font.name, p.font.size = "Arial", Pt(size)
        return s

    s = slide("GINE versus Ridge: full-budget nested evaluation", "Development reanalysis; clinical dates, acquisition protocols and ROI ordering remain unverified. Completed audited arms only.")
    text = "Matched paired dataset: 318 scans / 252 participants; CN evaluation: 237 scans / 189 participants.\n\nFive participant-grouped outer folds, three inner folds; early stopping separated inside inner training.\n\nGINE: 1×16, 2×32, 3×64 and 2×32 residual. Two sparsity levels selected inside inner CV.\n\nTraining-only mean-only ComBat versus no harmonization; imaging-only predictors.\n\nTraining-size study: prespecified 2×32 versus Ridge. Unsupported fractions are disclosed."
    box = s.shapes.add_textbox(Inches(.6), Inches(1.3), Inches(12), Inches(5)).text_frame
    box.word_wrap = True
    box.text = text
    for p in box.paragraphs:
        p.font.name, p.font.size = "Arial", Pt(21)
    colors = ["4878A8", "C6A83E", "BC5B58", "59926C", "8D74AA"]
    if len(performance):
        for mode in ["none", "age_mean"]:
            part = performance[performance["mode"].eq(mode)]
            s = slide("OOF MAE — " + mode, "Native editable chart. Point estimates; see HTML/CSV for participant-bootstrap intervals and paired differences. No architecture is selected using outer-test scores.")
            data = CategoryChartData()
            categories = ["DWI", "fMRI", "fusion"]
            data.categories = categories
            for arm, group in part.groupby("arm", sort=False):
                g = group.set_index("modality").reindex(categories)
                data.add_series(arm, [float(v) for v in g.MAE])
            chart = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(.7), Inches(1.3), Inches(11.9), Inches(5.1), data).chart
            chart.has_legend = True
            chart.legend.position = XL_LEGEND_POSITION.BOTTOM
            for i, series in enumerate(chart.series):
                series.format.fill.solid()
                series.format.fill.fore_color.rgb = RGBColor.from_string(colors[i % len(colors)])
    if len(gap):
        view = gap[gap.fraction.eq(1) & gap["mode"].eq("age_mean")][["arm", "modality", "parameters", "selected_epoch_median", "mean_training_MAE", "mean_stopping_MAE", "mean_stopping_train_gap"]]
        s = slide("Training and validation diagnostics — age_mean", "Mean across selected-candidate inner stopping splits. These validation errors are not outer-test estimates; each run report contains epoch-by-epoch curves.")
        table = s.shapes.add_table(len(view)+1, len(view.columns), Inches(.5), Inches(1.3), Inches(12.3), Inches(4.9)).table
        for c, col in enumerate(view.columns):
            table.cell(0, c).text = col.replace("_", " ")
        for r, values in enumerate(view.itertuples(index=False, name=None), 1):
            for c, value in enumerate(values):
                table.cell(r, c).text = f"{value:.2f}" if isinstance(value, float) else str(value)
        for row in table.rows:
            for cell in row.cells:
                for p in cell.text_frame.paragraphs:
                    p.font.name, p.font.size = "Arial", Pt(11)
    destination = OUT / "GINE_comparison_editable.pptx"
    temporary = OUT / "GINE_comparison_editable.tmp.pptx"
    deck.save(temporary)
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20261106_full")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    state_path = RUNTIME / f"gine_study_{args.tag}.json"
    last = None
    while True:
        try:
            state = json.loads(state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if not args.watch:
                raise
            time.sleep(10)
            continue
        signature = [(j["name"], j["status"]) for j in state["jobs"]]
        if signature != last:
            summarize(state)
            subprocess.run([sys.executable, str(ROOT / "code/reporting/build_pipeline_portal.py")], cwd=ROOT, check=True)
            last = signature
        if not args.watch or all(j["status"] in {"complete", "failed"} for j in state["jobs"]):
            break
        pid = int((RUNTIME / f"gine_study_driver_{args.tag}.pid").read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise RuntimeError("Study driver stopped; see logs and resume before publishing completion")
        time.sleep(30)


if __name__ == "__main__":
    main()
