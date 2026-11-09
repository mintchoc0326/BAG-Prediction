#!/usr/bin/env python3
"""Run the prespecified full-budget GINE study and publish audited aggregates.

Two GPU workers, one training process per GPU. No test-driven architecture
selection: every fixed-capacity arm is reported. Resume uses engine provenance
checks. Individual predictions and logs remain under ignored results/.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "results/final/new_pipeline"
PUBLIC = ROOT / "reports/new_pipeline/gine_study"
ARMS = ["gine_1x16", "gine_2x32", "gine_3x64", "gine_2x32_skip"]
LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def publish(jobs):
    """Atomic status publication; aggregate-only public artifacts."""
    import pandas as pd

    PUBLIC.mkdir(parents=True, exist_ok=True)
    records = []
    tables = []
    for job in jobs:
        row = {k: v for k, v in job.items() if k != "run"}
        run = Path(job["run"])
        summary = run / "task_summary.csv"
        row["completed_tasks"] = len(pd.read_csv(summary)) if summary.exists() else 0
        row["expected_tasks"] = 30
        records.append(row)
        perf = PUBLIC / job["name"] / "performance.csv"
        if job["status"] == "complete" and perf.exists():
            data = pd.read_csv(perf)
            data.insert(0, "arm", job["arm"])
            data.insert(1, "fraction", job["fraction"])
            tables.append(data)
    frame = pd.DataFrame(records)
    frame.to_csv(PUBLIC / "study_status.csv", index=False)
    sections = [
        "<h1>Full-budget GINE study</h1>",
        f"<p>Updated {html.escape(now())}. Status is based on completed fold tasks, not elapsed time.</p>",
        "<p>Shared paired participants and outer folds with Ridge. Five outer folds, three inner folds; separate participant early stopping inside inner training. Up to 250 epochs, patience 25. Two sparsity candidates selected by inner validation. No-ComBat and training-age mean-only ComBat are separate arms. All architectures are reported; outer scores do not select a final architecture.</p>",
        frame.to_html(index=False),
        "<p>Training-size comparison uses the prespecified 2-layer, 32-channel GINE, not the best observed architecture. Fractions must support training and stopping batches. Failed fractions are disclosed and not silently replaced.</p>",
        "<p>Only fully trained, independently audited runs enter the tables below. Running experiments are not scientific results. Confidence intervals condition on fitted OOF predictions; external validation remains necessary. Acquisition/clinical dates and ROI ordering remain unverified.</p>",
    ]
    if tables:
        combined = pd.concat(tables, ignore_index=True)
        combined.to_csv(PUBLIC / "performance.csv", index=False)
        full = combined[combined.fraction.eq(1) & combined.cohort.eq("Pooled")]
        ridge = pd.read_csv(ROOT / "reports/new_pipeline/matched_ridge/performance.csv")
        ridge = ridge[ridge.cohort.eq("Pooled")].copy()
        ridge["arm"] = "ridge"
        ridge["fraction"] = 1.0
        comparison = pd.concat([ridge, full], ignore_index=True)
        comparison.to_csv(PUBLIC / "ridge_gine_comparison.csv", index=False)
        sections += ["<h2>Completed full-training comparisons</h2>", comparison.to_html(index=False, float_format=lambda x: f"{x:.3f}")]
        plot_comparison(comparison)
        sections.append('<p><a href="architecture_comparison.png">Architecture comparison figure</a> · <a href="architecture_comparison.svg">Editable SVG</a></p>')
    sections.append("<h2>Reports and editable slides</h2><ul>")
    for job in jobs:
        if job["status"] == "complete":
            name = html.escape(job["name"])
            sections.append(f'<li><a href="{name}/index.html">{name}: metrics and training/validation curves</a> · <a href="{name}/slides/Unified_pipeline_editable.pptx">Editable PowerPoint</a></li>')
    sections.append('</ul><p><a href="../index.html">Main pipeline report</a></p>')
    temporary = PUBLIC / "index.html.tmp"
    temporary.write_text('<!doctype html><meta charset="utf-8"><style>body{font:15px Arial;margin:35px;line-height:1.5}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd}</style>' + "\n".join(sections))
    temporary.replace(PUBLIC / "index.html")


def plot_comparison(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["#4878A8", "#C6A83E", "#BC5B58", "#59926C", "#8D74AA"]
    arms = ["ridge"] + [a for a in ARMS if a in set(data.arm)]
    with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 10}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, mode in zip(axes, ["none", "age_mean"]):
            for i, arm in enumerate(arms):
                group = data[data.arm.eq(arm) & data["mode"].eq(mode)].set_index("modality")
                group = group.reindex(["DWI", "fMRI", "fusion"])
                width = .8 / len(arms)
                x = np.arange(3) - .4 + width / 2 + i * width
                ax.bar(x, group.MAE, width=width, color=colors[i], label=arm.replace("gine_", "GINE "))
                ax.errorbar(x, group.MAE, yerr=np.vstack([np.maximum(0, group.MAE-group.MAE_CI_low), np.maximum(0, group.MAE_CI_high-group.MAE)]), fmt="none", ecolor="#333333", capsize=2, linewidth=.8)
            ax.set_xticks(range(3), ["DWI", "fMRI", "Fusion"])
            ax.set_title("No harmonization" if mode == "none" else "Age-informed mean-only harmonization")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("OOF MAE (years)")
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(PUBLIC / f"architecture_comparison.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20261106_full")
    parser.add_argument("--prepared", type=Path, default=RUNTIME / "prepared_20261106")
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    state_path = RUNTIME / f"gine_study_{args.tag}.json"
    jobs = []
    for arm, fraction in [(a, 1.0) for a in ARMS] + [(a, f) for f in [.25, .5, .6, .75, .9] for a in ["gine_2x32", "ridge"]]:
        name = f"{arm}_fraction{round(fraction*100):03d}"
        jobs.append(dict(name=name, arm=arm, fraction=fraction, status="queued", run=str(RUNTIME / f"{name}_{args.tag}")))
    if state_path.exists():
        old = {j["name"]: j for j in json.loads(state_path.read_text())["jobs"]}
        for job in jobs:
            if job["name"] in old:
                job.update(old[job["name"]])

    def update():
        state_path.write_text(json.dumps(dict(updated=now(), python=sys.executable, jobs=jobs), indent=2))
        publish(jobs)

    if args.status_only:
        update()
        return
    for job in jobs:
        if job["status"] != "complete":
            job["status"] = "queued"
    update()
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    extra = "/tmp/avery_pipeline_slides_deps"
    environment["PYTHONPATH"] = extra + os.pathsep + environment.get("PYTHONPATH", "")
    logs = RUNTIME / f"gine_study_logs_{args.tag}"
    logs.mkdir(exist_ok=True)

    def worker(gpu):
        while True:
            with LOCK:
                job = next((j for j in jobs if j["status"] == "queued"), None)
                if job is None:
                    return
                job.update(status="training", gpu=gpu, started=now())
                update()
            env = dict(environment, CUDA_VISIBLE_DEVICES=gpu)
            run = Path(job["run"])
            command = [sys.executable, "code/run_pipeline.py"]
            config = f"code/brainage_pipeline/configs/{job['arm']}.json"
            train = command + ["train", "--prepared", str(args.prepared), "--output", str(run), "--config", config, "--fraction", str(job["fraction"])]
            if run.exists():
                train.append("--resume")
            steps = [train, command + ["audit", "--run", str(run)], command + ["report", "--run", str(run), "--output", str(PUBLIC / job["name"])], command + ["slides", "--run", str(run), "--prepared", str(args.prepared), "--output", str(PUBLIC / job["name"] / "slides")]]
            try:
                with (logs / f"{job['name']}.log").open("a") as log:
                    for index, step in enumerate(steps):
                        log.write(f"\n{now()} {json.dumps(step)}\n")
                        log.flush()
                        if index:
                            with LOCK:
                                job["status"] = ["training", "auditing", "reporting", "slides"][index]
                                update()
                        subprocess.run(step, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                with LOCK:
                    job.update(status="complete", finished=now())
                    update()
            except Exception as exc:
                with LOCK:
                    job.update(status="failed", error=str(exc), finished=now())
                    update()

    stop_refresh = threading.Event()

    def refresh():
        while not stop_refresh.wait(30):
            with LOCK:
                update()

    heartbeat = threading.Thread(target=refresh, daemon=True)
    heartbeat.start()
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(pool.map(worker, args.gpus))
    stop_refresh.set()
    # Size summaries only combine identical model configurations and held-out folds.
    for arm in ["gine_2x32", "ridge"]:
        eligible = [j for j in jobs if j["arm"] == arm and j["status"] == "complete"]
        runs = [j["run"] for j in sorted(eligible, key=lambda j: j["fraction"])]
        if arm == "ridge":
            runs.append(str(RUNTIME / "matched_ridge_20261106"))
        if len(runs) >= 2:
            with (logs / f"learning_curves_{arm}.log").open("a") as log:
                subprocess.run([sys.executable, "code/run_pipeline.py", "learning-curves", "--runs", *runs, "--output", str(PUBLIC / f"training_size_{arm}")], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    update()


if __name__ == "__main__":
    main()
