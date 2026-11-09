"""Aggregate reports and figures; historical and pilot results are labeled."""
import html
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_foldwise_DWI_Ridge import metrics

STYLES = [
    ("ridge", "DWI", "Ridge–DWI", "#4878A8"),
    ("ridge", "fMRI", "Ridge–fMRI", "#C6A83E"),
    ("ridge", "fusion", "Ridge–fusion", "#BC5B58"),
    ("gine", "DWI", "GINE–DWI", "#59926C"),
    ("gine", "fMRI", "GINE–fMRI", "#8D74AA"),
    ("gine", "fusion", "GINE–fusion", "#D08B4B"),
]


def bootstrap_metrics(frame, replicates=3000, seed=20261106):
    """Fixed-OOF participant bootstrap stratified by cohort; no model refitting."""
    d = frame.copy()
    d["n"] = 1.0
    d["ae"] = abs(d.predicted_age - d.age)
    d["se"] = (d.predicted_age - d.age) ** 2
    d["a2"] = d.age**2
    d["p2"] = d.predicted_age**2
    d["ap"] = d.age * d.predicted_age
    columns = ["n", "ae", "se", "age", "a2", "predicted_age", "p2", "ap"]
    groups = d.groupby(["cohort", "participant_group"])[columns].sum().reset_index()
    totals = np.zeros((replicates, len(columns)))
    rng = np.random.default_rng(seed)
    for _, part in groups.groupby("cohort"):
        weights = rng.multinomial(
            len(part), np.full(len(part), 1 / len(part)), size=replicates
        )
        totals += weights @ part[columns].to_numpy()
    n, ae, se, a, a2, p, p2, ap = totals.T
    with np.errstate(divide="ignore", invalid="ignore"):
        age_ss = a2 - a * a / n
        prediction_ss = p2 - p * p / n
        draws = {
            "MAE": ae / n,
            "RMSE": np.sqrt(se / n),
            "R2": 1 - se / age_ss,
            "Pearson_r": (ap - a * p / n) / np.sqrt(age_ss * prediction_ss),
        }
    result = {}
    for name, values in draws.items():
        valid = values[np.isfinite(values)]
        lo, hi = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
        result[name + "_CI_low"] = lo
        result[name + "_CI_high"] = hi
    return result


def savefig(fig, path):
    for ext in ["png", "pdf", "svg"]:
        target = path.with_suffix("." + ext)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        if ext == "svg":
            target.write_text(
                "\n".join(x.rstrip() for x in target.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def report(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text())

    def save_current(fig, path):
        if manifest["settings"].get("validation_only"):
            fig.suptitle("Software validation only — short training budget", fontsize=9)
        savefig(fig, path)

    rows = []
    probes = []
    calibration_fallbacks = []
    for p in run.glob("fold*/*/*/*/predictions_private.csv"):
        d = pd.read_csv(p)
        d["fold"] = int(p.parts[-5].replace("fold", ""))
        d["mode"] = p.parts[-4]
        d["modality"] = p.parts[-3]
        d["learner"] = p.parts[-2]
        rows.append(d)
        task_info = json.loads(p.with_name("task.json").read_text())
        calibration_fallbacks.append(
            dict(
                fold=d["fold"].iloc[0],
                mode=d["mode"].iloc[0],
                modality=d["modality"].iloc[0],
                learner=d["learner"].iloc[0],
                global_fallback_cohorts=", ".join(
                    task_info.get("calibration_fallback_cohorts", [])
                ),
            )
        )
        probe_path = p.with_name("retention_probes.json")
        if probe_path.exists():
            probes.append(
                dict(
                    fold=d["fold"].iloc[0],
                    mode=d["mode"].iloc[0],
                    modality=d["modality"].iloc[0],
                    learner=d["learner"].iloc[0],
                    **json.loads(probe_path.read_text()),
                )
            )
    if not rows:
        raise ValueError("No completed predictions")
    pred = pd.concat(rows, ignore_index=True)
    summary = []
    calibration = []
    for (mode, modality, learner), part in pred[pred.is_cn].groupby(
        ["mode", "modality", "learner"]
    ):
        for cohort, d in [("Pooled", part)] + list(part.groupby("cohort")):
            summary.append(
                dict(
                    mode=mode,
                    modality=modality,
                    learner=learner,
                    cohort=cohort,
                    n_participants=d.participant_group.nunique(),
                    **metrics(d.age.to_numpy(), d.predicted_age.to_numpy()),
                    **bootstrap_metrics(d),
                )
            )
            for field in ["BAG", "cBAG_global", "cBAG"]:
                calibration.append(
                    dict(
                        mode=mode,
                        modality=modality,
                        learner=learner,
                        cohort=cohort,
                        measure=field,
                        age_r=d.age.corr(d[field]),
                        mean=d[field].mean(),
                        SD=d[field].std(),
                        n_scans=len(d),
                    )
                )
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "performance.csv", index=False)
    calibration = pd.DataFrame(calibration)
    calibration.to_csv(output / "age_calibration.csv", index=False)
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        for mode in summary["mode"].unique():
            data = summary[(summary["mode"] == mode) & summary.cohort.eq("Pooled")]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
            labels = []
            mae = []
            r2 = []
            colors = []
            for learner, modality, label, color in STYLES:
                d = data[data.learner.eq(learner) & data.modality.eq(modality)]
                if len(d):
                    labels.append(label)
                    mae.append(d.MAE.iloc[0])
                    r2.append(d.R2.iloc[0])
                    colors.append(color)
            for ax, values, name in zip(axes, [mae, r2], ["MAE (years)", "R²"]):
                ax.bar(np.arange(len(values)), values, color=colors, width=0.65)
                metric = "MAE" if name.startswith("MAE") else "R2"
                lows = []
                highs = []
                for learner, modality, _, _ in STYLES:
                    entry = data[data.learner.eq(learner) & data.modality.eq(modality)]
                    if len(entry):
                        lows.append(entry[metric + "_CI_low"].iloc[0])
                        highs.append(entry[metric + "_CI_high"].iloc[0])
                positions = np.arange(len(values))
                ax.vlines(positions, lows, highs, color="#444444", lw=0.8)
                ax.hlines(
                    lows, positions - 0.05, positions + 0.05, color="#444444", lw=0.8
                )
                ax.hlines(
                    highs, positions - 0.05, positions + 0.05, color="#444444", lw=0.8
                )

                ax.set_xticks(np.arange(len(values)), labels, rotation=30, ha="right")
                ax.set_ylabel(name)
                ax.spines[["top", "right"]].set_visible(False)
                ax.axhline(0, color="#999", lw=0.6)
            fig.tight_layout()
            save_current(fig, output / f"performance_{mode}")
        # Raw loss/MAE curves for the selected candidate only, including independent stopping set.
        for p in run.glob("fold*/*/*/gine/learning_history.csv"):
            d = pd.read_csv(p)
            selection = pd.read_csv(p.with_name("selection.csv"))
            selected = int(selection.loc[selection.selected, "candidate"].iloc[0])
            d = d[(d.candidate == selected) & d.stage.eq("early_stopping")]
            if d.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
            for inner, h in d.groupby("inner_fold"):
                axes[0].plot(h.epoch, h.train_mse_z, label=f"Train {inner}")
                axes[0].plot(
                    h.epoch, h.validation_mse_z, alpha=0.7, label=f"Validation {inner}"
                )
                axes[1].plot(h.epoch, h.train_mae)
                axes[1].plot(h.epoch, h.validation_mae, alpha=0.7)
                for ax in axes:
                    ax.axvline(h.selected_epoch.iloc[0], lw=0.6, color="#999")
            axes[0].set(xlabel="Epoch", ylabel="MSE (standardized age)")
            axes[1].set(xlabel="Epoch", ylabel="MAE (years)")
            axes[0].legend(frameon=False, fontsize=6, ncol=2)
            for ax in axes:
                ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            stem = "_".join(p.parts[-5:-1])
            save_current(fig, output / f"curves_{stem}")
        for (mode, modality, learner), part in pred[pred.is_cn].groupby(
            ["mode", "modality", "learner"]
        ):
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
            for cohort, d in part.groupby("cohort"):
                axes[0].scatter(d.age, d.cBAG_global, s=9, alpha=0.35, label=cohort)
                axes[1].scatter(d.age, d.cBAG, s=9, alpha=0.35, label=cohort)
            for ax, name in zip(
                axes, ["Global training calibration", "Cohort training calibration"]
            ):
                ax.set(xlabel="Chronological age", ylabel="OOF cBAG", title=name)
                ax.axhline(0, lw=0.6, color="#999")
                ax.spines[["top", "right"]].set_visible(False)
            axes[0].legend(frameon=False, fontsize=7)
            fig.tight_layout()
            save_current(fig, output / f"calibration_{mode}_{modality}_{learner}")
    probes = pd.DataFrame(probes)
    probes.to_csv(output / "retention_probes.csv", index=False)
    calibration_fallbacks = pd.DataFrame(calibration_fallbacks)
    calibration_fallbacks.to_csv(output / "calibration_fallbacks.csv", index=False)
    sections = [
        "<h1>Unified participant-nested brain-age pipeline</h1>",
        "<p>Run status: <b>"
        + html.escape(manifest["status"])
        + "</b>. Population: "
        + html.escape(manifest["population"])
        + ". "
        + (
            "Validation/pilot epoch budget; not a scientific performance benchmark."
            if manifest["settings"].get("validation_only")
            else "Development evaluation; prior OOF results informed pipeline design, so independent validation remains necessary."
        )
        + "</p>",
        "<p>Common participant folds across DWI, fMRI, fusion, Ridge and GINE within this run. Intervals are participant-cluster bootstrap 95% intervals stratified by cohort, conditional on fixed OOF models; refitting, selection and split uncertainty are excluded. All preprocessing, graph sparsity and model choices remain inside training. GINE stopping uses a separate split within inner training, followed by refitting before inner validation. Cohort calibration uses training-only inner OOF predictions. No held-out age enters features.</p>",
        "<h2>Prediction metrics — CN only</h2>",
        summary.to_html(index=False, float_format=lambda x: f"{x:.3f}"),
        "<h2>Held-out retention probes</h2>",
        probes.to_html(index=False, float_format=lambda x: f"{x:.3f}"),
        "<h2>Age calibration — inspect each cohort</h2>",
        calibration.to_html(index=False, float_format=lambda x: f"{x:.3f}"),
        "<h2>Sparse-cohort calibration fallbacks</h2><p>Cohorts with fewer than ten training-reference scans use the global trend; this is not a cohort-specific fit.</p>",
        calibration_fallbacks.to_html(index=False),
    ]
    for p in sorted(output.glob("*.svg")):
        sections += [
            "<h2>"
            + html.escape(p.stem.replace("_", " "))
            + '</h2><img src="'
            + html.escape(p.name)
            + '">'
        ]
    sections += [
        "<h2>Limits and next steps</h2><p>Unresolved source diagnoses, scan/clinical dates, pairing age conflicts, ADRC protocol consistency and ROI order. No claim that lower scanner predictability proves complete technical removal or that cBAG is already a validated biomarker. T2D/HbA1c require verified coding and units. Fold-specific embedding spaces must not be pooled without alignment.</p>"
    ]
    css = "<style>body{font:14px Arial;margin:32px;color:#222}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px}img{max-width:100%}h2{margin-top:32px}</style>"
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">' + css + "".join(sections)
    )
    return summary
