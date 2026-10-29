#!/usr/bin/env python3
"""Compare raw fMRI CN OOF predictions on identical scans and outer folds."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2] / "results" / "final/fMRI"


def main():
    gnn = pd.read_csv(ROOT / "GNN_BAG/fMRI_GNN_BAG_CN_OOF_predictions.csv").rename(
        columns={"_manifest_dataset": "DATASET", "_manifest_id": "CONNECTOME_ID",
                 "_age": "AGE_GNN", "BRAIN_AGE_GNN_fMRI_OOF": "GNN",
                 "FOLD": "FOLD_GNN"}
    )
    ridge = pd.read_csv(ROOT / "Ridge_BAG/fMRI_Ridge_CN_OOF_predictions.csv").rename(
        columns={"CHRONOLOGICAL_AGE": "AGE_RIDGE", "PREDICTED_AGE_OOF": "RIDGE",
                 "FOLD": "FOLD_RIDGE"}
    )
    paired = gnn.merge(ridge, on=["DATASET", "CONNECTOME_ID"], validate="one_to_one")
    if len(paired) != len(gnn) or len(paired) != len(ridge):
        raise ValueError("GNN and Ridge must have exactly the same CN OOF scans.")
    if not np.allclose(paired.AGE_GNN, paired.AGE_RIDGE):
        raise ValueError("Chronological ages do not match.")
    if not (paired.FOLD_GNN == paired.FOLD_RIDGE).all():
        raise ValueError("Outer CV folds do not match.")
    if not (paired._participant_group == paired.PARTICIPANT_GROUP).all():
        raise ValueError("Participant groups do not match.")
    if not np.isfinite(paired[["AGE_GNN", "GNN", "RIDGE"]].to_numpy()).all():
        raise ValueError("Non-finite ages or predictions.")

    output = ROOT / "model_comparison"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, frame in [("ALL", paired), *paired.groupby("DATASET")]:
        age = frame.AGE_GNN.to_numpy()
        for model in ["GNN", "RIDGE"]:
            predicted = frame[model].to_numpy()
            bag = predicted - age
            rows.append({
                "DATASET": dataset, "model": model, "n_scans": len(frame),
                "n_participants": frame.PARTICIPANT_GROUP.nunique(),
                "MAE_years": mean_absolute_error(age, predicted),
                "RMSE_years": float(np.sqrt(mean_squared_error(age, predicted))),
                "R2": r2_score(age, predicted),
                "Pearson_r": float(np.corrcoef(age, predicted)[0, 1]),
                "mean_BAG_years": float(bag.mean()),
                "BAG_age_r": float(np.corrcoef(age, bag)[0, 1]),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "fMRI_GNN_vs_Ridge_CN_OOF_metrics.csv", index=False)
    paired.to_csv(output / "fMRI_GNN_vs_Ridge_paired_CN_OOF_predictions.csv", index=False)
    errors_gnn = abs(paired.GNN - paired.AGE_GNN)
    errors_ridge = abs(paired.RIDGE - paired.AGE_GNN)
    summary = {
        "n_scans": len(paired),
        "n_participants": int(paired.PARTICIPANT_GROUP.nunique()),
        "same_outer_folds": True,
        "ridge_MAE_reduction_years": float(errors_gnn.mean() - errors_ridge.mean()),
        "ridge_MAE_reduction_percent": float(100 * (1 - errors_ridge.mean() / errors_gnn.mean())),
        "ridge_lower_absolute_error_fraction": float((errors_ridge < errors_gnn).mean()),
        "interpretation": "Raw CN OOF predictions; GNN threshold was selected using these CV results. Upstream full-data ComBat may affect both models' CV estimates.",
    }
    (output / "fMRI_GNN_vs_Ridge_summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    limits = [paired.AGE_GNN.min() - 5, paired.AGE_GNN.max() + 5]
    for ax, model in zip(axes[:2], ["GNN", "RIDGE"]):
        for dataset, frame in paired.groupby("DATASET"):
            ax.scatter(frame.AGE_GNN, frame[model], s=12, alpha=0.5, label=dataset)
        ax.plot(limits, limits, "k--", linewidth=1)
        ax.set(xlabel="Chronological age (years)", ylabel="Predicted age (years)", title=f"{model}: CN OOF")
        ax.legend(fontsize=8)
    datasets = metrics.DATASET.unique()
    x = np.arange(len(datasets))
    for offset, model in [(-0.18, "GNN"), (0.18, "RIDGE")]:
        values = metrics[metrics.model == model].set_index("DATASET").loc[datasets, "MAE_years"]
        axes[2].bar(x + offset, values, width=0.36, label=model)
    axes[2].set(xticks=x, xticklabels=datasets, ylabel="MAE (years)", title="Raw CN OOF error")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output / "fMRI_GNN_vs_Ridge_CN_OOF.png", dpi=200)
    plt.close(fig)
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
