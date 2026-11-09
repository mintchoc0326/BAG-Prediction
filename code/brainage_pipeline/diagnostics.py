"""Training-only shared-coordinate PCA of held-out pre/post-harmonization data."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .data import load
from .reporting import savefig


def diagnostics(run, prepared, output, fold=1):
    run, prepared, output = Path(run), Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text())
    if (
        hashlib.sha256((prepared / "manifest.json").read_bytes()).hexdigest()
        != manifest["prepared_manifest_sha256"]
    ):
        raise ValueError("Prepared inputs differ from training")
    raw, meta = load(prepared, manifest["population"], ("DWI", "fMRI"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    checks = []
    for row, modality in enumerate(["DWI", "fMRI"]):
        records = {}
        for mode in ["none", "age_mean"]:
            path = run / f"fold{fold}" / mode / modality / "ridge/model_private.joblib"
            records[mode] = joblib.load(path)
        a, b = records["none"], records["age_mean"]
        train = a["train_indices"]
        test = a["test_indices"]
        np.testing.assert_array_equal(train, b["train_indices"])
        np.testing.assert_array_equal(test, b["test_indices"])
        np.testing.assert_array_equal(
            a["preprocessor"].transforms[modality].mask_,
            b["preprocessor"].transforms[modality].mask_,
        )
        transformed = {}
        for mode, record in records.items():
            prep = record["preprocessor"]
            transformed[mode] = (
                prep.matrix({modality: raw[modality][train]}, meta.batch.iloc[train]),
                prep.matrix({modality: raw[modality][test]}, meta.batch.iloc[test]),
            )
        pca = PCA(n_components=2, svd_solver="randomized", random_state=42).fit(
            np.concatenate([transformed[m][0] for m in ["none", "age_mean"]])
        )
        colors = {
            label: plt.get_cmap("tab20")(i % 20)
            for i, label in enumerate(sorted(meta.batch.unique()))
        }
        for column, mode in enumerate(["none", "age_mean"]):
            projected = pca.transform(transformed[mode][1])
            ax = axes[row, column]
            labels = meta.batch.iloc[test].to_numpy()
            for label in sorted(set(labels)):
                ix = labels == label
                ax.scatter(
                    projected[ix, 0],
                    projected[ix, 1],
                    s=16,
                    alpha=0.7,
                    color=colors[label],
                    label=label,
                )
            ax.set(
                xlabel="PC1 — shared training-only basis",
                ylabel="PC2 — shared training-only basis",
                title=modality
                + (" / no ComBat" if mode == "none" else " / training-age mean-only"),
            )
            ax.spines[["top", "right"]].set_visible(False)
        checks.append(
            dict(
                modality=modality,
                fold=fold,
                training_CN_scans=len(train),
                heldout_scans=len(test),
                PCA_fit="concatenated training features from both prespecified arms; no held-out rows",
            )
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    savefig(fig, output / "harmonization_PCA")
    (output / "diagnostics.json").write_text(
        json.dumps(
            dict(
                status="complete",
                checks=checks,
                caveat="One descriptive outer fold. Shared axes are fitted on training only. Visual mixing is not evidence that only technical effects were removed.",
            ),
            indent=2,
        )
    )
