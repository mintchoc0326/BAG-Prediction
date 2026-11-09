"""Fold-specific embedding diagnostics, original-checkpoint gradients and size curves."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from .reporting import savefig


def embeddings(run, output, clinical_spec=None):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(run / "metadata_private.csv")
    results = []
    clinical = None
    extra_targets = {}
    if clinical_spec is not None:
        spec = json.loads(Path(clinical_spec).read_text())
        if not spec.get("source_verified") or not spec.get("visit_alignment_verified"):
            raise ValueError(
                "Embedding overlays require verified clinical source and visit alignment"
            )
        clinical = pd.read_csv(spec["source_csv"])
        if clinical.duplicated(["cohort", "scan_id"]).any():
            raise ValueError("Duplicate clinical keys")
        for target, field in spec["variables"].items():
            if not field.get("verified"):
                continue
            if target == "T2D" and not field.get("type2_definition_verified"):
                continue
            if target == "HbA1c" and field.get("unit") not in {"percent", "mmol/mol"}:
                continue
            name = "overlay_" + target
            if field["kind"] == "categorical":
                mapping = {str(k): v for k, v in field["mapping"].items()}
                original = clinical[field["column"]].astype("string").str.strip()
                unknown = set(original.dropna()) - set(mapping)
                if unknown:
                    raise ValueError("Unmapped embedding overlay codes: " + target)
                clinical[name] = original.map(mapping)
            else:
                clinical[name] = pd.to_numeric(
                    clinical[field["column"]], errors="raise"
                )
            extra_targets[name] = field["kind"]
    manifest = json.loads((run / "manifest.json").read_text())
    for p in run.glob("fold*/*/*/gine/embeddings_private.npz"):
        archive = np.load(p)
        test = meta.iloc[archive["test_indices"]].reset_index(drop=True)
        projection = archive["test_projection"]
        if clinical is not None:
            test = test.merge(
                clinical[["cohort", "scan_id"] + list(extra_targets)],
                on=["cohort", "scan_id"],
                how="left",
                validate="one_to_one",
            )
        if projection.shape[1] < 2:
            continue
        stem = "_".join(p.parts[-5:-1])
        for target, kind in extra_targets.items():
            known = test[target].notna().to_numpy()
            if not known.any():
                continue
            fig, ax = plt.subplots(figsize=(5, 4))
            if kind == "continuous":
                points = ax.scatter(
                    projection[known, 0],
                    projection[known, 1],
                    c=test.loc[known, target],
                    s=14,
                    cmap="viridis",
                )
                fig.colorbar(points, ax=ax, label=target)
            else:
                for label in sorted(test.loc[known, target].unique()):
                    ix = test[target].eq(label).to_numpy()
                    ax.scatter(
                        projection[ix, 0],
                        projection[ix, 1],
                        s=14,
                        alpha=0.6,
                        label=label,
                    )
                ax.legend(frameon=False, fontsize=7)
            ax.set(
                xlabel="Training-fit embedding PC1", ylabel="Training-fit embedding PC2"
            )
            fig.tight_layout()
            savefig(fig, output / f"{stem}_{target}")
        for target in ["cohort", "sex", "clinical_diagnosis"]:
            fig, ax = plt.subplots(figsize=(5, 4))
            labels = test[target].astype(str)
            for label in sorted(labels.unique()):
                ix = labels.eq(label).to_numpy()
                ax.scatter(
                    projection[ix, 0], projection[ix, 1], s=14, alpha=0.6, label=label
                )
            ax.set(
                xlabel="Training-fit embedding PC1", ylabel="Training-fit embedding PC2"
            )
            ax.legend(frameon=False, fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            savefig(fig, output / f"{stem}_{target}")
            counts = labels.value_counts()
            score = None
            if 1 < len(counts) < len(labels) and counts.min() >= 2:
                score = float(silhouette_score(archive["test"], labels))
            results.append(
                dict(
                    task=stem,
                    target=target,
                    silhouette=score,
                    n_heldout_scans=len(test),
                    status="within-fold exploratory; no cBAG used as an input",
                )
            )
        fig, ax = plt.subplots(figsize=(5, 4))
        points = ax.scatter(
            projection[:, 0], projection[:, 1], c=test.age, s=14, cmap="viridis"
        )
        fig.colorbar(points, ax=ax, label="Chronological age (overlay only)")
        fig.tight_layout()
        savefig(fig, output / f"{stem}_age")
    pd.DataFrame(results).to_csv(output / "embedding_diagnostics.csv", index=False)
    (output / "status.json").write_text(
        json.dumps(
            dict(
                status="complete" if results else "no_completed_GINE_embeddings",
                validation_only=manifest["settings"].get("validation_only", False),
                caveat="Each fold has its own model and coordinate system. Raw embeddings are never pooled. PCA uses training embeddings only. Colors are validation overlays, not inputs or evidence of causal subtypes.",
            ),
            indent=2,
        )
    )


def attribute(run, task, prepared, output, roi_labels=None):
    import torch
    from torch_geometric.data import Batch
    import joblib
    from .gnn import GINE, graphs, DEVICE
    from .data import load

    run, task, output = Path(run), Path(task), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not task.resolve().is_relative_to(run.resolve()):
        raise ValueError("Task is outside the selected run")
    manifest = json.loads((run / "manifest.json").read_text())
    if (
        hashlib.sha256((Path(prepared) / "manifest.json").read_bytes()).hexdigest()
        != manifest["prepared_manifest_sha256"]
    ):
        raise ValueError("Attribution inputs do not match training preparation")
    record = joblib.load(task / "model_private.joblib")
    if record["learner"] != "gine":
        raise ValueError("Attribution requires original GINE checkpoint")
    modalities = tuple(record["preprocessor"].transforms)
    raw, meta = load(prepared, manifest["population"], modalities)
    test = np.asarray(record["test_indices"])
    test = test[meta.is_cn.iloc[test].to_numpy()]
    values = {m: X[test] for m, X in raw.items()}
    prep = record["preprocessor"]
    config = record["config"]
    items = graphs(
        values,
        prep.graph_values(values, meta.batch.iloc[test]),
        config["threshold"],
        prep.transforms,
    )
    checkpoint = torch.load(
        task / "checkpoint_private.pt", map_location=DEVICE, weights_only=True
    )
    model = GINE(
        checkpoint["node_dim"],
        checkpoint["edge_dim"],
        config["layers"],
        config["width"],
        config.get("residual", False),
        manifest["settings"]["training"].get("dropout", 0.3),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    node_sum = np.zeros(84)
    edge_sum = np.zeros((84, 84))
    edge_count = np.zeros((84, 84))
    for item in items:
        batch = Batch.from_data_list([item]).to(DEVICE)
        batch.x.requires_grad_(True)
        batch.edge_attr.requires_grad_(True)
        prediction = model(batch).sum() * checkpoint["age_std"]
        node_grad, edge_grad = torch.autograd.grad(
            prediction, [batch.x, batch.edge_attr]
        )
        node_sum += (node_grad * batch.x).abs().sum(dim=1).detach().cpu().numpy()
        edge = (edge_grad * batch.edge_attr).abs().sum(dim=1).detach().cpu().numpy()
        indices = batch.edge_index.detach().cpu().numpy()
        np.add.at(edge_sum, (indices[0], indices[1]), edge)
        np.add.at(edge_count, (indices[0], indices[1]), 1)
    labels = [f"ROI_{i+1:02}" for i in range(84)]
    verified = False
    networks = None
    if roi_labels is not None:
        specification = json.loads(Path(roi_labels).read_text())
        if (
            specification.get("roi_order_verified") is not True
            or len(specification.get("labels", [])) != 84
        ):
            raise ValueError("Require verified 84-ROI order and labels")
        labels = specification["labels"]
        verified = True
        if "networks" in specification:
            if len(specification["networks"]) != 84 or any(
                not str(x).strip() for x in specification["networks"]
            ):
                raise ValueError(
                    "Network mapping must contain 84 nonempty verified entries"
                )
            networks = specification["networks"]
    pd.DataFrame(
        {
            "roi_index": range(84),
            "label": labels,
            "mean_abs_gradient_times_input": node_sum / max(len(items), 1),
        }
    ).to_csv(output / "node_contributions.csv", index=False)
    rows = []
    for i, j in zip(*np.triu_indices(84, 1)):
        count = edge_count[i, j] + edge_count[j, i]
        rows.append(
            dict(
                roi_i=i,
                roi_j=j,
                mean_abs_gradient_times_input=(edge_sum[i, j] + edge_sum[j, i]) / count
                if count
                else 0.0,
                directed_occurrences=int(count),
            )
        )
    pd.DataFrame(rows).to_csv(output / "edge_contributions.csv", index=False)
    if networks is not None:
        node_table = pd.DataFrame(
            {
                "network": networks,
                "mean_abs_node_contribution": node_sum / max(len(items), 1),
            }
        )
        node_table.groupby("network").agg(
            total_abs_node_contribution=("mean_abs_node_contribution", "sum"),
            mean_abs_node_contribution=("mean_abs_node_contribution", "mean"),
            n_ROIs=("network", "size"),
        ).to_csv(output / "network_node_contributions.csv")
        edge_table = pd.DataFrame(rows)
        pairs = [
            tuple(sorted([networks[row["roi_i"]], networks[row["roi_j"]]]))
            for row in rows
        ]
        edge_table["network_i"] = [pair[0] for pair in pairs]
        edge_table["network_j"] = [pair[1] for pair in pairs]
        edge_table.groupby(["network_i", "network_j"]).agg(
            mean_abs_edge_contribution=("mean_abs_gradient_times_input", "mean"),
            n_ROI_pairs=("roi_i", "size"),
        ).to_csv(output / "network_edge_contributions.csv")
    (output / "status.json").write_text(
        json.dumps(
            dict(
                status="complete",
                validation_only=manifest["settings"].get("validation_only", False),
                CN_heldout_scans=len(items),
                roi_order_verified=verified,
                network_mapping_verified=networks is not None,
                method="absolute gradient-times-input on standardized node profiles and signed edge attributes; local sensitivity, not causal effect",
                scope="one original outer-fold checkpoint; average separately by fold before scientific interpretation",
            ),
            indent=2,
        )
    )


def learning_curves(runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    reference = None
    for run in map(Path, runs):
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError("Training-size curves require completed runs")
        plan = json.loads((run / "fold_plan_private.json").read_text())
        test_map = [f["test"] for f in plan["folds"]]
        if reference is None:
            reference = (
                manifest["prepared_manifest_sha256"],
                test_map,
                manifest["population"],
                manifest["settings_sha256"],
            )
        if reference != (
            manifest["prepared_manifest_sha256"],
            test_map,
            manifest["population"],
            manifest["settings_sha256"],
        ):
            raise ValueError(
                "Size curves require identical held-out observations/folds"
            )
        d = pd.read_csv(run / "task_summary.csv")
        for row in d.to_dict("records"):
            rows.append(
                dict(
                    training_fraction=manifest["training_fraction"], run=str(run), **row
                )
            )
    d = pd.DataFrame(rows)
    d.to_csv(output / "training_size_metrics.csv", index=False)
    for (mode, modality), part in d.groupby(["mode", "modality"]):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for learner, g in part.groupby("learner"):
            agg = (
                g.groupby("training_fraction")
                .agg(n_training=("n_training_CN", "mean"), MAE=("MAE", "mean"))
                .sort_index()
            )
            ax.plot(agg.n_training, agg.MAE, marker="o", label=learner)
        ax.set(
            xlabel="CN training scans (actual)", ylabel="Mean outer-fold MAE (years)"
        )
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        savefig(fig, output / f"training_size_{mode}_{modality}")
