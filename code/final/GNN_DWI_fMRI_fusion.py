#!/usr/bin/env python3
"""Supplementary two-branch GNN with matched-sample unimodal controls.

Uses Weighted_Ridge's exact CN scans and outer folds. Graph thresholds are
fixed in advance (top 50% positive edges), never selected using outer OOF
scores. Inner grouped validation selects epochs; each outer model is then
reinitialized and trained on ALL outer-training scans for those epochs.
Run: python code/GNN_DWI_fMRI_fusion.py
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from sklearn.model_selection import GroupShuffleSplit

from Ridge_DWI_fMRI_fusion import ROOT, features, metrics


MODES = ["DWI_GNN", "fMRI_GNN", "GNN_FUSION"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN = 32
DROPOUT = 0.3
RETAIN_PERCENT = 50


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def graph_from_edges(edges, dwi):
    matrix = np.zeros((84, 84), dtype=np.float32)
    upper = np.triu_indices(84, 1)
    matrix[upper] = edges
    matrix[(upper[1], upper[0])] = edges
    if dwi:
        matrix = np.log1p(matrix)
    weights = matrix[upper]
    positive = weights > 0
    if not positive.any():
        raise ValueError("No positive edges in connectome")
    cutoff = np.percentile(weights[positive], 100 - RETAIN_PERCENT)
    keep = positive & (weights >= cutoff)
    i, j = upper[0][keep], upper[1][keep]
    # Positive bounded edge attributes; signed fMRI remains in node features.
    attr = weights[keep] / max(float(weights[positive].max()), 1e-8)
    return Data(x=torch.from_numpy(matrix),
                edge_index=torch.tensor(np.stack([np.r_[i, j], np.r_[j, i]]), dtype=torch.long),
                edge_attr=torch.tensor(np.r_[attr, attr,].reshape(-1, 1), dtype=torch.float32))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        def mlp(n):
            return nn.Sequential(nn.Linear(n, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN))
        self.conv1 = GINEConv(mlp(84), edge_dim=1)
        self.conv2 = GINEConv(mlp(HIDDEN), edge_dim=1)
        self.norm1 = nn.LayerNorm(HIDDEN)
        self.norm2 = nn.LayerNorm(HIDDEN)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, graph):
        x = self.dropout(torch.relu(self.norm1(self.conv1(graph.x, graph.edge_index, graph.edge_attr))))
        x = self.dropout(torch.relu(self.norm2(self.conv2(x, graph.edge_index, graph.edge_attr))))
        return torch.cat([global_mean_pool(x, graph.batch), global_max_pool(x, graph.batch)], dim=1)


class FusionGNN(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.dwi = Encoder() if mode != "fMRI_GNN" else None
        self.fmri = Encoder() if mode != "DWI_GNN" else None
        n = HIDDEN * (4 if mode == "GNN_FUSION" else 2)
        self.head = nn.Sequential(nn.Linear(n, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 1))

    def forward(self, dwi, fmri):
        vectors = []
        if self.dwi is not None:
            vectors.append(self.dwi(dwi))
        if self.fmri is not None:
            vectors.append(self.fmri(fmri))
        return self.head(torch.cat(vectors, dim=1)).flatten()


def collate(rows):
    dwi, fmri, age = zip(*rows)
    return Batch.from_data_list(dwi), Batch.from_data_list(fmri), torch.tensor(age, dtype=torch.float32)


def normalizers(graphs, train):
    stats = []
    for modality in range(2):
        x = torch.cat([graphs[i][modality].x for i in train], dim=0)
        stats.append((x.mean(dim=0), x.std(dim=0, unbiased=False).clamp_min(1e-6)))
    return stats


def loader(graphs, ages, indices, stats, shuffle=False):
    rows = []
    for i in indices:
        pair = []
        for modality in range(2):
            graph = graphs[i][modality].clone()
            mean, std = stats[modality]
            graph.x = (graph.x - mean) / std
            pair.append(graph)
        rows.append((*pair, float(ages[i])))
    return DataLoader(rows, batch_size=16, shuffle=shuffle, collate_fn=collate)


def predict(model, batches, mean, std):
    model.eval()
    values = []
    with torch.no_grad():
        for dwi, fmri, _ in batches:
            values.extend((model(dwi.to(DEVICE), fmri.to(DEVICE)) * std + mean).cpu().tolist())
    return np.asarray(values)


def train_epoch(model, batches, optimizer, mean, std):
    model.train()
    for dwi, fmri, age in batches:
        optimizer.zero_grad()
        target = (age.to(DEVICE) - mean) / std
        loss = nn.functional.mse_loss(model(dwi.to(DEVICE), fmri.to(DEVICE)), target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite GNN loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()


def fit(graphs, ages, train, mode, seed, epochs, validation=None, patience=20):
    seed_everything(seed)
    stats = normalizers(graphs, train)
    mean, std = float(ages[train].mean()), max(float(ages[train].std()), 1e-6)
    batches = loader(graphs, ages, train, stats, shuffle=True)
    validation_batches = loader(graphs, ages, validation, stats) if validation is not None else None
    model = FusionGNN(mode).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    best, best_epoch, stale = np.inf, 1, 0
    history = []
    for epoch in range(1, epochs + 1):
        train_epoch(model, batches, optimizer, mean, std)
        if validation is not None:
            score = float(np.mean(abs(predict(model, validation_batches, mean, std) - ages[validation])))
            history.append({"epoch": epoch, "validation_MAE": score})
            if score < best:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            if stale >= patience:
                break
    return model, stats, mean, std, best_epoch if validation is not None else epochs, history


def sample_graph_fingerprint(cn, graphs):
    """Bind checkpoints to sample order, labels, folds and actual graph inputs."""
    columns = ["dataset", "scan_id", "participant_group", "unified_age", "FOLD"]
    digest = hashlib.sha256(cn[columns].to_csv(index=False, float_format="%.17g").encode())
    for pair in graphs:
        for graph in pair:
            for value in (graph.x, graph.edge_index, graph.edge_attr):
                array = value.detach().cpu().contiguous().numpy()
                digest.update(str((array.shape, str(array.dtype))).encode())
                digest.update(array.tobytes())
    return digest.hexdigest()


def save_outer_checkpoint(path, model, stats, mean, std, mode, fold, epochs,
                          cn, train, test, test_predictions, fingerprint):
    """Save the exact outer model used for OOF metrics, plus inference context."""
    checkpoint = {
        "schema_version": 1,
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "node_normalizers": [(a.detach().cpu().clone(), b.detach().cpu().clone()) for a, b in stats],
        "age_mean": mean, "age_std": std, "mode": mode, "fold": fold,
        "epochs": epochs, "seed": 1042 + fold, "hidden": HIDDEN, "dropout": DROPOUT,
        "retain_positive_percent": RETAIN_PERCENT, "input_fingerprint": fingerprint,
        "train_indices": train.tolist(), "test_indices": test.tolist(),
        "test_keys": cn.iloc[test][["dataset", "scan_id"]].values.tolist(),
        "test_predictions": torch.tensor(test_predictions, dtype=torch.float64),
        "torch_version": str(torch.__version__), "device": str(DEVICE),
        "preprocessing": "DWI clip+log1p; signed fMRI; top50% positive edges; per-graph max edge normalization; training-only node/age statistics",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "results/final/Fusion/Weighted_Ridge")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/final/Fusion/GNN_fusion")
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    if argv is None and "ipykernel" in sys.modules:
        argv = []
    args = parser.parse_args(argv)
    if args.max_epochs < 1 or args.patience < 1:
        parser.error("Epochs and patience must be positive")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    # Use a shared table containing actual OOF predictions and fold IDs.
    cn = pd.read_csv(args.baseline_dir / "CN_OOF_predictions.csv")
    keys = ["dataset", "scan_id"]
    if cn.duplicated(keys).any() or cn.groupby("participant_group").FOLD.nunique().max() != 1:
        raise ValueError("Duplicate scans or participant leakage in baseline folds")
    if set(cn.FOLD) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected five baseline outer folds")
    y = cn.unified_age.to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Missing training ages")
    groups = cn.participant_group.to_numpy()
    graphs = [(graph_from_edges(features(row.connectome_dwi, "DWI"), True),
               graph_from_edges(features(row.connectome_fmri, "fMRI"), False)) for row in cn.itertuples()]
    cn[keys + ["participant_group", "unified_age", "FOLD"]].to_csv(out / "outer_fold_assignments.csv", index=False)
    fingerprint = sample_graph_fingerprint(cn, graphs)
    predictions = {mode: np.full(len(cn), np.nan) for mode in MODES}
    epoch_counts = {mode: [] for mode in MODES}
    fold_rows, inner_rows = [], []
    print(f"Device={DEVICE}; CN scans={len(cn)}; participants={len(set(groups))}", flush=True)
    for fold in range(1, 6):
        train = np.flatnonzero(cn.FOLD.to_numpy() != fold)
        test = np.flatnonzero(cn.FOLD.to_numpy() == fold)
        assert not set(groups[train]) & set(groups[test])
        local_train, local_validation = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42 + fold).split(train, groups=groups[train]))
        inner_train, validation = train[local_train], train[local_validation]
        assert not set(groups[inner_train]) & set(groups[validation])
        for index in train:
            inner_rows.append({"fold": fold, "dataset": cn.iloc[index].dataset,
                               "scan_id": cn.iloc[index].scan_id,
                               "role": "validation" if index in validation else "train"})
        for mode in MODES:
            selected, _, _, _, epochs, history = fit(graphs, y, inner_train, mode, 42 + fold,
                                                     args.max_epochs, validation, args.patience)
            del selected
            pd.DataFrame(history).to_csv(out / f"{mode}_fold_{fold}_early_stopping.csv", index=False)
            model, stats, mean, std, _, _ = fit(graphs, y, train, mode, 1042 + fold, epochs)
            values = predict(model, loader(graphs, y, test, stats), mean, std)
            save_outer_checkpoint(out / "outer_fold_models" / f"{mode}_fold_{fold}.pt",
                                  model, stats, mean, std, mode, fold, epochs,
                                  cn, train, test, values, fingerprint)
            predictions[mode][test] = values
            epoch_counts[mode].append(epochs)
            row = {"fold": fold, "model": mode, "selected_epochs": epochs,
                   "n_inner_train": len(inner_train), "n_inner_validation": len(validation),
                   "n_refit_train": len(train), "n_test": len(test), **metrics(y[test], values)}
            fold_rows.append(row)
            print(f"Fold {fold} {mode}: epochs={epochs}; MAE={row['MAE_years']:.3f}; R2={row['R2']:.3f}", flush=True)
            del model
    pd.DataFrame(inner_rows).to_csv(out / "inner_validation_assignments.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    for mode, values in predictions.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Incomplete OOF predictions: {mode}")
        cn[f"PREDICTED_AGE_{mode}_OOF"] = values
        cn[f"BAG_{mode}_OOF"] = values - y
    cn.to_csv(out / "CN_OOF_predictions.csv", index=False)
    rows = []
    baseline_models = ["DWI", "fMRI", "EARLY_FUSION", "WEIGHTED_EARLY_FUSION", "LATE_MEAN"]
    for dataset, frame in [("ALL", cn), *cn.groupby("dataset")]:
        for mode in baseline_models + MODES:
            column = f"PREDICTED_AGE_{mode}_OOF"
            if column in frame:
                rows.append({"dataset": dataset, "model": mode, "n_scans": len(frame),
                             "n_participants": frame.participant_group.nunique(), **metrics(frame.unified_age, frame[column])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "OOF_model_comparison.csv", index=False)
    final_epochs = {}
    for mode in MODES:
        epochs = max(1, int(np.median(epoch_counts[mode])))
        final_epochs[mode] = epochs
        model, stats, mean, std, _, _ = fit(graphs, y, np.arange(len(cn)), mode, 2042, epochs)
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "mode": mode, "node_normalizers": stats, "age_mean": mean, "age_std": std,
                    "epochs": epochs, "hidden": HIDDEN, "dropout": DROPOUT,
                    "retain_positive_percent": RETAIN_PERCENT,
                    "dwi_transform": "log1p after nonnegative clipping",
                    "edge_attribute_transform": "positive weights divided by per-graph maximum"},
                   out / f"{mode}_final_model.pt")
        del model
    summary = {"n_CN_scans": len(cn), "n_CN_participants": len(set(groups)),
               "device": str(DEVICE), "baseline_dir": str(args.baseline_dir),
               "final_epochs": final_epochs, "fixed_positive_edge_percent": RETAIN_PERCENT,
               "training": "Inner grouped 20% validation selects epochs; fresh refit on all outer-training scans. Feature/target statistics fit separately for inner selection and outer refit.",
               "architecture": "Separate two-layer GINE encoders; hidden=32; mean+max pooling; concatenated embeddings and nonlinear age head",
               "limitations": ["Single seed exploratory comparison; no claim of significant improvement.",
                               "Same upstream full-data ComBat and metadata/age filtering limitations as Ridge baselines.",
                               "GNN log-scaled DWI and graph representations differ from Ridge features.",
                               "Final epochs are median outer-fold choices; final models are not independent evaluation results."]}
    (out / "GNN_fusion_summary.json").write_text(json.dumps(summary, indent=2))
    pooled = table[table.dataset.eq("ALL")]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(pooled.model, pooled.MAE_years)
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_ylabel("CN OOF MAE (years)")
    fig.tight_layout()
    fig.savefig(out / "OOF_model_comparison.png", dpi=200)
    plt.close(fig)
    print(pooled.to_string(index=False))
    print("Saved to:", out)


if __name__ == "__main__":
    main()
