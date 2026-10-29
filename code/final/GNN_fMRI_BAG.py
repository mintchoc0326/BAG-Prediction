#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fMRI GNN brain-age prediction with threshold selection.

Selects the graph threshold with the lowest pooled CN OOF MAE
(RMSE is the tie-breaker), then reports per-dataset and pooled CN
OOF metrics.
"""

import copy
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from fmri_bag_cv import GroupKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/mnt/newStor/paros/paros_WORK/avery/results/final/fMRI"
)

MANIFEST_PATH = (
    PROJECT_DIR
    / "harmonized_connectomes"
    / "harmonized_fMRI_manifest.csv"
)

METADATA_SEARCH_DIR = PROJECT_DIR
METADATA_FILENAME = "02_canonical_metadata.csv"

OUTPUT_DIR = PROJECT_DIR / "GNN_BAG"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# None keeps all positive edges.
# Numeric values keep the strongest X percent of positive edges.
RETAIN_POSITIVE_PERCENT = [None, 75, 50, 25, 10]

# fMRI HABS has explicit CN/MCI/AD diagnoses: use those labels.
# Do not apply the DWI whole-cohort HABS override to this population.
DATASET_CN_OVERRIDES = {}

SEED = 42
N_SPLITS = 5

HIDDEN_DIM = 64
DROPOUT = 0.30
BATCH_SIZE = 16
MAX_EPOCHS = 250
PATIENCE = 25
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# METADATA AND ID HELPERS
# ============================================================

def find_column(df, candidates, label):
    """Find a column without case sensitivity."""
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise KeyError(
        f"Missing {label}. "
        f"Tried {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_id(value):
    """Normalize IDs for matching manifest and metadata rows."""
    if pd.isna(value):
        return ""

    return re.sub(
        r"\.0$",
        "",
        str(value).strip().upper(),
    )


def is_cn(value):
    """Identify cognitively normal/control diagnosis labels."""
    if pd.isna(value):
        return False

    return str(value).strip().upper() in {
        "CN",
        "HC",
        "NC",
        "CU",
        "CONTROL",
        "HEALTHY",
        "NORMAL",
        "COGNITIVELY NORMAL",
        "HEALTHY CONTROL",
    }


def make_participant_group(dataset, scan_id):
    """Keep repeated visits from the same participant in one CV group."""
    base_id = re.sub(
        r"_Y\d+$",
        "",
        normalize_id(scan_id),
    )

    return f"{str(dataset).strip().upper()}__{base_id}"


def load_inputs():
    """Load manifest and metadata, then match rows by dataset and ID."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    metadata_candidates = sorted(
        METADATA_SEARCH_DIR.rglob(METADATA_FILENAME)
    )

    if not metadata_candidates:
        raise FileNotFoundError(
            f"Could not find {METADATA_FILENAME} "
            f"under {METADATA_SEARCH_DIR}"
        )

    if len(metadata_candidates) > 1:
        print(
            "Multiple metadata files found. Using:",
            metadata_candidates[0],
        )

    metadata_path = metadata_candidates[0]
    manifest = pd.read_csv(MANIFEST_PATH)
    metadata = pd.read_csv(metadata_path, low_memory=False)

    manifest_dataset_col = find_column(
        manifest,
        ["DATASET", "dataset"],
        "manifest dataset",
    )

    manifest_id_col = find_column(
        manifest,
        ["connectome_key", "__id__", "subject_id"],
        "manifest connectome ID",
    )

    connectome_path_col = find_column(
        manifest,
        ["harmonized_connectome"],
        "harmonized connectome path",
    )

    metadata_dataset_col = find_column(
        metadata,
        ["DATASET", "dataset"],
        "metadata dataset",
    )

    metadata_id_col = find_column(
        metadata,
        ["__id__", "connectome_key", "subject_id", "SUBJECT_ID"],
        "metadata connectome ID",
    )

    age_col = find_column(
        metadata,
        ["AGE", "age", "Age"],
        "age",
    )

    # Prefer explicit diagnosis/cognition columns over generic GROUP.
    diagnosis_col = find_column(
        metadata,
        [
            "DIAGNOSIS_BAG",
            "DIAGNOSIS",
            "diagnosis",
            "NORMCOG",
            "GROUP",
            "group",
            "group_status",
        ],
        "diagnosis/control status",
    )

    manifest["_dataset_key"] = (
        manifest[manifest_dataset_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    manifest["_id_key"] = manifest[manifest_id_col].map(normalize_id)

    # Preserve manifest values before merging.
    manifest["_manifest_dataset"] = manifest[manifest_dataset_col]
    manifest["_manifest_id"] = manifest[manifest_id_col]

    metadata["_dataset_key"] = (
        metadata[metadata_dataset_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    metadata["_id_key"] = metadata[metadata_id_col].map(normalize_id)
    metadata["_age"] = pd.to_numeric(
        metadata[age_col],
        errors="coerce",
    )

    # Keep one metadata row per dataset and connectome ID.
    metadata = metadata.drop_duplicates(
        ["_dataset_key", "_id_key"],
        keep="first",
    )

    aligned = manifest.merge(
        metadata,
        on=["_dataset_key", "_id_key"],
        how="inner",
        suffixes=("_manifest", ""),
    )

    aligned = aligned.dropna(
        subset=["_age", connectome_path_col]
    ).reset_index(drop=True)

    if aligned.empty:
        raise ValueError(
            "No connectomes matched metadata. "
            "Check dataset and ID columns."
        )

    # Identify CN from the selected diagnosis field.
    aligned["_is_cn"] = aligned[diagnosis_col].map(is_cn)

    # Apply explicit cohort-level overrides.
    dataset_labels = (
        aligned["_manifest_dataset"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    for dataset, is_cn_override in DATASET_CN_OVERRIDES.items():
        dataset_mask = dataset_labels == dataset.upper()
        aligned.loc[dataset_mask, "_is_cn"] = is_cn_override

    print("\nCN scans by dataset:")
    print(
        aligned.groupby("_manifest_dataset")["_is_cn"]
        .sum()
        .to_string()
    )

    aligned["_participant_group"] = [
        make_participant_group(dataset, scan_id)
        for dataset, scan_id in zip(
            aligned["_manifest_dataset"],
            aligned["_manifest_id"],
        )
    ]

    print(f"\nMatched scans: {len(aligned)}")
    print(f"CN scans: {int(aligned['_is_cn'].sum())}")
    print(f"Diagnosis column selected: {diagnosis_col}")
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nScans by dataset:")
    print(
        aligned["_manifest_dataset"]
        .value_counts()
        .to_string()
    )

    return (
        aligned,
        "_manifest_dataset",
        "_manifest_id",
        connectome_path_col,
        metadata_path,
    )


# ============================================================
# CONNECTOME LOADING AND GRAPH CONSTRUCTION
# ============================================================

def load_matrix(path):
    """Load and prepare one harmonized 84-by-84 connectome matrix."""
    matrix = pd.read_csv(
        path,
        header=None,
    ).to_numpy(dtype=np.float32)

    if matrix.shape != (84, 84):
        raise ValueError(
            f"Expected an 84x84 matrix, got {matrix.shape}: {path}"
        )

    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)

    # Retain negative fMRI correlations in node connectivity profiles.

    return matrix


def matrix_to_graph(matrix, age, retain_percent):
    """
    Convert a connectome matrix to a graph.

    Node features are the full 84-value connectivity profiles.
    Graph edges are selected from positive connections.
    """
    upper_i, upper_j = np.triu_indices(84, k=1)
    upper_weights = matrix[upper_i, upper_j]

    positive_mask = upper_weights > 0
    positive_weights = upper_weights[positive_mask]

    if positive_weights.size == 0:
        raise ValueError("Connectome has no positive edges.")

    if retain_percent is None:
        keep_mask = positive_mask
        cutoff_value = float(np.min(positive_weights))
    else:
        percentile = 100.0 - float(retain_percent)
        cutoff_value = float(
            np.percentile(positive_weights, percentile)
        )
        keep_mask = (
            positive_mask
            & (upper_weights >= cutoff_value)
        )

    source_nodes = upper_i[keep_mask]
    target_nodes = upper_j[keep_mask]
    edge_weights = upper_weights[keep_mask]

    if edge_weights.size == 0:
        raise ValueError("No edges remain after thresholding.")

    # Add both directions for message passing.
    edge_index = np.vstack([
        np.concatenate([source_nodes, target_nodes]),
        np.concatenate([target_nodes, source_nodes]),
    ])

    edge_attr = np.concatenate(
        [edge_weights, edge_weights]
    ).astype(np.float32).reshape(-1, 1)

    graph = Data(
        x=torch.tensor(matrix, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor([float(age)], dtype=torch.float32),
    )

    # Count undirected ROI pairs once.
    return graph, int(len(edge_weights)), cutoff_value


# ============================================================
# GNN ARCHITECTURE
# ============================================================

class FmriGNN(nn.Module):
    """Three-layer GINEConv graph-level age regressor."""

    def __init__(self, node_dim=84):
        super().__init__()

        def make_mlp(input_dim, output_dim):
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            )

        self.conv1 = GINEConv(
            make_mlp(node_dim, HIDDEN_DIM),
            edge_dim=1,
        )

        self.conv2 = GINEConv(
            make_mlp(HIDDEN_DIM, HIDDEN_DIM),
            edge_dim=1,
        )

        self.conv3 = GINEConv(
            make_mlp(HIDDEN_DIM, HIDDEN_DIM),
            edge_dim=1,
        )

        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM)
        self.bn2 = nn.BatchNorm1d(HIDDEN_DIM)
        self.bn3 = nn.BatchNorm1d(HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)

        self.regressor = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, data):
        x = self.conv1(
            data.x,
            data.edge_index,
            data.edge_attr,
        )
        x = F.relu(self.bn1(x))
        x = self.dropout(x)

        x = self.conv2(
            x,
            data.edge_index,
            data.edge_attr,
        )
        x = F.relu(self.bn2(x))
        x = self.dropout(x)

        x = self.conv3(
            x,
            data.edge_index,
            data.edge_attr,
        )
        x = F.relu(self.bn3(x))

        graph_embedding = global_mean_pool(
            x,
            data.batch,
        )

        return self.regressor(graph_embedding).squeeze(-1)


# ============================================================
# TRAINING AND PREDICTION HELPERS
# ============================================================

def scale_age_targets(indices, graphs, age_mean, age_std):
    """Copy graphs and standardize age using training-set statistics."""
    scaled_graphs = []

    for index in indices:
        graph = copy.deepcopy(graphs[int(index)])
        age = float(graph.y.item())
        standardized_age = (age - age_mean) / age_std
        graph.y = torch.tensor(
            [standardized_age],
            dtype=torch.float32,
        )
        scaled_graphs.append(graph)

    return scaled_graphs


def predict_standardized(model, graph_list):
    loader = DataLoader(
        graph_list,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            batch_predictions = model(batch)
            predictions.extend(
                batch_predictions.cpu().numpy().tolist()
            )

    return np.asarray(predictions, dtype=float)


def train_one_epoch(model, loader, optimizer):
    model.train()

    total_loss = 0.0
    total_scans = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        predictions = model(batch)
        targets = batch.y.view(-1).float()
        loss = F.mse_loss(predictions, targets)

        loss.backward()
        optimizer.step()

        n_scans = int(targets.numel())
        total_loss += float(loss.item()) * n_scans
        total_scans += n_scans

    return total_loss / max(total_scans, 1)


def fit_one_fold(graphs, train_indices, validation_indices, seed):
    """Fit a fold model and use validation MAE for early stopping."""
    set_seed(seed)

    train_ages = np.asarray(
        [float(graphs[i].y.item()) for i in train_indices],
        dtype=float,
    )

    age_mean = float(train_ages.mean())
    age_std = max(float(train_ages.std()), 1e-8)

    train_graphs = scale_age_targets(
        train_indices,
        graphs,
        age_mean,
        age_std,
    )

    validation_graphs = scale_age_targets(
        validation_indices,
        graphs,
        age_mean,
        age_std,
    )

    train_loader = DataLoader(
        train_graphs,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = FmriGNN(
        node_dim=graphs[0].x.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_validation_mae = np.inf
    best_model_state = None
    best_epoch = 1
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer)

        validation_prediction_z = predict_standardized(
            model,
            validation_graphs,
        )

        validation_prediction = (
            validation_prediction_z * age_std + age_mean
        )

        validation_age = np.asarray(
            [float(graphs[i].y.item()) for i in validation_indices],
            dtype=float,
        )

        validation_mae = mean_absolute_error(
            validation_age,
            validation_prediction,
        )

        if validation_mae < best_validation_mae:
            best_validation_mae = float(validation_mae)
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    if best_model_state is None:
        raise RuntimeError("No best model state was recorded.")

    model.load_state_dict(best_model_state)

    return model, age_mean, age_std, best_epoch


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics for valid predictions."""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        raise ValueError("No valid predictions available.")

    if (
        len(y_true) > 1
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):
        pearson_r = float(pearsonr(y_true, y_pred).statistic)
    else:
        pearson_r = float("nan")

    return {
        "n_CN_OOF_scans": int(len(y_true)),
        "MAE_years": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_years": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": pearson_r,
    }


def get_threshold_label(retain_percent):
    if retain_percent is None:
        return "No threshold (all positive edges)"

    return f"Top {retain_percent}% of positive edges"


# ============================================================
# CROSS-VALIDATION FOR ONE THRESHOLD
# ============================================================

def run_cross_validation(
    graphs,
    cn_indices,
    cn_groups,
    n_splits,
    retain_percent,
):
    """Generate participant-grouped CN OOF predictions."""
    oof_predictions = np.full(len(graphs), np.nan, dtype=float)
    best_epochs = []
    fold_rows = []

    outer_splitter = GroupKFold(n_splits=n_splits)

    for fold, (train_local, test_local) in enumerate(
        outer_splitter.split(cn_indices, groups=cn_groups),
        start=1,
    ):
        outer_train_indices = cn_indices[train_local]
        outer_test_indices = cn_indices[test_local]
        inner_groups = cn_groups[train_local]

        # Hold out 20% of outer-training participants for early stopping.
        inner_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=SEED + fold,
        )

        inner_train_local, inner_validation_local = next(
            inner_splitter.split(
                outer_train_indices,
                groups=inner_groups,
            )
        )

        train_indices = outer_train_indices[inner_train_local]
        validation_indices = outer_train_indices[
            inner_validation_local
        ]

        model, age_mean, age_std, best_epoch = fit_one_fold(
            graphs=graphs,
            train_indices=train_indices,
            validation_indices=validation_indices,
            seed=SEED + fold,
        )

        test_graphs = [graphs[i] for i in outer_test_indices]
        test_prediction_z = predict_standardized(model, test_graphs)

        oof_predictions[outer_test_indices] = (
            test_prediction_z * age_std + age_mean
        )

        best_epochs.append(best_epoch)

        fold_age = np.asarray(
            [float(graphs[i].y.item()) for i in outer_test_indices],
            dtype=float,
        )

        fold_prediction = oof_predictions[outer_test_indices]
        fold_metrics = calculate_metrics(fold_age, fold_prediction)

        mean_fold_edges = float(
            np.mean([
                graphs[i].edge_index.shape[1] / 2
                for i in outer_test_indices
            ])
        )

        fold_rows.append({
            "threshold": get_threshold_label(retain_percent),
            "retain_positive_percent": (
                "all" if retain_percent is None else retain_percent
            ),
            "fold": fold,
            **fold_metrics,
            "best_epoch": int(best_epoch),
            "mean_undirected_edges": mean_fold_edges,
        })

        print(
            f"{get_threshold_label(retain_percent)} | "
            f"fold {fold}/{n_splits} | "
            f"MAE = {fold_metrics['MAE_years']:.3f} years | "
            f"selected epoch = {best_epoch}"
        )

    cn_age = np.asarray(
        [float(graphs[i].y.item()) for i in cn_indices],
        dtype=float,
    )

    cn_oof_prediction = oof_predictions[cn_indices]
    pooled_metrics = calculate_metrics(cn_age, cn_oof_prediction)
    fold_table = pd.DataFrame(fold_rows)

    return {
        "oof_predictions": oof_predictions,
        "best_epochs": best_epochs,
        "fold_table": fold_table,
        "pooled_metrics": pooled_metrics,
        "mean_fold_mae": float(fold_table["MAE_years"].mean()),
        "sd_fold_mae": float(fold_table["MAE_years"].std(ddof=1)),
        "mean_fold_rmse": float(fold_table["RMSE_years"].mean()),
    }


# ============================================================
# FINAL MODEL FOR SELECTED THRESHOLD
# ============================================================

def train_final_model(graphs, cn_indices, best_epochs, seed):
    """Train the selected-threshold model on all CN scans."""
    set_seed(seed)

    final_epochs = max(1, int(np.median(best_epochs)))

    cn_ages = np.asarray(
        [float(graphs[i].y.item()) for i in cn_indices],
        dtype=float,
    )

    age_mean = float(cn_ages.mean())
    age_std = max(float(cn_ages.std()), 1e-8)

    cn_graphs = scale_age_targets(
        cn_indices,
        graphs,
        age_mean,
        age_std,
    )

    train_loader = DataLoader(
        cn_graphs,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = FmriGNN(
        node_dim=graphs[0].x.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        f"\nTraining final model for {final_epochs} epochs."
    )

    for epoch in range(1, final_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer)

        if epoch == 1 or epoch % 10 == 0 or epoch == final_epochs:
            print(
                f"Epoch {epoch}/{final_epochs}, "
                f"training loss = {loss:.4f}"
            )

    return model, age_mean, age_std, final_epochs


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    set_seed(SEED)

    (
        aligned,
        dataset_col,
        id_col,
        connectome_path_col,
        metadata_path,
    ) = load_inputs()

    # Load each matrix once and reuse it for all thresholds.
    matrices = []
    valid_row_indices = []
    failed_rows = []

    for row_index, row in aligned.iterrows():
        path = Path(str(row[connectome_path_col]))

        try:
            if not path.exists():
                raise FileNotFoundError(path)

            matrices.append(load_matrix(path))
            valid_row_indices.append(row_index)

        except Exception as exc:
            failed_rows.append({
                "row": int(row_index),
                "path": str(path),
                "reason": str(exc),
            })

    aligned = aligned.loc[valid_row_indices].reset_index(drop=True)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(
            OUTPUT_DIR / "graph_build_failures.csv",
            index=False,
        )

    if len(matrices) == 0:
        raise RuntimeError("No valid connectome matrices were loaded.")

    if len(matrices) != len(aligned):
        raise RuntimeError("Matrix and metadata alignment error.")

    cn_indices = np.flatnonzero(aligned["_is_cn"].to_numpy())

    cn_groups = aligned.loc[
        cn_indices,
        "_participant_group",
    ].to_numpy()

    n_unique_cn_participants = len(np.unique(cn_groups))
    n_splits = min(N_SPLITS, n_unique_cn_participants)

    if n_splits < 2:
        raise ValueError(
            "At least two unique CN participants are required."
        )

    # Persist folds so comparisons can verify identical participant splits.
    fold_assignments = aligned[[
        "_manifest_dataset", "_manifest_id", "_participant_group", "_age", "_is_cn"
    ]].copy()
    fold_assignments["FOLD"] = -1
    for fold, (_, test_local) in enumerate(
        GroupKFold(n_splits=n_splits).split(cn_indices, groups=cn_groups), 1
    ):
        fold_assignments.loc[cn_indices[test_local], "FOLD"] = fold
    fold_assignments.to_csv(
        OUTPUT_DIR / "fMRI_outer_fold_assignments.csv", index=False
    )

    print(f"\nUnique CN participants: {n_unique_cn_participants}")
    print(f"Outer CV folds: {n_splits}")
    print(f"Threshold candidates: {RETAIN_POSITIVE_PERCENT}")

    threshold_results = {}
    comparison_rows = []
    all_fold_tables = []

    # Compare candidate thresholds using the same CV design.
    for retain_percent in RETAIN_POSITIVE_PERCENT:
        print("\n" + "=" * 78)
        print("Evaluating:", get_threshold_label(retain_percent))
        print("=" * 78)

        graphs = []
        undirected_edge_counts = []
        numeric_cutoffs = []
        ages = aligned["_age"].to_numpy(dtype=float)

        for matrix, age in zip(matrices, ages):
            graph, edge_count, cutoff = matrix_to_graph(
                matrix=matrix,
                age=age,
                retain_percent=retain_percent,
            )

            graphs.append(graph)
            undirected_edge_counts.append(edge_count)
            numeric_cutoffs.append(cutoff)

        cv_result = run_cross_validation(
            graphs=graphs,
            cn_indices=cn_indices,
            cn_groups=cn_groups,
            n_splits=n_splits,
            retain_percent=retain_percent,
        )

        key = (
            "all_positive"
            if retain_percent is None
            else f"top_{retain_percent}_percent"
        )

        threshold_results[key] = {
            "graphs": graphs,
            "edge_counts": undirected_edge_counts,
            "numeric_cutoffs": numeric_cutoffs,
            **cv_result,
        }

        pooled = cv_result["pooled_metrics"]

        comparison_rows.append({
            "threshold": get_threshold_label(retain_percent),
            "retain_positive_percent": (
                "all" if retain_percent is None else retain_percent
            ),
            "mean_undirected_edges_per_scan": float(
                np.mean(undirected_edge_counts)
            ),
            "sd_undirected_edges_per_scan": float(
                np.std(undirected_edge_counts, ddof=1)
            ),
            "mean_numeric_cutoff": float(np.mean(numeric_cutoffs)),
            **pooled,
            "mean_fold_MAE_years": cv_result["mean_fold_mae"],
            "sd_fold_MAE_years": cv_result["sd_fold_mae"],
            "mean_fold_RMSE_years": cv_result["mean_fold_rmse"],
            "median_selected_epoch": int(
                np.median(cv_result["best_epochs"])
            ),
        })

        all_fold_tables.append(cv_result["fold_table"])

    # Sort by pooled OOF MAE, using RMSE as tie-breaker.
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["MAE_years", "RMSE_years"],
        ascending=[True, True],
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "rank_by_CN_OOF_MAE",
        np.arange(1, len(comparison) + 1),
    )

    comparison.to_csv(
        OUTPUT_DIR / "fMRI_GNN_threshold_comparison.csv",
        index=False,
    )

    pd.concat(
        all_fold_tables,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "fMRI_GNN_threshold_comparison_by_fold.csv",
        index=False,
    )

    report_columns = [
        "rank_by_CN_OOF_MAE",
        "threshold",
        "mean_undirected_edges_per_scan",
        "MAE_years",
        "RMSE_years",
        "R2",
        "Pearson_r",
        "n_CN_OOF_scans",
    ]

    report_table = comparison[report_columns].copy()

    for column in [
        "mean_undirected_edges_per_scan",
        "MAE_years",
        "RMSE_years",
        "R2",
        "Pearson_r",
    ]:
        report_table[column] = report_table[column].map(
            lambda value: f"{value:.3f}"
        )

    markdown_lines = [
        "| " + " | ".join(report_table.columns) + " |",
        "|" + "|".join(["---"] * len(report_table.columns)) + "|",
    ]

    for row in report_table.itertuples(index=False, name=None):
        markdown_lines.append(
            "| " + " | ".join(map(str, row)) + " |"
        )

    (
        OUTPUT_DIR / "fMRI_GNN_threshold_comparison_table.md"
    ).write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )

    best_row = comparison.iloc[0]
    best_threshold_label = best_row["threshold"]
    best_key = None

    for retain_percent in RETAIN_POSITIVE_PERCENT:
        if get_threshold_label(retain_percent) == best_threshold_label:
            best_key = (
                "all_positive"
                if retain_percent is None
                else f"top_{retain_percent}_percent"
            )
            break

    if best_key is None:
        raise RuntimeError("Could not identify the selected threshold.")

    best_result = threshold_results[best_key]

    print("\nThreshold comparison:")
    print(comparison[report_columns].to_string(index=False))
    print("\nSelected threshold:", best_threshold_label)

    # Plot threshold MAE comparison.
    plot_data = comparison.sort_values(
        "MAE_years",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(10, 5.5))
    x_positions = np.arange(len(plot_data))

    axis.bar(
        x_positions,
        plot_data["MAE_years"],
        color="#4C78A8",
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        plot_data["threshold"],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("CN pooled out-of-fold MAE (years)")
    axis.set_title("fMRI GNN Threshold Comparison")
    axis.grid(axis="y", alpha=0.25)

    for x, value in enumerate(plot_data["MAE_years"]):
        axis.text(
            x,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "figure_threshold_comparison_MAE.png",
        dpi=300,
    )
    plt.close(figure)

    # Train the final selected-threshold model.
    selected_graphs = best_result["graphs"]

    final_model, age_mean, age_std, final_epochs = train_final_model(
        graphs=selected_graphs,
        cn_indices=cn_indices,
        best_epochs=best_result["best_epochs"],
        seed=SEED + 10000,
    )

    all_graphs = [copy.deepcopy(graph) for graph in selected_graphs]

    predicted_age_z = predict_standardized(final_model, all_graphs)
    predicted_brain_age = predicted_age_z * age_std + age_mean
    chronological_age = aligned["_age"].to_numpy(dtype=float)
    bag = predicted_brain_age - chronological_age

    # Correct age bias using CN OOF BAG.
    oof_predictions = best_result["oof_predictions"]

    oof_cn_indices = cn_indices[
        np.isfinite(oof_predictions[cn_indices])
    ]

    oof_age = np.asarray(
        [float(selected_graphs[i].y.item()) for i in oof_cn_indices],
        dtype=float,
    )

    oof_bag = oof_predictions[oof_cn_indices] - oof_age

    if len(oof_age) >= 2 and np.std(oof_age) > 0:
        bias_slope, bias_intercept = np.polyfit(
            oof_age,
            oof_bag,
            deg=1,
        )
    else:
        bias_slope = 0.0
        bias_intercept = 0.0

    cbag = bag - (
        bias_slope * chronological_age
        + bias_intercept
    )

    aligned["BRAIN_AGE_GNN_fMRI"] = predicted_brain_age
    aligned["BAG_GNN_fMRI"] = bag
    aligned["cBAG_GNN_fMRI"] = cbag
    aligned["BRAIN_AGE_GNN_fMRI_OOF"] = oof_predictions

    aligned.to_csv(
        OUTPUT_DIR / "fMRI_GNN_BAG_all_predictions.csv",
        index=False,
    )

    # Record the same participant-grouped outer folds used by Ridge.
    aligned["FOLD"] = -1
    for fold, (_, test_local) in enumerate(
        GroupKFold(n_splits=n_splits).split(cn_indices, groups=cn_groups), 1
    ):
        aligned.loc[cn_indices[test_local], "FOLD"] = fold

    # CN OOF prediction table.
    oof_table = aligned.loc[
        cn_indices,
        [
            dataset_col,
            id_col,
            "_participant_group",
            "_age",
            "BRAIN_AGE_GNN_fMRI_OOF",
            "FOLD",
        ],
    ].copy()

    oof_table["BAG_OOF"] = (
        oof_table["BRAIN_AGE_GNN_fMRI_OOF"]
        - oof_table["_age"]
    )

    oof_table.to_csv(
        OUTPUT_DIR / "fMRI_GNN_BAG_CN_OOF_predictions.csv",
        index=False,
    )

    # Per-dataset and pooled CN OOF metrics.
    valid_oof = (
        np.isfinite(oof_table["_age"].to_numpy(dtype=float))
        & np.isfinite(
            oof_table["BRAIN_AGE_GNN_fMRI_OOF"].to_numpy(dtype=float)
        )
    )

    oof_valid = oof_table.loc[valid_oof].copy()
    dataset_metric_rows = []

    for dataset in sorted(aligned[dataset_col].dropna().unique()):
        dataset_rows = oof_valid.loc[
            oof_valid[dataset_col] == dataset
        ]

        y_true = dataset_rows["_age"].to_numpy(dtype=float)
        y_pred = dataset_rows[
            "BRAIN_AGE_GNN_fMRI_OOF"
        ].to_numpy(dtype=float)

        n_scans = len(y_true)

        if n_scans == 0:
            mae = rmse = r2 = pearson_r_value = np.nan
        else:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            r2 = (
                r2_score(y_true, y_pred)
                if n_scans >= 2 and np.std(y_true) > 0
                else np.nan
            )

            pearson_r_value = (
                pearsonr(y_true, y_pred).statistic
                if n_scans >= 2
                and np.std(y_true) > 0
                and np.std(y_pred) > 0
                else np.nan
            )

        if n_scans == 0:
            pearson_r_value = np.nan

        dataset_metric_rows.append({
            "evaluation": "by_dataset",
            "DATASET": dataset,
            "n_CN_OOF_scans": n_scans,
            "n_unique_CN_participants": (
                dataset_rows["_participant_group"].nunique()
            ),
            "MAE_years": mae,
            "RMSE_years": rmse,
            "R2": r2,
            "Pearson_r": pearson_r_value,
        })

    # Pooled metrics across all datasets.
    y_true_pooled = oof_valid["_age"].to_numpy(dtype=float)
    y_pred_pooled = oof_valid[
        "BRAIN_AGE_GNN_fMRI_OOF"
    ].to_numpy(dtype=float)

    pooled_metrics = {
        "evaluation": "pooled",
        "DATASET": "ALL",
        "n_CN_OOF_scans": len(y_true_pooled),
        "n_unique_CN_participants": (
            oof_valid["_participant_group"].nunique()
        ),
        "MAE_years": mean_absolute_error(
            y_true_pooled,
            y_pred_pooled,
        ),
        "RMSE_years": np.sqrt(
            mean_squared_error(
                y_true_pooled,
                y_pred_pooled,
            )
        ),
        "R2": (
            r2_score(y_true_pooled, y_pred_pooled)
            if len(y_true_pooled) >= 2
            and np.std(y_true_pooled) > 0
            else np.nan
        ),
        "Pearson_r": (
            pearsonr(y_true_pooled, y_pred_pooled).statistic
            if len(y_true_pooled) >= 2
            and np.std(y_true_pooled) > 0
            and np.std(y_pred_pooled) > 0
            else np.nan
        ),
    }

    metrics_by_dataset_and_pooled = pd.concat(
        [
            pd.DataFrame(dataset_metric_rows),
            pd.DataFrame([pooled_metrics]),
        ],
        ignore_index=True,
    )

    metrics_by_dataset_and_pooled.to_csv(
        OUTPUT_DIR
        / "fMRI_GNN_selected_threshold_metrics_by_dataset_and_pooled.csv",
        index=False,
    )

    print(
        "\nSelected-threshold CN OOF metrics by dataset and pooled:"
    )
    print(metrics_by_dataset_and_pooled.to_string(index=False))

    # Save selected-threshold summary.
    summary = {
        "selected_threshold": best_threshold_label,
        "selection_rule": (
            "Lowest pooled CN OOF MAE; RMSE used as tie-breaker"
        ),
        "threshold_comparison_csv": str(
            OUTPUT_DIR / "fMRI_GNN_threshold_comparison.csv"
        ),
        "threshold_comparison_table": str(
            OUTPUT_DIR / "fMRI_GNN_threshold_comparison_table.md"
        ),
        "n_CN_OOF_scans": int(
            best_result["pooled_metrics"]["n_CN_OOF_scans"]
        ),
        "n_unique_CN_participants": int(
            n_unique_cn_participants
        ),
        "n_folds": int(n_splits),
        "selected_threshold_CN_OOF_metrics": (
            best_result["pooled_metrics"]
        ),
        "dataset_CN_overrides": DATASET_CN_OVERRIDES,
        "fold_best_epochs": [
            int(epoch)
            for epoch in best_result["best_epochs"]
        ],
        "final_epochs": int(final_epochs),
        "age_mean_used_by_final_model": float(age_mean),
        "age_std_used_by_final_model": float(age_std),
        "cBAG_bias_slope": float(bias_slope),
        "cBAG_bias_intercept": float(bias_intercept),
        "device": str(DEVICE),
        "manifest": str(MANIFEST_PATH),
        "metadata": str(metadata_path),
        "output_dir": str(OUTPUT_DIR),
        "interpretation_note": (
            "Thresholding changes graph edge topology. The node feature "
            "matrix remains the full signed connectivity profile."
        ),
        "harmonization_warning": (
            "ComBat was fit upstream on the full dataset and preserved age; "
            "CV metrics may contain harmonization leakage."
        ),
        "model_selection_warning": (
            "The same CV comparison selects the threshold and reports its "
            "performance. Treat this as a model-selection estimate; nested "
            "CV or independent test data is needed for an unbiased estimate."
        ),
    }

    with open(
        OUTPUT_DIR / "fMRI_GNN_threshold_selection_summary.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(summary, output_file, indent=2)

    selected_retain_percent = (
        None
        if best_key == "all_positive"
        else int(
            best_key.replace("top_", "").replace("_percent", "")
        )
    )

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "age_mean": age_mean,
            "age_std": age_std,
            "selected_threshold_label": best_threshold_label,
            "selected_retain_positive_percent": selected_retain_percent,
            "final_epochs": int(final_epochs),
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "bias_slope": float(bias_slope),
            "bias_intercept": float(bias_intercept),
        },
        OUTPUT_DIR / "fMRI_GNN_BAG_best_threshold_model.pt",
    )

    print("\nAll outputs saved under:")
    print(OUTPUT_DIR)

    # ============================================================
    # All-scan prediction metrics: by dataset + pooled
    # ============================================================

    all_scan_table = aligned.copy()
    all_scan_table["_true_age"] = aligned["_age"].to_numpy(dtype=float)
    all_scan_table["_predicted_age"] = np.asarray(
        predicted_brain_age, dtype=float
    )

    # Keep scans with valid age and prediction
    valid_mask = (
        np.isfinite(all_scan_table["_true_age"].to_numpy(dtype=float))
        & np.isfinite(all_scan_table["_predicted_age"].to_numpy(dtype=float))
    )
    all_scan_valid = all_scan_table.loc[valid_mask].copy()

    metric_rows = []

    # Metrics for each dataset
    for dataset in sorted(all_scan_valid["_manifest_dataset"].dropna().unique()):
        rows = all_scan_valid.loc[
            all_scan_valid["_manifest_dataset"] == dataset
        ]
        y_true = rows["_true_age"].to_numpy(dtype=float)
        y_pred = rows["_predicted_age"].to_numpy(dtype=float)
        n = len(y_true)

        metric_rows.append({
            "evaluation": "all_scans_by_dataset",
            "DATASET": dataset,
            "n_scans": n,
            "n_unique_participants": (
                rows["_participant_group"].nunique()
                if "_participant_group" in rows.columns else np.nan
            ),
            "MAE_years": mean_absolute_error(y_true, y_pred),
            "RMSE_years": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": (
                r2_score(y_true, y_pred)
                if n >= 2 and np.std(y_true) > 0 else np.nan
            ),
            "Pearson_r": (
                pearsonr(y_true, y_pred).statistic
                if n >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0
                else np.nan
            ),
        })

    # Pooled metrics across all datasets
    y_true = all_scan_valid["_true_age"].to_numpy(dtype=float)
    y_pred = all_scan_valid["_predicted_age"].to_numpy(dtype=float)
    n = len(y_true)

    metric_rows.append({
        "evaluation": "all_scans_pooled",
        "DATASET": "ALL",
        "n_scans": n,
        "n_unique_participants": (
            all_scan_valid["_participant_group"].nunique()
            if "_participant_group" in all_scan_valid.columns else np.nan
        ),
        "MAE_years": mean_absolute_error(y_true, y_pred),
        "RMSE_years": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": (
            r2_score(y_true, y_pred)
            if n >= 2 and np.std(y_true) > 0 else np.nan
        ),
        "Pearson_r": (
            pearsonr(y_true, y_pred).statistic
            if n >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0
            else np.nan
        ),
    })

    all_scan_metrics = pd.DataFrame(metric_rows)

    # Save CSV
    metrics_path = OUTPUT_DIR / "fMRI_GNN_all_scan_metrics_by_dataset_and_pooled.csv"
    all_scan_metrics.to_csv(metrics_path, index=False)

    print("\nAll-scan metrics by dataset and pooled:")
    print(all_scan_metrics.to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
