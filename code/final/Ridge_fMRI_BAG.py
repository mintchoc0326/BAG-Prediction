#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fMRI connectome brain-age prediction using Ridge regression.

Input:
    Harmonized 84 x 84 fMRI connectomes listed in the manifest.
    Metadata containing dataset, connectome ID, age, and diagnosis.

Training/evaluation:
    Train on CN scans only.
    Use participant-grouped five-fold outer CV.
    Select Ridge alpha using grouped CV inside each outer training fold.
    Use a StandardScaler fitted only on each training split.

Outputs:
    Ridge_BAG/fMRI_Ridge_CN_OOF_predictions.csv
    Ridge_BAG/fMRI_Ridge_full_cohort_predictions.csv
    Ridge_BAG/fMRI_Ridge_CV_metrics.json
    Ridge_BAG/fMRI_Ridge_fold_metrics.csv
    Ridge_BAG/fMRI_Ridge_BAG_summary_by_dataset.csv
    Ridge_BAG/figure_Ridge_CN_OOF_predicted_vs_actual.png
    Ridge_BAG/figure_Ridge_CN_OOF_BAG_vs_age.png
    Ridge_BAG/figure_Ridge_full_cohort_BAG_by_dataset.png
    Ridge_BAG/fMRI_Ridge_final_model.joblib

Important:
    This code fits StandardScaler and tunes alpha inside the CV folds.
    However, if the harmonized connectomes were produced by fitting ComBat
    once on the entire dataset while preserving age, that upstream step may
    still leak held-out age information into the connectomes. For a strict
    prediction evaluation, ComBat must also be handled within each fold.
"""

import copy
import json
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from fmri_bag_cv import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

OUTPUT_DIR = PROJECT_DIR / "Ridge_BAG"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4

# fMRI HABS has explicit CN/MCI/AD diagnoses: use those labels.
# Do not apply the DWI whole-cohort HABS override to this population.
DATASET_CN_OVERRIDES = {}

# Ridge regularization values to evaluate.
# StandardScaler is fitted separately inside each CV training split.
ALPHAS = np.logspace(-3, 4, 15)


# ============================================================
# HELPERS: IDs, COLUMNS, DIAGNOSIS
# ============================================================

def find_column(dataframe, candidates, label):
    """Find a dataframe column without case sensitivity."""
    lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise KeyError(
        f"Could not find {label}. "
        f"Tried: {candidates}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def normalize_id(value):
    """Normalize IDs so manifest and metadata keys can be matched."""
    if pd.isna(value):
        return ""

    return re.sub(
        r"\.0$",
        "",
        str(value).strip().upper(),
    )


def is_cn(value):
    """Return True for common cognitively normal/control labels."""
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
    """
    Group repeated visits from the same participant together.

    Example:
        R4288_y0 and R4288_y4 are assigned to the same group.
    """
    base_id = re.sub(
        r"_Y\d+$",
        "",
        normalize_id(scan_id),
    )

    return (
        f"{str(dataset).strip().upper()}__"
        f"{base_id}"
    )


# ============================================================
# LOAD AND ALIGN MANIFEST AND METADATA
# ============================================================

def load_aligned_metadata():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    metadata_files = sorted(
        METADATA_SEARCH_DIR.rglob(METADATA_FILENAME)
    )

    if not metadata_files:
        raise FileNotFoundError(
            f"Could not find {METADATA_FILENAME} "
            f"under {METADATA_SEARCH_DIR}"
        )

    if len(metadata_files) > 1:
        print(
            "Multiple metadata files found. Using:",
            metadata_files[0],
        )

    metadata_path = metadata_files[0]

    manifest = pd.read_csv(MANIFEST_PATH)
    metadata = pd.read_csv(
        metadata_path,
        low_memory=False,
    )

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

    diagnosis_col = find_column(
        metadata,
        [
            "DIAGNOSIS_BAG",
            "DIAGNOSIS",
            "diagnosis",
            "GROUP",
            "group",
            "group_status",
            "NORMCOG",
        ],
        "diagnosis/control status",
    )

    manifest["_dataset_key"] = (
        manifest[manifest_dataset_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    manifest["_id_key"] = manifest[
        manifest_id_col
    ].map(normalize_id)

    # Preserve original manifest identifiers before merging.
    manifest["_manifest_dataset"] = manifest[
        manifest_dataset_col
    ]

    manifest["_manifest_id"] = manifest[
        manifest_id_col
    ]

    metadata["_dataset_key"] = (
        metadata[metadata_dataset_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    metadata["_id_key"] = metadata[
        metadata_id_col
    ].map(normalize_id)

    metadata["_age"] = pd.to_numeric(
        metadata[age_col],
        errors="coerce",
    )

    metadata["_is_cn"] = metadata[
        diagnosis_col
    ].map(is_cn)

    # Keep one metadata record per dataset and connectome ID.
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

    # Apply cohort overrides before selecting CN scans for training and CV.
    dataset_labels = (
        aligned["_manifest_dataset"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    for dataset, is_cn_override in DATASET_CN_OVERRIDES.items():
        dataset_mask = dataset_labels == dataset.upper()
        aligned.loc[dataset_mask, "_is_cn"] = is_cn_override

    aligned["_participant_group"] = [
        make_participant_group(dataset, scan_id)
        for dataset, scan_id in zip(
            aligned["_manifest_dataset"],
            aligned["_manifest_id"],
        )
    ]

    print(f"Matched scans with age: {len(aligned)}")
    print(f"CN scans: {int(aligned['_is_cn'].sum())}")
    print("\nDiagnosis labels by dataset:")
    print(
        aligned.groupby("_manifest_dataset")[diagnosis_col]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nCN labels recognized by dataset:")
    print(
        aligned.groupby("_manifest_dataset")["_is_cn"]
        .sum()
        .to_string()
    )
    print("Scans by dataset:")
    print(
        aligned["_manifest_dataset"]
        .value_counts()
        .to_string()
    )

    return (
        aligned,
        connectome_path_col,
        metadata_path,
    )


# ============================================================
# LOAD CONNECTOMES AND CREATE RIDGE FEATURES
# ============================================================

def load_connectome_features(path):
    """
    Load a harmonized 84 x 84 matrix and vectorize its unique edges.

    The upper triangle excluding the diagonal contains 3,486 unique edges.
    Signed fMRI connections are retained to match the fMRI GNN node features.
    """
    matrix = pd.read_csv(
        path,
        header=None,
    ).to_numpy(dtype=np.float32)

    if matrix.shape != (84, 84):
        raise ValueError(
            f"Expected an 84x84 matrix, got "
            f"{matrix.shape}: {path}"
        )

    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Ensure symmetry and remove self-connections.
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)

    # Retain negative fMRI correlations in the edge feature vector.

    upper_i, upper_j = np.triu_indices(
        84,
        k=1,
    )

    features = matrix[upper_i, upper_j]

    if len(features) != 3486:
        raise RuntimeError(
            f"Expected 3,486 edge features, got {len(features)}."
        )

    return features.astype(np.float64)


def load_feature_matrix(aligned, connectome_path_col):
    """Load all valid connectomes and keep metadata aligned with features."""
    feature_rows = []
    valid_row_indices = []
    failures = []

    for row_index, row in aligned.iterrows():
        path = Path(
            str(row[connectome_path_col])
        )

        try:
            if not path.exists():
                raise FileNotFoundError(path)

            feature_rows.append(
                load_connectome_features(path)
            )

            valid_row_indices.append(row_index)

        except Exception as error:
            failures.append(
                {
                    "row": int(row_index),
                    "path": str(path),
                    "reason": str(error),
                }
            )

    aligned = aligned.loc[
        valid_row_indices
    ].reset_index(drop=True)

    if failures:
        pd.DataFrame(failures).to_csv(
            OUTPUT_DIR / "fMRI_Ridge_invalid_connectomes.csv",
            index=False,
        )

    if not feature_rows:
        raise RuntimeError(
            "No valid connectome matrices were loaded."
        )

    X = np.vstack(feature_rows)

    if X.shape[0] != len(aligned):
        raise RuntimeError(
            "Feature and metadata rows are not aligned."
        )

    print(f"Valid scans: {X.shape[0]}")
    print(f"Ridge features per scan: {X.shape[1]}")

    return X, aligned


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics for a set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        raise ValueError(
            "No valid predictions available."
        )

    if (
        len(y_true) > 1
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):
        pearson_r, pearson_p = pearsonr(
            y_true,
            y_pred,
        )
        pearson_r = float(pearson_r)
        pearson_p = float(pearson_p)
    else:
        pearson_r = float("nan")
        pearson_p = float("nan")

    return {
        "n_scans": int(len(y_true)),
        "MAE_years": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE_years": float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),
        "R2": float(
            r2_score(y_true, y_pred)
        ),
        "Pearson_r": pearson_r,
        "Pearson_p": pearson_p,
    }


# ============================================================
# NESTED PARTICIPANT-GROUPED CROSS-VALIDATION
# ============================================================

def main():
    (
        aligned,
        connectome_path_col,
        metadata_path,
    ) = load_aligned_metadata()

    X, aligned = load_feature_matrix(
        aligned,
        connectome_path_col,
    )

    cn_indices = np.flatnonzero(
        aligned["_is_cn"].to_numpy()
    )

    if len(cn_indices) < 2:
        raise ValueError(
            "At least two CN scans are required."
        )

    y_all = aligned["_age"].to_numpy(
        dtype=float
    )

    cn_groups = aligned.loc[
        cn_indices,
        "_participant_group",
    ].to_numpy()

    n_unique_cn_participants = len(
        np.unique(cn_groups)
    )

    n_outer_folds = min(
        N_OUTER_FOLDS,
        n_unique_cn_participants,
    )

    if n_outer_folds < 2:
        raise ValueError(
            "At least two unique CN participants are required."
        )

    # Persist folds so comparisons can verify identical participant splits.
    fold_assignments = aligned[[
        "_manifest_dataset", "_manifest_id", "_participant_group", "_age", "_is_cn"
    ]].copy()
    fold_assignments["FOLD"] = -1
    for fold, (_, test_local) in enumerate(
        GroupKFold(n_splits=n_outer_folds).split(cn_indices, groups=cn_groups), 1
    ):
        fold_assignments.loc[cn_indices[test_local], "FOLD"] = fold
    fold_assignments.to_csv(
        OUTPUT_DIR / "fMRI_outer_fold_assignments.csv", index=False
    )

    print(f"\nCN scans: {len(cn_indices)}")
    print(
        "Unique CN participants:",
        n_unique_cn_participants,
    )
    print(
        "Outer participant-grouped folds:",
        n_outer_folds,
    )

    oof_prediction = np.full(
        len(aligned),
        np.nan,
        dtype=float,
    )

    oof_fold = np.full(
        len(aligned),
        -1,
        dtype=int,
    )

    fold_rows = []
    best_alphas = []

    outer_cv = GroupKFold(
        n_splits=n_outer_folds
    )

    for fold, (train_local, test_local) in enumerate(
        outer_cv.split(
            X[cn_indices],
            y_all[cn_indices],
            groups=cn_groups,
        ),
        start=1,
    ):
        outer_train_indices = cn_indices[
            train_local
        ]

        outer_test_indices = cn_indices[
            test_local
        ]

        X_train = X[
            outer_train_indices
        ]

        y_train = y_all[
            outer_train_indices
        ]

        X_test = X[
            outer_test_indices
        ]

        y_test = y_all[
            outer_test_indices
        ]

        train_groups = aligned.loc[
            outer_train_indices,
            "_participant_group",
        ].to_numpy()

        n_inner_folds = min(
            N_INNER_FOLDS,
            len(np.unique(train_groups)),
        )

        if n_inner_folds < 2:
            raise ValueError(
                f"Not enough unique participants for inner CV "
                f"in outer fold {fold}."
            )

        inner_cv = GroupKFold(
            n_splits=n_inner_folds
        )

        model_pipeline = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "ridge",
                    Ridge(),
                ),
            ]
        )

        parameter_grid = {
            "ridge__alpha": ALPHAS,
        }

        search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=parameter_grid,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )

        # Groups are used by inner GroupKFold so repeat visits
        # from one participant remain in the same split.
        search.fit(
            X_train,
            y_train,
            groups=train_groups,
        )

        fold_prediction = search.best_estimator_.predict(
            X_test
        )

        oof_prediction[
            outer_test_indices
        ] = fold_prediction

        oof_fold[
            outer_test_indices
        ] = fold

        fold_metrics = calculate_metrics(
            y_test,
            fold_prediction,
        )

        best_alpha = float(
            search.best_params_["ridge__alpha"]
        )

        best_alphas.append(best_alpha)

        fold_rows.append(
            {
                "fold": fold,
                **fold_metrics,
                "best_alpha": best_alpha,
                "n_unique_train_participants": int(
                    len(np.unique(train_groups))
                ),
                "n_unique_test_participants": int(
                    len(
                        np.unique(
                            aligned.loc[
                                outer_test_indices,
                                "_participant_group",
                            ]
                        )
                    )
                ),
            }
        )

        print(
            f"Fold {fold}/{n_outer_folds}: "
            f"MAE = {fold_metrics['MAE_years']:.3f} years; "
            f"RMSE = {fold_metrics['RMSE_years']:.3f} years; "
            f"alpha = {best_alpha:g}"
        )

    # Pooled OOF metrics: each CN scan has one prediction
    # from a model that did not train on that participant.
    cn_y_true = y_all[
        cn_indices
    ]

    cn_oof_pred = oof_prediction[
        cn_indices
    ]

    pooled_metrics = calculate_metrics(
        cn_y_true,
        cn_oof_pred,
    )

    fold_table = pd.DataFrame(
        fold_rows
    )

    fold_table.to_csv(
        OUTPUT_DIR / "fMRI_Ridge_fold_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save CN OOF predictions and BAG.
    # --------------------------------------------------------
    oof_table = aligned.loc[
        cn_indices,
        [
            "_manifest_dataset",
            "_manifest_id",
            "_participant_group",
            "_age",
        ],
    ].copy()

    oof_table = oof_table.rename(
        columns={
            "_manifest_dataset": "DATASET",
            "_manifest_id": "CONNECTOME_ID",
            "_participant_group": "PARTICIPANT_GROUP",
            "_age": "CHRONOLOGICAL_AGE",
        }
    )

    oof_table["PREDICTED_AGE_OOF"] = cn_oof_pred
    oof_table["BAG_OOF"] = (
        oof_table["PREDICTED_AGE_OOF"]
        - oof_table["CHRONOLOGICAL_AGE"]
    )
    oof_table["FOLD"] = oof_fold[
        cn_indices
    ]

    oof_table.to_csv(
        OUTPUT_DIR / "fMRI_Ridge_CN_OOF_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Train final Ridge model using all CN scans.
    # The final alpha is the median alpha selected across outer folds.
    # --------------------------------------------------------
    final_alpha = float(
        np.median(best_alphas)
    )

    final_model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(alpha=final_alpha),
            ),
        ]
    )

    final_model.fit(
        X[cn_indices],
        y_all[cn_indices],
    )

    # Predict all valid scans, including non-CN participants.
    full_cohort_prediction = final_model.predict(X)

    full_cohort_bag = (
        full_cohort_prediction
        - y_all
    )

    # Fit age-bias correction using CN OOF BAG only.
    oof_age = y_all[
        cn_indices
    ]

    oof_bag = (
        cn_oof_pred
        - oof_age
    )

    if (
        len(oof_age) >= 2
        and np.std(oof_age) > 0
    ):
        bias_slope, bias_intercept = np.polyfit(
            oof_age,
            oof_bag,
            deg=1,
        )
    else:
        bias_slope = 0.0
        bias_intercept = 0.0

    full_cohort_cbag = full_cohort_bag - (
        bias_slope * y_all
        + bias_intercept
    )

    full_table = aligned.copy()

    full_table["PREDICTED_AGE_RIDGE"] = (
        full_cohort_prediction
    )

    full_table["BAG_RIDGE"] = (
        full_cohort_bag
    )

    full_table["cBAG_RIDGE"] = (
        full_cohort_cbag
    )

    full_table["PREDICTED_AGE_RIDGE_OOF"] = (
        oof_prediction
    )

    full_table.to_csv(
        OUTPUT_DIR / "fMRI_Ridge_full_cohort_predictions.csv",
        index=False,
    )

        # --------------------------------------------------------
    # Metrics: CN OOF and all-scan predictions
    # Insert after full_table.to_csv(...) and before dataset_summary
    # --------------------------------------------------------

    metric_rows = []

    def make_metric_row(
        evaluation,
        dataset,
        rows,
        true_col,
        pred_col,
        participant_col,
    ):
        """Calculate MAE, RMSE, R2, and Pearson r for a row subset."""
        y_true = pd.to_numeric(
            rows[true_col], errors="coerce"
        ).to_numpy(dtype=float)

        y_pred = pd.to_numeric(
            rows[pred_col], errors="coerce"
        ).to_numpy(dtype=float)

        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        n = len(y_true)

        if participant_col in rows.columns:
            participants = rows.loc[
                valid, participant_col
            ].nunique()
        else:
            participants = np.nan

        if n == 0:
            mae = rmse = r2 = pearson_r = np.nan
        else:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            r2 = (
                r2_score(y_true, y_pred)
                if n >= 2 and np.std(y_true) > 0
                else np.nan
            )

            pearson_r = (
                pearsonr(y_true, y_pred).statistic
                if n >= 2
                and np.std(y_true) > 0
                and np.std(y_pred) > 0
                else np.nan
            )

        return {
            "evaluation": evaluation,
            "DATASET": dataset,
            "n_scans": n,
            "n_unique_participants": participants,
            "MAE_years": mae,
            "RMSE_years": rmse,
            "R2": r2,
            "Pearson_r": pearson_r,
        }

    datasets = sorted(
        aligned["_manifest_dataset"].dropna().unique()
    )

    # ---- CN out-of-fold metrics, by dataset ----
    # Every prediction comes from a fold model that did not train
    # on that participant.
    for dataset in datasets:
        dataset_rows = oof_table.loc[
            oof_table["DATASET"] == dataset
        ]

        metric_rows.append(
            make_metric_row(
                evaluation="CN_OOF_by_dataset",
                dataset=dataset,
                rows=dataset_rows,
                true_col="CHRONOLOGICAL_AGE",
                pred_col="PREDICTED_AGE_OOF",
                participant_col="PARTICIPANT_GROUP",
            )
        )

    # ---- Pooled CN out-of-fold metrics ----
    metric_rows.append(
        make_metric_row(
            evaluation="CN_OOF_pooled",
            dataset="ALL",
            rows=oof_table,
            true_col="CHRONOLOGICAL_AGE",
            pred_col="PREDICTED_AGE_OOF",
            participant_col="PARTICIPANT_GROUP",
        )
    )

    # ---- All-scan metrics, by dataset ----
    # Uses predictions from the final model trained on all CN scans.
    for dataset in datasets:
        dataset_rows = full_table.loc[
            full_table["_manifest_dataset"] == dataset
        ]

        metric_rows.append(
            make_metric_row(
                evaluation="all_scans_by_dataset",
                dataset=dataset,
                rows=dataset_rows,
                true_col="_age",
                pred_col="PREDICTED_AGE_RIDGE",
                participant_col="_participant_group",
            )
        )

    # ---- Pooled all-scan metrics ----
    metric_rows.append(
        make_metric_row(
            evaluation="all_scans_pooled",
            dataset="ALL",
            rows=full_table,
            true_col="_age",
            pred_col="PREDICTED_AGE_RIDGE",
            participant_col="_participant_group",
        )
    )

    ridge_metrics = pd.DataFrame(metric_rows)

    metrics_path = (
        OUTPUT_DIR
        / "fMRI_Ridge_metrics_CN_OOF_and_all_scans.csv"
    )
    ridge_metrics.to_csv(metrics_path, index=False)

    print("\nRidge metrics: CN OOF and all scans")
    print(ridge_metrics.to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path}")

    # --------------------------------------------------------
    # Summarize BAG and cBAG by dataset.
    # --------------------------------------------------------
    dataset_summary = (
        full_table.groupby(
            "_manifest_dataset",
            dropna=False,
        )
        .agg(
            n_scans=("BAG_RIDGE", "count"),
            mean_BAG=("BAG_RIDGE", "mean"),
            sd_BAG=("BAG_RIDGE", "std"),
            median_BAG=("BAG_RIDGE", "median"),
            mean_cBAG=("cBAG_RIDGE", "mean"),
            sd_cBAG=("cBAG_RIDGE", "std"),
            median_cBAG=("cBAG_RIDGE", "median"),
            mean_chronological_age=("_age", "mean"),
            mean_predicted_age=("PREDICTED_AGE_RIDGE", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "_manifest_dataset": "DATASET",
            }
        )
    )

    dataset_summary.to_csv(
        OUTPUT_DIR / "fMRI_Ridge_BAG_summary_by_dataset.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        oof_table["CHRONOLOGICAL_AGE"],
        oof_table["PREDICTED_AGE_OOF"],
        alpha=0.65,
        s=24,
    )

    lower = min(
        oof_table["CHRONOLOGICAL_AGE"].min(),
        oof_table["PREDICTED_AGE_OOF"].min(),
    )

    upper = max(
        oof_table["CHRONOLOGICAL_AGE"].max(),
        oof_table["PREDICTED_AGE_OOF"].max(),
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="black",
        linewidth=1,
        label="Identity line",
    )

    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("OOF predicted age (years)")
    ax.set_title(
        "fMRI Ridge Regression: CN Out-of-Fold Predictions"
    )
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR
        / "figure_Ridge_CN_OOF_predicted_vs_actual.png",
        dpi=300,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.scatter(
        oof_table["CHRONOLOGICAL_AGE"],
        oof_table["BAG_OOF"],
        alpha=0.65,
        s=24,
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("OOF BAG (years)")
    ax.set_title(
        "fMRI Ridge Regression: CN OOF BAG by Age"
    )
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR
        / "figure_Ridge_CN_OOF_BAG_vs_age.png",
        dpi=300,
    )
    plt.close(fig)

    datasets = list(
        full_table["_manifest_dataset"]
        .dropna()
        .unique()
    )

    bag_values = [
        full_table.loc[
            full_table["_manifest_dataset"] == dataset,
            "BAG_RIDGE",
        ].dropna().to_numpy()
        for dataset in datasets
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.boxplot(
        bag_values,
        showfliers=True,
    )
    # Set ticks separately to support Matplotlib versions using either
    # labels or tick_labels for boxplot's category names.
    ax.set_xticks(np.arange(1, len(datasets) + 1))
    ax.set_xticklabels(datasets)

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("BAG (years)")
    ax.set_title(
        "fMRI Ridge Regression: Full-Cohort BAG by Dataset"
    )
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR
        / "figure_Ridge_full_cohort_BAG_by_dataset.png",
        dpi=300,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Save metrics and final model.
    # --------------------------------------------------------
    summary = {
        "model": "Ridge regression",
        "input_manifest": str(MANIFEST_PATH),
        "input_metadata": str(metadata_path),
        "output_dir": str(OUTPUT_DIR),
        "n_valid_scans": int(len(aligned)),
        "n_CN_OOF_scans": int(len(cn_indices)),
        "dataset_CN_overrides": DATASET_CN_OVERRIDES,
        "n_unique_CN_participants": int(
            n_unique_cn_participants
        ),
        "n_outer_folds": int(n_outer_folds),
        "inner_folds": int(N_INNER_FOLDS),
        "features": int(X.shape[1]),
        "alpha_grid": [
            float(alpha)
            for alpha in ALPHAS
        ],
        "best_alpha_by_outer_fold": [
            float(alpha)
            for alpha in best_alphas
        ],
        "final_alpha_median": float(final_alpha),
        "CN_pooled_OOF_metrics": pooled_metrics,
        "cBAG_bias_slope": float(bias_slope),
        "cBAG_bias_intercept": float(bias_intercept),
        "bag_definition": (
            "predicted age minus chronological age"
        ),
        "cBAG_definition": (
            "BAG minus the age-related linear trend estimated "
            "from CN out-of-fold predictions"
        ),
        "warning": (
            "Full-cohort predictions for CN scans come from a model "
            "trained on all CN scans and are therefore in-sample. "
            "Use CN OOF predictions for an unbiased CN reference."
        ),
        "harmonization_warning": (
            "If ComBat was fit once on the full dataset and preserved "
            "age, the CV prediction metrics may contain upstream "
            "harmonization leakage."
        ),
    }

    with open(
        OUTPUT_DIR / "fMRI_Ridge_CV_metrics.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            summary,
            output_file,
            indent=2,
        )

    joblib.dump(
        final_model,
        OUTPUT_DIR / "fMRI_Ridge_final_model.joblib",
    )

    print("\nPooled CN OOF metrics:")
    print(
        json.dumps(
            pooled_metrics,
            indent=2,
        )
    )

    print(f"\nFinal alpha (median across folds): {final_alpha:g}")
    print("\nDataset BAG summary:")
    print(dataset_summary.to_string(index=False))
    print("\nAll Ridge outputs saved under:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
