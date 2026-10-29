#!/usr/bin/env python3
"""Matched-sample DWI/fMRI Ridge baselines, early fusion and mean late fusion.

Run: python code/Ridge_DWI_fMRI_fusion.py
Only CN scans with agreeing ages and no sex conflicts enter training.
CN follows the intersection of existing unimodal policies (including DWI's
HABS cohort override and fMRI's explicit diagnoses). Unresolved age and CN
conflicts are excluded and logged. All models share grouped nested CV.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fmri_bag_cv import GroupKFold
from fusion_transformers import ModalityBlockWeight
from prepare_DWI_fMRI_fusion_metadata import DWI, FMRI, ROOT, load_modality, audit_pairs


ALPHAS = np.logspace(-3, 4, 15)
MODELS = ["DWI", "fMRI", "EARLY_FUSION"]
FMRI_WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def features(path, modality):
    matrix = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
    if matrix.shape != (84, 84) or not np.isfinite(matrix).all():
        raise ValueError(f"Expected finite 84x84 matrix: {path}")
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0)
    if modality == "DWI":
        matrix = np.clip(matrix, 0, None)
    return matrix[np.triu_indices(84, 1)].astype(np.float64)


def model_search(groups, weighted=False):
    n = min(4, len(np.unique(groups)))
    if n < 2:
        raise ValueError("Need at least two training participants for inner CV")
    steps = [("scaler", StandardScaler())]
    grid = {"ridge__alpha": ALPHAS}
    if weighted:
        steps.append(("block_weight", ModalityBlockWeight()))
        grid["block_weight__fmri_weight"] = FMRI_WEIGHTS
    steps.append(("ridge", Ridge()))
    return GridSearchCV(
        Pipeline(steps), grid, scoring="neg_mean_absolute_error",
        cv=GroupKFold(n), n_jobs=1, refit=True,
    )


def metrics(age, predicted):
    age = np.asarray(age, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    bag = predicted - age
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else None
    return {
        "MAE_years": float(mean_absolute_error(age, predicted)),
        "RMSE_years": float(np.sqrt(mean_squared_error(age, predicted))),
        "R2": float(r2_score(age, predicted)) if len(age) > 1 and np.std(age) > 0 else None,
        "Pearson_r": corr(age, predicted), "mean_BAG_years": float(bag.mean()),
        "BAG_age_r": corr(age, bag),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=None)
    parser.add_argument("--weighted", action="store_true",
                        help="Add nested-CV weighted feature concatenation and save separately")
    # Notebook kernels add their own --f connection argument to sys.argv.
    # Accept explicit arguments via main([...]); preserve strict CLI parsing.
    if argv is None and "ipykernel" in sys.modules:
        argv = []
    args = parser.parse_args(argv)
    out = args.output_dir or (ROOT / "results/final/Fusion" /
                              ("Weighted_Ridge" if args.weighted else "Ridge_baseline"))
    model_names = MODELS + (["WEIGHTED_EARLY_FUSION"] if args.weighted else [])
    out.mkdir(parents=True, exist_ok=True)
    d = load_modality(DWI / "harmonized_connectomes/harmonized_DWI_manifest.csv",
                      DWI / "metadata/metadata_used_for_joint_combat.csv", "dwi")
    f = load_modality(FMRI / "harmonized_connectomes/harmonized_fMRI_manifest.csv",
                      FMRI / "02_canonical_metadata.csv", "fmri")
    paired = audit_pairs(d.merge(f, on=["dataset", "scan_id"], validate="one_to_one"), 0.01)
    paired = paired.sort_values(["dataset", "scan_id"]).reset_index(drop=True)
    arrays = {"DWI": [], "fMRI": []}
    valid = []
    failures = []
    for i, row in paired.iterrows():
        try:
            vd = features(row.connectome_dwi, "DWI")
            vf = features(row.connectome_fmri, "fMRI")
            arrays["DWI"].append(vd)
            arrays["fMRI"].append(vf)
            valid.append(i)
        except Exception as error:
            failures.append({"dataset": row.dataset, "scan_id": row.scan_id, "error": str(error)})
    paired["matrix_valid"] = paired.index.isin(valid)
    reasons = []
    for _, row in paired.iterrows():
        flags = []
        if not row.candidate_cn_existing_policies:
            flags.append("not_CN_under_both_existing_policies")
        if row.age_status != "agrees_within_tolerance":
            flags.append("age_missing_or_conflicting")
        if row.sex_conflict:
            flags.append("sex_conflict")
        if not row.matrix_valid:
            flags.append("invalid_matrix")
        if not (row.metadata_found_dwi and row.metadata_found_fmri):
            flags.append("missing_metadata")
        reasons.append(";".join(flags))
    paired["training_exclusion_reason"] = reasons
    paired["training_included"] = paired.training_exclusion_reason.eq("")
    paired.to_csv(out / "paired_sample_selection.csv", index=False)
    paired.loc[~paired.training_included].to_csv(out / "excluded_from_CN_training.csv", index=False)
    pd.DataFrame(failures, columns=["dataset", "scan_id", "error"]).to_csv(out / "matrix_failures.csv", index=False)
    all_valid = paired.loc[valid].reset_index(drop=True)
    if not valid:
        raise ValueError("No valid paired matrices")
    all_x = {name: np.vstack(rows) for name, rows in arrays.items()}
    # StandardScaler operates per column, so scaling after concatenation
    # equals separately scaling the two feature blocks within each CV split.
    all_x["EARLY_FUSION"] = np.concatenate([all_x["DWI"], all_x["fMRI"]], axis=1)
    if args.weighted:
        all_x["WEIGHTED_EARLY_FUSION"] = all_x["EARLY_FUSION"]
    mask = all_valid.training_included.to_numpy()
    cn = all_valid.loc[mask].reset_index(drop=True)
    xs = {name: value[mask] for name, value in all_x.items()}
    y = cn.unified_age.to_numpy(dtype=float)
    groups = cn.participant_group.to_numpy()
    if len(np.unique(groups)) < 5:
        raise ValueError("Need at least five CN participants")
    predictions = {name: np.full(len(cn), np.nan) for name in model_names}
    folds = np.full(len(cn), -1)
    fold_metrics = []
    print(cn.groupby("dataset").agg(scans=("scan_id", "size"), participants=("participant_group", "nunique")).to_string(), flush=True)
    for fold, (train, test) in enumerate(GroupKFold(5).split(y, groups=groups), 1):
        assert not set(groups[train]) & set(groups[test])
        folds[test] = fold
        for name in model_names:
            search = model_search(groups[train], weighted=name == "WEIGHTED_EARLY_FUSION")
            search.fit(xs[name][train], y[train], groups=groups[train])
            predicted = search.predict(xs[name][test])
            predictions[name][test] = predicted
            result = metrics(y[test], predicted)
            fold_metrics.append({"fold": fold, "model": name,
                                 "alpha": float(search.best_params_["ridge__alpha"]),
                                 "fmri_feature_weight": search.best_params_.get("block_weight__fmri_weight"),
                                 "n_train": len(train), "n_test": len(test),
                                 "inner_CV_MAE": float(-search.best_score_), **result})
            print(f"Fold {fold} {name}: MAE={result['MAE_years']:.3f}; R2={result['R2']:.3f}", flush=True)
            if name == "WEIGHTED_EARLY_FUSION":
                pd.DataFrame(search.cv_results_).to_csv(out / f"weighted_inner_search_fold_{fold}.csv", index=False)
                print("  Selected fMRI feature weight:", search.best_params_["block_weight__fmri_weight"], flush=True)
    predictions["LATE_MEAN"] = (predictions["DWI"] + predictions["fMRI"]) / 2
    cn["FOLD"] = folds
    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Incomplete OOF predictions for {name}")
        cn[f"PREDICTED_AGE_{name}_OOF"] = values
        cn[f"BAG_{name}_OOF"] = values - y
    cn.to_csv(out / "CN_OOF_predictions.csv", index=False)
    cn[["dataset", "scan_id", "participant_group", "unified_age", "FOLD"]].to_csv(out / "outer_fold_assignments.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(out / "fold_metrics.csv", index=False)
    rows = []
    for dataset, frame in [("ALL", cn), *cn.groupby("dataset")]:
        for name in predictions:
            rows.append({"dataset": dataset, "model": name, "n_scans": len(frame),
                         "n_participants": frame.participant_group.nunique(),
                         **metrics(frame.unified_age, frame[f"PREDICTED_AGE_{name}_OOF"])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "OOF_metrics_by_dataset_and_pooled.csv", index=False)
    final_parameters = {}
    final_weights = {}
    for name in model_names:
        search = model_search(groups, weighted=name == "WEIGHTED_EARLY_FUSION")
        search.fit(xs[name], y, groups=groups)
        final_parameters[name] = float(search.best_params_["ridge__alpha"])
        final_weights[name] = search.best_params_.get("block_weight__fmri_weight")
        if name == "WEIGHTED_EARLY_FUSION":
            pd.DataFrame(search.cv_results_).to_csv(out / "weighted_final_inner_search.csv", index=False)
        slope, intercept = np.polyfit(y, predictions[name] - y, 1)
        joblib.dump({"model": search.best_estimator_, "model_name": name,
                     "feature_order": "DWI upper triangle then fMRI upper triangle" if "FUSION" in name else name + " upper triangle",
                     "fmri_feature_weight": final_weights[name],
                     "alpha": final_parameters[name], "bag_bias_slope": float(slope),
                     "bag_bias_intercept": float(intercept)}, out / f"{name}_final_model.joblib")
        all_valid[f"PREDICTED_AGE_{name}"] = search.predict(all_x[name])
        all_valid[f"BAG_{name}"] = all_valid[f"PREDICTED_AGE_{name}"] - all_valid.unified_age
    all_valid["PREDICTED_AGE_LATE_MEAN"] = (all_valid.PREDICTED_AGE_DWI + all_valid.PREDICTED_AGE_fMRI) / 2
    all_valid.to_csv(out / "full_paired_predictions.csv", index=False)
    pooled = table[table.dataset.eq("ALL")].set_index("model")
    best = pooled.loc[["DWI", "fMRI"], "MAE_years"].idxmin()
    summary = {
        "n_paired_scans": len(paired), "n_CN_scans": len(cn),
        "n_CN_participants": int(cn.participant_group.nunique()),
        "CN_policy": "Intersection of existing modality CN policies; DWI HABS override retained, fMRI explicit CN required",
        "age_policy": "Absolute difference <=0.01 year; fMRI age used only when agreeing; otherwise excluded from training",
        "sex_policy": "Explicit conflicts excluded; missing sex allowed because sex is not a model feature",
        "outer_CV": "Shared deterministic five-fold participant-grouped CV",
        "inner_CV": "Four-fold participant-grouped alpha selection; scaling fit inside every split",
        "alphas": ALPHAS.tolist(), "final_alphas": final_parameters,
        "weighted_fusion_enabled": args.weighted,
        "fmri_weight_grid": FMRI_WEIGHTS if args.weighted else None,
        "final_fmri_weights": final_weights,
        "weight_interpretation": "After per-column training-only standardization: [Z_DWI, w*Z_fMRI]. DWI scale=1; w is not a percentage. Weight and alpha selected jointly inside each outer training fold.",
        "best_unimodal_model": best,
        "early_fusion_MAE_reduction_years": float(pooled.loc[best, "MAE_years"] - pooled.loc["EARLY_FUSION", "MAE_years"]),
        "roi_provider_confirmation": "User reports professor confirmed both modalities used the referenced label; atlas order corroborated by table/source audit. Per-scan regeneration not performed.",
        "roi_lookup_sha256": hashlib.sha256((ROOT / "data/final/atlas/IITmean_RPI/IITmean_RPI_labels_lookup.txt").read_bytes()).hexdigest(),
        "limitations": ["Age filtering removes most HABS scans and changes the evaluation population.",
                        "Matching visit IDs does not establish scan-date proximity; dates unavailable.",
                        "Upstream full-data ComBat preserving age may leak target information.",
                        "Full-paired predictions include training CN scans; use OOF metrics for model comparison.",
                        "Unresolved ages remain missing in full-paired BAG outputs."],
    }
    if args.weighted:
        summary["weighted_MAE_reduction_vs_best_unimodal_years"] = float(pooled.loc[best, "MAE_years"] - pooled.loc["WEIGHTED_EARLY_FUSION", "MAE_years"])
        summary["weighted_MAE_reduction_vs_unweighted_years"] = float(pooled.loc["EARLY_FUSION", "MAE_years"] - pooled.loc["WEIGHTED_EARLY_FUSION", "MAE_years"])
    (out / "fusion_summary.json").write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = table.dataset.unique()
    x = np.arange(len(datasets))
    width = 0.8 / len(predictions)
    for offset, name in enumerate(predictions):
        values = table[table.model.eq(name)].set_index("dataset").loc[datasets, "MAE_years"]
        ax.bar(x + (offset - (len(predictions) - 1) / 2) * width, values, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("CN OOF MAE (years)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "OOF_MAE_comparison.png", dpi=200)
    plt.close(fig)
    print(table.to_string(index=False))
    print("Saved to:", out)


if __name__ == "__main__":
    main()
