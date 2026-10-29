#!/usr/bin/env python3
"""Pair original DWI/fMRI manifests and audit metadata without resolving conflicts.

Run from any directory. This prepares metadata only; it does not train models
or alter single-modality inputs/results. Matching visit IDs is provisional:
the current metadata does not provide acquisition dates or ROI labels.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DWI = ROOT / "results/final/DWI"
FMRI = ROOT / "results/final/fMRI"
DEFAULT_OUTPUT = ROOT / "results/final/Fusion/metadata"
CN_LABELS = {"CN", "HC", "NC", "CU", "CONTROL", "HEALTHY", "NORMAL",
             "COGNITIVELY NORMAL", "COGNITIVELY UNIMPAIRED", "HEALTHY CONTROL"}


def normalize(value):
    return "" if pd.isna(value) else str(value).strip().upper()


def diagnosis(value):
    value = re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", normalize(value)))
    if value in CN_LABELS:
        return "CN"
    if value in {"EMCI", "LMCI", "MCI"}:
        return "MCI"
    return value if value not in {"", "NAN", "UNKNOWN"} else "UNKNOWN"


def check_unique(frame, keys, source):
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, keys].head(10).to_dict("records")
        raise ValueError(f"Duplicate normalized keys in {source}: {examples}")


def load_modality(manifest_path, metadata_path, modality):
    manifest = pd.read_csv(manifest_path)
    meta = pd.read_csv(metadata_path)
    id_column = "__id__" if modality == "dwi" else "connectome_key"
    age_column = "AGE" if modality == "dwi" else "Age"
    sex_column = "SEX" if modality == "dwi" else "Sex"
    diag_column = "GROUP" if modality == "dwi" else "Diagnosis"
    for frame, column in [(manifest, "connectome_key"), (meta, id_column)]:
        frame["dataset"] = frame.DATASET.map(normalize)
        frame["scan_id"] = frame[column].map(normalize)
        if (frame[["dataset", "scan_id"]] == "").any().any():
            raise ValueError(f"Empty matching keys in {modality}")
    keys = ["dataset", "scan_id"]
    check_unique(manifest, keys, str(manifest_path))
    check_unique(meta, keys, str(metadata_path))
    manifest = manifest[keys + ["connectome_key", "harmonized_connectome"]].rename(
        columns={"connectome_key": f"original_scan_id_{modality}",
                 "harmonized_connectome": f"connectome_{modality}"}
    )
    fields = meta[keys].copy()
    fields[f"age_{modality}"] = pd.to_numeric(meta[age_column], errors="coerce")
    fields[f"sex_{modality}"] = meta[sex_column].map(normalize).replace(
        {"MALE": "M", "FEMALE": "F", "NAN": ""}
    )
    fields[f"diagnosis_raw_{modality}"] = meta[diag_column]
    fields[f"diagnosis_{modality}"] = meta[diag_column].map(diagnosis)
    fields[f"metadata_source_{modality}"] = str(metadata_path)
    result = manifest.merge(fields, on=keys, how="left", validate="one_to_one", indicator=True)
    result[f"metadata_found_{modality}"] = result.pop("_merge").eq("both")
    result[f"path_exists_{modality}"] = result[f"connectome_{modality}"].map(
        lambda path: Path(str(path)).is_file()
    )
    return result


def audit_pairs(paired, tolerance):
    paired = paired.copy()
    # Outer joins can promote boolean columns to object; restore booleans
    # before applying logical negation in the audit masks.
    for modality in ["dwi", "fmri"]:
        for field in ["metadata_found", "path_exists"]:
            column = f"{field}_{modality}"
            paired[column] = paired[column].astype("boolean").fillna(False).astype(bool)
    paired["participant_id"] = paired.scan_id.str.replace(r"_Y\d+$", "", regex=True)
    paired["participant_group"] = paired.dataset + "__" + paired.participant_id
    paired["visit"] = paired.scan_id.str.extract(r"_(Y\d+)$", expand=False).fillna("")
    paired["pairing_status"] = "same_normalized_visit_id_dates_unverified"
    paired["age_difference_dwi_minus_fmri"] = paired.age_dwi - paired.age_fmri
    finite_ages = np.isfinite(paired.age_dwi) & np.isfinite(paired.age_fmri)
    age_agrees = finite_ages & (paired.age_difference_dwi_minus_fmri.abs() <= tolerance)
    paired["age_status"] = np.select(
        [~finite_ages, age_agrees], ["missing_or_nonfinite", "agrees_within_tolerance"],
        default="conflict_needs_source_review",
    )
    # Choose a documented source only when both ages already agree; never average conflicts.
    paired["unified_age"] = paired.age_fmri.where(age_agrees)
    paired["unified_age_source"] = np.where(age_agrees, "fMRI_agrees_with_DWI", "unresolved")
    d = paired.diagnosis_dwi.fillna("UNKNOWN")
    f = paired.diagnosis_fmri.fillna("UNKNOWN")
    both_known = d.ne("UNKNOWN") & f.ne("UNKNOWN")
    agrees = both_known & d.eq(f)
    paired["diagnosis_status"] = np.select(
        [d.eq("UNKNOWN") | f.eq("UNKNOWN"), agrees],
        ["missing_or_unknown_in_at_least_one_modality", "agrees_after_normalization"],
        default="conflict_needs_source_review",
    )
    paired["unified_diagnosis"] = d.where(agrees, "UNRESOLVED")
    known_sex = paired.sex_dwi.isin(["M", "F"]) & paired.sex_fmri.isin(["M", "F"])
    paired["sex_conflict"] = known_sex & paired.sex_dwi.ne(paired.sex_fmri)
    paired["sex_status"] = np.select(
        [~known_sex, paired.sex_conflict], ["missing_or_unknown", "conflict"], default="agrees"
    )
    paired["cn_dwi_explicit"] = d.eq("CN")
    paired["cn_fmri_explicit"] = f.eq("CN")
    paired["cn_dwi_existing_policy"] = d.eq("CN") | paired.dataset.eq("HABS")
    paired["candidate_cn_existing_policies"] = paired.cn_dwi_existing_policy & f.eq("CN")
    paired["cn_policy_disagreement"] = paired.cn_dwi_existing_policy.ne(f.eq("CN"))
    paired["cn_explicit_consensus"] = d.eq("CN") & f.eq("CN")
    paired["scan_date_dwi"] = ""
    paired["scan_date_fmri"] = ""
    paired["scan_interval_days"] = np.nan
    paired["acquisition_dates_verified"] = False
    paired["roi_order_verified"] = False
    paired["metadata_consensus_cn"] = (
        paired.cn_explicit_consensus & age_agrees & paired.sex_status.eq("agrees")
        & paired.metadata_found_dwi & paired.metadata_found_fmri
        & paired.path_exists_dwi & paired.path_exists_fmri
    )
    paired["review_required"] = (
        ~age_agrees | ~agrees | paired.sex_status.ne("agrees")
        | ~paired.path_exists_dwi | ~paired.path_exists_fmri
        | ~paired.metadata_found_dwi | ~paired.metadata_found_fmri
    )
    return paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--age-tolerance", type=float, default=0.01,
                        help="Age agreement tolerance in years; default 0.01.")
    args = parser.parse_args()
    if not np.isfinite(args.age_tolerance) or args.age_tolerance < 0:
        parser.error("--age-tolerance must be finite and nonnegative")
    dwi = load_modality(DWI / "harmonized_connectomes/harmonized_DWI_manifest.csv",
                        DWI / "metadata/metadata_used_for_joint_combat.csv", "dwi")
    fmri = load_modality(FMRI / "harmonized_connectomes/harmonized_fMRI_manifest.csv",
                         FMRI / "02_canonical_metadata.csv", "fmri")
    joined = dwi.merge(fmri, on=["dataset", "scan_id"], how="outer",
                       validate="one_to_one", indicator=True)
    paired = audit_pairs(joined.loc[joined._merge.eq("both")].drop(columns="_merge"),
                         args.age_tolerance)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "paired_metadata": paired,
        "unpaired_scans": joined.loc[joined._merge.ne("both")],
        "metadata_conflicts": paired.loc[paired.review_required],
        "age_conflicts": paired.loc[paired.age_status.ne("agrees_within_tolerance")],
        "diagnosis_review": paired.loc[paired.diagnosis_status.ne("agrees_after_normalization")],
        "candidate_CN_existing_policies": paired.loc[paired.candidate_cn_existing_policies],
        "explicit_consensus_CN": paired.loc[paired.cn_explicit_consensus],
    }
    for name, frame in tables.items():
        frame.to_csv(output / f"fusion_{name}.csv", index=False)
    by_dataset = paired.groupby("dataset").agg(
        paired_scans=("scan_id", "size"), participants=("participant_group", "nunique"),
        candidate_cn_scans=("candidate_cn_existing_policies", "sum"),
        explicit_consensus_cn_scans=("cn_explicit_consensus", "sum"),
        cn_policy_disagreements=("cn_policy_disagreement", "sum"),
        metadata_consensus_cn_scans=("metadata_consensus_cn", "sum"),
        review_scans=("review_required", "sum"),
    )
    by_dataset.to_csv(output / "fusion_summary_by_dataset.csv")
    summary = {
        "n_dwi_scans": len(dwi), "n_fmri_scans": len(fmri),
        "n_paired_scans": len(paired),
        "n_paired_participants": int(paired.participant_group.nunique()),
        "n_candidate_cn_existing_policies": int(paired.candidate_cn_existing_policies.sum()),
        "n_candidate_cn_participants": int(paired.loc[paired.candidate_cn_existing_policies, "participant_group"].nunique()),
        "n_explicit_consensus_cn": int(paired.cn_explicit_consensus.sum()),
        "n_age_conflicts": int(paired.age_status.eq("conflict_needs_source_review").sum()),
        "n_sex_conflicts": int(paired.sex_conflict.sum()),
        "age_tolerance_years": args.age_tolerance,
        "diagnosis_normalization": "CN synonyms -> CN; EMCI/LMCI -> MCI; SMC remains separate; raw labels retained",
        "age_resolution": "Use fMRI age only when both ages agree within tolerance; otherwise unresolved",
        "candidate_policy": "Reproduce prior DWI HABS-all-CN and fMRI explicit-CN policies for auditing only",
        "limitations": [
            "Candidate CN membership is not a confirmed training inclusion decision.",
            "Acquisition dates are absent from the input metadata; visit IDs alone do not establish temporal proximity.",
            "ROI order and matrix contents are not validated by this metadata-only step.",
            "No conflicts are automatically adjudicated and no models are trained.",
        ],
    }
    (output / "fusion_metadata_audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(by_dataset.to_string())
    print(json.dumps(summary, indent=2))
    print("Saved to:", output)


if __name__ == "__main__":
    main()
