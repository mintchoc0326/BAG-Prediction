"""Train/apply preprocessing for known-batch DWI evaluation.

Transform never accepts age/sex/diagnosis. An optional mean-only ComBat fit uses
training ages to estimate batch offsets; additive correction then cancels the
subject-specific age term exactly. Biological retention is measured, not assumed.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "ADNI": ("ADNI/ADNI_harmonized_metadata(in).csv", "PTID"),
    "HABS": ("HABS/HABS_harmonized_metadata(in).csv", "Subject"),
    "ADRC": ("ADRC/ADRC_harmonized_metadata_merged.csv", "PTID"),
    "ADDECODE": ("ADDecode/AD_DECODE_harmonized_metadata.csv", "ID"),
}
CN_LABELS = {"NORMAL", "CN", "CU", "HC", "COGNITIVELY NORMAL", "COGNITIVELY UNIMPAIRED"}


def normalized(values):
    return values.astype("string").str.strip().str.upper()


def load_eligible(root=ROOT, min_cn_participants=5, modality="DWI"):
    """Re-read raw files; exclude ambiguous joins, never select their first row.

    Cohort universe is the archived 1234-row DWI eligibility manifest, whose
    prior exclusions remain a limitation. No harmonized arrays are read.
    """
    root = Path(root).resolve()
    if modality not in {"DWI", "fMRI"}:
        raise ValueError("Unsupported modality")
    relative = (
        "results/final/DWI/metadata/metadata_used_for_joint_combat.csv"
        if modality == "DWI"
        else "results/final/fMRI/evaluation_inputs/fMRI_metadata_aligned.csv"
    )
    metadata_path = root / relative
    meta = pd.read_csv(metadata_path, low_memory=False)
    if modality == "fMRI":
        meta = meta.rename(
            columns={
                "connectome_key": "__id__",
                "file_path": "filepath",
                "Age": "AGE",
                "Sex": "SEX",
            }
        )
    meta["cohort"] = normalized(meta.DATASET)
    meta["scan_id"] = normalized(meta["__id__"])
    if meta.duplicated(["cohort", "scan_id"]).any():
        raise ValueError("Duplicate scan keys in DWI eligibility manifest")
    frames = []
    sources = []
    for cohort, (relative, participant_column) in SOURCES.items():
        if not meta.cohort.eq(cohort).any():
            continue
        path = root / "data/final" / relative
        frame = pd.read_csv(path, low_memory=False).copy()
        frame["scan_id"] = normalized(frame.CONNECTOME_KEY_USED_FOR_INTERSECTION)
        duplicate = frame.scan_id.duplicated(keep=False)
        fields = pd.DataFrame(
            {
                "scan_id": frame.scan_id,
                "participant_id": normalized(frame[participant_column]),
                "clinical_diagnosis": normalized(frame.DX_Label_harmonized),
                "ambiguous_clinical_key": duplicate,
            }
        )
        # Only unique keys can supply labels/IDs. Keep excluded keys as flags.
        ambiguous = set(fields.loc[duplicate, "scan_id"].dropna())
        fields = fields.loc[~duplicate & fields.scan_id.notna() & fields.scan_id.ne("")]
        part = meta.loc[meta.cohort.eq(cohort)].copy()
        part = part.merge(fields, on="scan_id", how="left", validate="one_to_one")
        part["ambiguous_clinical_key"] = part.scan_id.isin(ambiguous)
        frames.append(part)
        sources.append(
            {
                "source": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    meta = pd.concat(frames, ignore_index=True)
    meta["age"] = pd.to_numeric(meta.AGE, errors="coerce")
    meta["sex"] = normalized(meta.SEX).replace({"MALE": "M", "FEMALE": "F"})
    meta["batch"] = normalized(meta.DATASET_SITE)
    meta["is_cn"] = meta.clinical_diagnosis.isin(CN_LABELS)
    meta["participant_group"] = meta.cohort + "__" + meta.participant_id
    reason = pd.Series("", index=meta.index)

    def exclude(mask, label):
        reason.loc[mask & reason.eq("")] = label

    exclude(meta.ambiguous_clinical_key, "ambiguous_clinical_key")
    exclude(
        meta.participant_id.isna() | meta.participant_id.eq(""),
        "missing_canonical_participant",
    )
    exclude(~np.isfinite(meta.age), "invalid_age")
    exclude(meta.batch.isna() | meta.batch.eq(""), "missing_batch")
    # Fixed feasibility restriction for the known-site estimand, before model evaluation.
    support = (
        meta.loc[reason.eq("") & meta.is_cn]
        .groupby("batch")
        .participant_group.nunique()
    )
    supported = set(support[support >= min_cn_participants].index)
    exclude(
        ~meta.batch.isin(supported), "insufficient_confirmed_CN_participants_in_batch"
    )
    exclusion = meta[
        ["cohort", "scan_id", "participant_group", "batch", "clinical_diagnosis"]
    ].copy()
    exclusion["exclusion_reason"] = reason
    eligible = meta.loc[reason.eq("")].reset_index(drop=True)
    if eligible.empty:
        raise ValueError("No eligible rows")
    raw = []
    for path in eligible.filepath:
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        a = pd.read_csv(path, header=None).to_numpy(dtype=float)
        if a.shape != (84, 84):
            raise ValueError(f"Unexpected matrix shape at {path}")
        # Fixed per-scan undirected representation; no cross-subject parameter fitting.
        a = (a + a.T) / 2
        raw.append(a[np.triu_indices(84, k=1)])
    X = np.asarray(raw)
    if not np.isfinite(X).all():
        raise ValueError("Raw matrices contain nonfinite edges")
    if modality == "DWI" and (X < 0).any():
        raise ValueError("Negative DWI weights")
    if modality == "fMRI" and np.max(np.abs(X)) > 1 + 1e-6:
        raise ValueError("fMRI inputs must be Pearson correlations within [-1,1]")
    sources.append(
        {
            "source": str(metadata_path.relative_to(root)),
            "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        }
    )
    return X, eligible, exclusion, sources


from harmonization import CombatPreprocessor


class DwiPreprocessor(CombatPreprocessor):
    """Compatibility name for saved historical foldwise DWI artifacts."""
