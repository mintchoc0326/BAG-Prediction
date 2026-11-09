"""Raw-input preparation and conservative matched-scan construction."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from foldwise_dwi import load_eligible


def prepare(root, output, min_cn=5, tolerance=0.01):
    output = Path(output)
    if output.exists():
        raise ValueError("Prepared output already exists; choose a new directory")
    output.mkdir(parents=True)
    data = {}
    sources = []
    counts = []
    for modality in ["DWI", "fMRI"]:
        X, meta, excluded, provenance = load_eligible(
            root=Path(root), min_cn_participants=min_cn, modality=modality
        )
        data[modality] = (X, meta)
        sources += provenance
        np.savez_compressed(output / f"{modality}_raw_private.npz", X=X)
        meta.to_csv(output / f"{modality}_metadata_private.csv", index=False)
        excluded.to_csv(output / f"{modality}_eligibility_private.csv", index=False)
        counts.append(
            dict(
                population=modality,
                scans=len(meta),
                participants=meta.participant_group.nunique(),
                CN_scans=int(meta.is_cn.sum()),
            )
        )
    d, f = data["DWI"][1].copy(), data["fMRI"][1].copy()
    d["index_DWI"] = np.arange(len(d))
    f["index_fMRI"] = np.arange(len(f))
    paired = d.merge(
        f, on=["cohort", "scan_id"], suffixes=("_DWI", "_fMRI"), validate="one_to_one"
    )
    reason = pd.Series("", index=paired.index)
    conditions = [
        (
            paired.participant_group_DWI.ne(paired.participant_group_fMRI),
            "participant_conflict",
        ),
        (paired.batch_DWI.ne(paired.batch_fMRI), "batch_conflict"),
        (
            paired.clinical_diagnosis_DWI.ne(paired.clinical_diagnosis_fMRI),
            "diagnosis_conflict",
        ),
        ((paired.age_DWI - paired.age_fMRI).abs() > tolerance, "age_conflict"),
        (paired.sex_DWI.ne(paired.sex_fMRI), "sex_conflict"),
    ]
    for mask, label in conditions:
        reason.loc[mask & reason.eq("")] = label
    support = (
        paired.loc[reason.eq("") & paired.is_cn_DWI]
        .groupby("batch_DWI")
        .participant_group_DWI.nunique()
    )
    reason.loc[
        reason.eq("") & ~paired.batch_DWI.isin(support[support >= min_cn].index)
    ] = "insufficient_paired_CN_support"
    paired["exclusion_reason"] = reason
    paired.to_csv(output / "pairing_audit_private.csv", index=False)
    paired.groupby("exclusion_reason", dropna=False).size().rename(
        "scans"
    ).reset_index().to_csv(output / "pairing_counts.csv", index=False)
    matched = paired.loc[reason.eq("")].reset_index(drop=True)
    if matched.empty:
        raise ValueError("No unambiguous supported pairs; inspect pairing audit")
    meta = d.iloc[matched.index_DWI].reset_index(drop=True).drop(columns="index_DWI")
    meta["age_fMRI"] = matched.age_fMRI.to_numpy()
    meta["pairing_status"] = "same_key_age_sex_diagnosis_agree_dates_unverified"
    meta.to_csv(output / "paired_metadata_private.csv", index=False)
    np.savez_compressed(
        output / "paired_raw_private.npz",
        DWI=data["DWI"][0][matched.index_DWI],
        fMRI=data["fMRI"][0][matched.index_fMRI],
    )
    counts.append(
        dict(
            population="paired",
            scans=len(meta),
            participants=meta.participant_group.nunique(),
            CN_scans=int(meta.is_cn.sum()),
        )
    )
    pd.DataFrame(counts).to_csv(output / "sample_counts.csv", index=False)
    manifest = dict(
        status="complete",
        min_CN_participants_per_batch=min_cn,
        age_tolerance_years=tolerance,
        limitations=[
            "Archived eligible universes, not all original scans.",
            "Clinical label/visit provenance and ADRC protocol remain unverified.",
            "Matched pairs use DWI age only after agreement; dates not independently verified.",
            "Modality-wide runs are supplementary; cross-model comparisons use paired population.",
        ],
        sources=sources,
    )
    manifest["files_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.glob("*private*")
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return pd.DataFrame(counts)


def load(prepared, population="paired", modalities=("DWI", "fMRI")):
    prepared = Path(prepared)
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Prepared data are incomplete")
    for filename, digest in manifest.get("files_sha256", {}).items():
        if hashlib.sha256((prepared / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Prepared input changed: " + filename)
    if population == "paired":
        meta = pd.read_csv(prepared / "paired_metadata_private.csv")
        archive = np.load(prepared / "paired_raw_private.npz")
        return {m: archive[m] for m in modalities}, meta
    if modalities != (population,):
        raise ValueError(
            "Modality-wide populations cannot support fusion or matched comparisons"
        )
    return {
        population: np.load(prepared / f"{population}_raw_private.npz")["X"]
    }, pd.read_csv(prepared / f"{population}_metadata_private.csv")
