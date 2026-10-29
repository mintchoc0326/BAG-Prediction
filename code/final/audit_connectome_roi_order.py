#!/usr/bin/env python3
"""Record atlas-order evidence without executing historical pipeline scripts.

This audit can verify lookup/index-table agreement. Historical source code
is supporting evidence, not proof that every existing matrix used that code.
"""

import csv
import hashlib
import json
from pathlib import Path

import openpyxl


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
ATLAS = ROOT / "data/final/atlas/IITmean_RPI"
OUT = ROOT / "results/final/Fusion/roi_order_audit"
PIPELINE = BASE / "ADNI_connectomics/adrc_fmri_dti_pipeline"
SOURCES = {
    "lookup": ATLAS / "IITmean_RPI_labels_lookup.txt",
    "local_index": ATLAS / "IITmean_RPI_index.xlsx",
    "shared_index": BASE / "atlases/IITmean_RPI/IITmean_RPI_index.xlsx",
    "dwi_generation": PIPELINE / "code_DTI/ADRC_mrtrix_connectomes.py",
    "fmri_generation": PIPELINE / "ADRC_fmri_prep_pipeline/connec_fmri_prep.py",
    "dwi_harmonization": ROOT / "code/final/run_DWI_harmonization.py",
    "fmri_harmonization": ROOT / "code/final/run_fMRI_harmonization.py",
}


def read_index(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(book.active.values)
        headers = list(rows[0])
        result = []
        for row in rows[1:]:
            item = dict(zip(headers, row))
            if item.get("index") is None or item.get("index2") is None:
                continue
            if int(item["index"]) <= 0:
                continue
            result.append({"matrix_position_1based": int(item["index"]),
                           "atlas_id": int(item["index2"]),
                           "structure": item.get("Structure"),
                           "hemisphere": item.get("Hemisphere")})
        return sorted(result, key=lambda x: x["matrix_position_1based"])
    finally:
        book.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lookup = []
    for line in SOURCES["lookup"].read_text().splitlines():
        parts = line.split()
        if parts and int(parts[0]) != 0:
            lookup.append((int(parts[0]), parts[1]))
    index = read_index(SOURCES["local_index"])
    positions_valid = [r["matrix_position_1based"] for r in index] == list(range(1, 85))
    ids_unique = len({r["atlas_id"] for r in index}) == 84
    order_matches = [r["atlas_id"] for r in index] == [r[0] for r in lookup]
    shared_matches = (read_index(SOURCES["shared_index"]) == index
                      if SOURCES["shared_index"].exists() else None)
    names = dict(lookup)
    for row in index:
        position = row["matrix_position_1based"]
        row["lookup_name"] = names.get(row["atlas_id"], "")
        row["lookup_position_1based"] = next(
            (i for i, (label, _) in enumerate(lookup, 1) if label == row["atlas_id"]), None)
        row["position_matches_lookup"] = row["lookup_position_1based"] == position
        row["verification_scope"] = "atlas_tables_and_candidate_pipeline_not_per_scan"
    with (OUT / "roi_order_mapping.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)

    keywords = ["index2", "ordered_num", "roi_list", "timeseries_dict",
                "corrcoef", "savetxt", "tck2connectome", "parcels_csv_2",
                "new_label", "parcels_mif", "triu_indices", "upper_indices",
                "flatten_upper", "correlation_edges", "to_csv"]
    manifest = []
    excerpts = []
    for role, path in SOURCES.items():
        record = {"role": role, "path": str(path), "exists": path.exists()}
        if path.exists():
            data = path.read_bytes()
            record.update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
            if path.suffix == ".py":
                lines = data.decode().splitlines()
                excerpts.append(f"\nSOURCE: {path}\nSHA256: {record['sha256']}\n")
                selected = set()
                for i, line in enumerate(lines):
                    if any(key in line for key in keywords):
                        selected.update(range(max(0, i - 2), min(len(lines), i + 3)))
                for i in sorted(selected):
                    excerpts.append(f"{i + 1}: {lines[i]}\n")
        manifest.append(record)
    (OUT / "source_evidence_excerpts.txt").write_text("".join(excerpts))
    summary = {
        "status": "ATLAS_ORDER_VERIFIED_PER_SCAN_PROVENANCE_NOT_VERIFIED",
        "checks": {"positions_are_1_to_84": positions_valid,
                   "84_unique_atlas_ids": ids_unique,
                   "lookup_order_equals_index_order": order_matches,
                   "local_and_shared_index_mappings_equal": shared_matches},
        "source_files": manifest,
        "evidence_interpretation": [
            "The located DWI source maps atlas index2 IDs to contiguous index labels, then passes the relabeled parcellation to tck2connectome.",
            "The located fMRI source uses the same index2-to-index mapping, sorted unique ROI labels, ordered time series and np.corrcoef.",
            "These historical sources support the candidate mapping, but reference historical paths; they are not linked by execution records to every current cohort and matrix.",
            "Harmonization source excerpts document upper-triangle extraction and reconstruction; this is source evidence, not a historical execution attestation.",
        ],
        "missing_for_full_proof": [
            "Per-cohort/per-scan execution logs or provenance linking each original matrix to the exact generator and atlas mapping version.",
            "Subject-specific relabeled parcellations or exported ROI ID lists confirming all 84 regions and their ordering.",
            "For stronger numerical verification, regenerate fMRI correlations from ordered ROI time series; DWI requires the corresponding tractography/parcellation inputs.",
        ],
        "scanned_script_roots": [str(BASE / name) for name in [
            "harmonization", "ADNI_connectomics", "HABS_connectomics", "ADRC_connectomics", "CONN_fMRI"]],
        "fusion_metadata_roi_verified_updated": False,
    }
    (OUT / "roi_order_evidence.json").write_text(json.dumps(summary, indent=2))
    report = f"""ROI ORDER TRACEABILITY REPORT

Status: {summary['status']}

Verified by direct table comparison:
  Contiguous matrix positions 1..84: {positions_valid}
  84 unique atlas IDs: {ids_unique}
  Lookup non-background row order equals spreadsheet index order: {order_matches}
  Local and shared spreadsheet mappings agree: {shared_matches}

Candidate matrix order: see roi_order_mapping.csv.
The first position corresponds to atlas ID 8, Left_Cerebellum_Cortex,
under the located index2-to-index generation scheme.

Evidence chain:
  Original atlas IDs -> IITmean_RPI_index.xlsx index2-to-index mapping
  -> relabeled parcel IDs 1..84 -> candidate DWI/fMRI generation sources
  -> upper-triangle extraction/reconstruction in harmonization sources.

Important distinction:
The atlas table order is verified. Historical source code supports that
generation scheme. We cannot yet prove that all current matrices were
generated by those exact source versions. No per-scan ROI verification
flags were changed in the fusion metadata.

Review source_evidence_excerpts.txt for exact paths and numbered source
lines, and roi_order_evidence.json for source SHA256 fingerprints and
the missing evidence needed for full per-scan provenance verification.
"""
    (OUT / "ROI_order_traceability_report.txt").write_text(report)
    print(report)
    if not (positions_valid and ids_unique and order_matches):
        raise SystemExit("Atlas mapping checks failed; inspect the audit outputs.")


if __name__ == "__main__":
    main()
