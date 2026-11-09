"""Artifact audit for unified runs, including calibration and fit boundaries."""
import json
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .splits import audit_plan


def audit(output, allow_partial=False):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "complete" and not allow_partial:
        raise ValueError("Run incomplete")
    for source, digest in manifest.get("code_sha256", {}).items():
        if (
            hashlib.sha256(
                (output / "source_snapshot" / source).read_bytes()
            ).hexdigest()
            != digest
        ):
            raise AssertionError("Source snapshot hash mismatch")
    meta = pd.read_csv(output / "metadata_private.csv")
    plan = json.loads((output / "fold_plan_private.json").read_text())
    audit_plan(plan, meta)
    tasks = list(output.glob("fold*/*/*/*/task.json"))
    coverage = {}
    max_error = 0.0
    for path in tasks:
        task = path.parent
        foldnum = int(task.parts[-4].replace("fold", ""))
        fold = plan["folds"][foldnum - 1]
        model = joblib.load(task / "model_private.joblib")
        pred = pd.read_csv(task / "predictions_private.csv")
        cal = pd.read_csv(task / "calibration_private.csv")
        train = set(fold["train_CN"])
        test = set(fold["test"])
        if set(model["train_indices"]) != train or set(model["test_indices"]) != test:
            raise AssertionError("Model split mismatch")
        if set(pred.row_index) != test or len(pred) != len(test):
            raise AssertionError("Prediction coverage mismatch")
        if set(cal.row_index) != train:
            raise AssertionError("Calibration reference mismatch")
        if not meta.iloc[list(train)].is_cn.all():
            raise AssertionError("Non-CN model training")
        for kind, field in [
            ("global", "cBAG_global"),
            (manifest["settings"].get("calibration", "cohort"), "cBAG"),
        ]:
            from .calibration import BiasCorrection

            bias = BiasCorrection(kind).fit(
                meta.age.iloc[cal.row_index],
                cal.inner_oof_prediction,
                meta.cohort.iloc[cal.row_index],
            )
            np.testing.assert_allclose(
                pred[field],
                bias.apply(pred.predicted_age, pred.age, pred.cohort),
                atol=1e-9,
            )
        permitted = {tuple(sorted(train))}
        for inner in fold["inner"]:
            permitted.add(tuple(sorted(inner["train"])))
            permitted.add(tuple(sorted(inner["early_train"])))
        for fit in json.loads((task / "preprocessing_audit_private.json").read_text()):
            if tuple(sorted(fit["training_indices"])) not in permitted:
                raise AssertionError(
                    "Preprocessing fit outside allowed training partition"
                )
            if set(fit["training_indices"]) & test:
                raise AssertionError("Held-out preprocessing fit")
            max_error = max(max_error, fit["max_train_apply_error"])
        if max_error > 1e-8:
            raise AssertionError("ComBat fit/apply discrepancy")
        key = "/".join(task.parts[-3:])
        coverage.setdefault(key, []).extend(pred.row_index.to_list())
    if manifest["status"] == "complete":
        expected = (
            len(plan["folds"])
            * len(manifest["settings"]["harmonization_modes"])
            * len(manifest["settings"]["learners"])
            * (3 if manifest["population"] == "paired" else 1)
        )
        if len(tasks) != expected:
            raise AssertionError("Missing model tasks")
        for key, rows in coverage.items():
            if sorted(rows) != list(range(len(meta))):
                raise AssertionError("Incomplete OOF coverage " + key)
    result = dict(
        status="passed" if manifest["status"] == "complete" else "partial_tasks_passed",
        completed_tasks=len(tasks),
        complete_OOF_models=len(coverage) if manifest["status"] == "complete" else 0,
        max_train_apply_error=max_error,
        caveat="Implementation checks only; source labels, dates, protocols and ROI order remain unverified.",
    )
    (output / "audit.json").write_text(json.dumps(result, indent=2))
    return result
