"""Clinical validation gated on explicit source coding, units and visit verification."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .reporting import savefig


def clinical(run, spec_path, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(spec_path).read_text())
    required = ["cognitive_status", "APOE4", "sex", "T2D", "HbA1c"]
    missing = [
        name
        for name in required
        if name not in spec.get("variables", {})
        or not spec["variables"][name].get("verified", False)
    ]
    if (
        not spec.get("visit_alignment_verified")
        or not spec.get("source_verified")
        or missing
    ):
        status = dict(
            status="blocked_by_unverified_source",
            unverified_targets=missing,
            reason="Clinical visit alignment, coding and units must be verified before interpreting associations; no inferential tests were run.",
        )
        (output / "clinical_status.json").write_text(json.dumps(status, indent=2))
        return status
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest["settings"].get(
        "validation_only"
    ):
        raise ValueError("Clinical inference requires a completed non-pilot run")
    from statsmodels.api import OLS, add_constant
    from statsmodels.stats.multitest import multipletests

    source = Path(spec["source_csv"])
    targets = pd.read_csv(source)
    if targets.duplicated(["cohort", "scan_id"]).any():
        raise ValueError("Duplicate clinical scan keys")
    for name, field in spec["variables"].items():
        if name == "T2D" and not field.get("type2_definition_verified"):
            raise ValueError("General diabetes cannot be assumed to be T2D")
        if name == "HbA1c" and field.get("unit") not in {"percent", "mmol/mol"}:
            raise ValueError("Verify HbA1c unit")
        original = targets[field["column"]]
        if field["kind"] == "categorical":
            mapping = {str(k).strip(): v for k, v in field["mapping"].items()}
            unknown = set(original.dropna().astype(str).str.strip()) - set(mapping)
            if unknown:
                raise ValueError(
                    "Unmapped clinical codes for " + name + ": " + str(unknown)
                )
            targets[name] = original.astype("string").str.strip().map(mapping)
        else:
            targets[name] = pd.to_numeric(original, errors="raise")
    collected = []
    availability = []
    for p in run.glob("fold*/*/*/*/predictions_private.csv"):
        d = pd.read_csv(p)
        d = d.merge(
            targets[["cohort", "scan_id"] + list(spec["variables"])],
            on=["cohort", "scan_id"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_clinical"),
        )
        d["mode"] = p.parts[-4]
        d["modality"] = p.parts[-3]
        d["learner"] = p.parts[-2]
        collected.append(d)
    if not collected:
        raise ValueError("No held-out predictions")
    all_rows = pd.concat(collected)
    tests = []
    skipped = []
    for (mode, modality, learner), d in all_rows.groupby(
        ["mode", "modality", "learner"]
    ):
        for target, field in spec["variables"].items():
            # Clinical sex is kept separate from the acquisition metadata sex covariate.
            column = target + "_clinical" if target == "sex" else target
            for cohort, part in d.groupby("cohort"):
                availability.append(
                    dict(
                        mode=mode,
                        modality=modality,
                        learner=learner,
                        cohort=cohort,
                        target=target,
                        total_scans=len(part),
                        available_scans=int(part[column].notna().sum()),
                        available_participants=part.loc[
                            part[column].notna(), "participant_group"
                        ].nunique(),
                    )
                )
            frame = d.dropna(subset=[column, "age", "sex", "cBAG"]).copy()
            if frame.participant_group.nunique() < 20:
                skipped.append(
                    dict(
                        mode=mode,
                        modality=modality,
                        learner=learner,
                        target=target,
                        reason="fewer_than_20_complete_case_participants",
                    )
                )
                continue
            covars = pd.DataFrame({"age": frame.age}, index=frame.index)
            covars = pd.concat(
                [
                    covars,
                    pd.get_dummies(
                        frame.cohort, prefix="cohort", drop_first=True, dtype=float
                    ),
                ],
                axis=1,
            )
            if target != "sex":
                covars = pd.concat(
                    [
                        covars,
                        pd.get_dummies(
                            frame.sex, prefix="sex", drop_first=True, dtype=float
                        ),
                    ],
                    axis=1,
                )
            if field["kind"] == "categorical":
                levels = [
                    level for level in field["levels"] if frame[column].eq(level).any()
                ]
                if len(levels) < 2 or field["levels"][0] not in levels:
                    skipped.append(
                        dict(
                            mode=mode,
                            modality=modality,
                            learner=learner,
                            target=target,
                            reason="insufficient_levels_or_absent_reference",
                        )
                    )
                    continue
                categories = pd.Categorical(frame[column], categories=levels)
                design = pd.get_dummies(
                    categories, prefix="target", drop_first=True, dtype=float
                )
                design.index = frame.index
            else:
                design = pd.DataFrame(
                    {"target": frame[column].astype(float)}, index=frame.index
                )
            design = pd.concat([covars, design], axis=1)
            design = add_constant(design, has_constant="add").astype(float)
            if np.linalg.matrix_rank(design) != design.shape[1]:
                skipped.append(
                    dict(
                        mode=mode,
                        modality=modality,
                        learner=learner,
                        target=target,
                        reason="rank_deficient_design",
                    )
                )
                continue
            fit = OLS(frame.cBAG.astype(float), design).fit(
                cov_type="cluster",
                cov_kwds={"groups": frame.participant_group, "use_correction": True},
                use_t=True,
            )
            ci = fit.conf_int()
            for term in [x for x in design if x.startswith("target")]:
                tests.append(
                    dict(
                        mode=mode,
                        modality=modality,
                        learner=learner,
                        target=target,
                        term=term,
                        n_scans=len(frame),
                        n_participants=frame.participant_group.nunique(),
                        coefficient=fit.params[term],
                        CI_low=ci.loc[term, 0],
                        CI_high=ci.loc[term, 1],
                        p=fit.pvalues[term],
                    )
                )
            fig, ax = plt.subplots(figsize=(5.5, 4))
            if field["kind"] == "categorical":
                levels = field["levels"]
                groups = [
                    frame.loc[frame[column].eq(level), "cBAG"].to_numpy()
                    for level in levels
                ]
                ax.boxplot(groups, showfliers=False)
                ax.set_xticks(range(1, len(levels) + 1), levels)
            else:
                ax.scatter(frame[column], frame.cBAG, s=9, alpha=0.4)
                ax.set_xlabel(target + " (" + field.get("unit", "") + ")")
            ax.set_ylabel("OOF cBAG")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            savefig(fig, output / f"{mode}_{modality}_{learner}_{target}")
        # Age itself is a continuous validation target; no new correction is fitted.
        frame = d.dropna(subset=["age", "sex", "cBAG"]).copy()
        if frame.participant_group.nunique() >= 20:
            design = add_constant(
                pd.concat(
                    [
                        frame[["age"]],
                        pd.get_dummies(
                            frame[["cohort", "sex"]], drop_first=True, dtype=float
                        ),
                    ],
                    axis=1,
                ),
                has_constant="add",
            ).astype(float)
            if np.linalg.matrix_rank(design) == design.shape[1]:
                fit = OLS(frame.cBAG.astype(float), design).fit(
                    cov_type="cluster",
                    cov_kwds={
                        "groups": frame.participant_group,
                        "use_correction": True,
                    },
                    use_t=True,
                )
                ci = fit.conf_int()
                tests.append(
                    dict(
                        mode=mode,
                        modality=modality,
                        learner=learner,
                        target="chronological_age",
                        term="age",
                        n_scans=len(frame),
                        n_participants=frame.participant_group.nunique(),
                        coefficient=fit.params["age"],
                        CI_low=ci.loc["age", 0],
                        CI_high=ci.loc["age", 1],
                        p=fit.pvalues["age"],
                    )
                )
        # Continuous age dependence is checked within cohort, without fitting a new correction.
        for cohort, part in d.groupby("cohort"):
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.scatter(part.age, part.cBAG, s=9, alpha=0.4)
            ax.set(xlabel="Chronological age", ylabel="OOF cBAG")
            fig.tight_layout()
            savefig(fig, output / f"age_{mode}_{modality}_{learner}_{cohort}")
    pd.DataFrame(availability).to_csv(output / "availability.csv", index=False)
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["q_BH_all_exploratory_tests"] = multipletests(tests.p, method="fdr_bh")[1]
    tests.to_csv(output / "associations.csv", index=False)
    pd.DataFrame(skipped).to_csv(output / "skipped_tests.csv", index=False)
    status = dict(
        status="complete_exploratory",
        skipped_tests=len(skipped),
        multiplicity="BH across all tests in this invocation",
        covariates="age + cohort + sex except when sex is the target",
        caveat="Complete-case observational associations; no causal or diagnostic claim. Education/other covariates require a prespecified extension.",
    )
    (output / "clinical_status.json").write_text(json.dumps(status, indent=2))
    return status
