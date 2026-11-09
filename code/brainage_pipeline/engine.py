"""Shared nested evaluator for Ridge/GINE and unimodal/early-fusion inputs."""
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from run_foldwise_DWI_Ridge import metrics, probe
from .data import load
from .preprocessing import MultiPreprocessor
from .calibration import BiasCorrection
from .splits import make_plan, audit_plan, subsample_groups


def subset(raw, indices):
    return {m: X[indices] for m, X in raw.items()}


def evaluate_task(raw, meta, fold, learner, mode, settings, output, seed):
    output.mkdir(parents=True, exist_ok=True)
    train = np.array(fold["train_CN"])
    test = np.array(fold["test"])
    cache = {}
    prep_audit = []
    histories = []
    configs = (
        [{"alpha": float(a)} for a in settings["ridge_alphas"]]
        if learner == "ridge"
        else settings["gine_grid"]
    )
    candidates = []
    candidate_predictions = []
    candidate_epochs = []

    def preprocessing(indices):
        key = tuple(int(i) for i in indices)
        if key not in cache:
            prep = MultiPreprocessor(mode).fit(
                subset(raw, indices), meta.batch.iloc[indices], meta.age.iloc[indices]
            )
            cache[key] = prep
            prep_audit.append(
                dict(
                    training_indices=list(key),
                    retained_edges={
                        m: int(p.mask_.sum()) for m, p in prep.transforms.items()
                    },
                    max_train_apply_error=max(
                        p.training_transform_max_error_
                        for p in prep.transforms.values()
                    ),
                    state_hash=joblib.hash(prep),
                )
            )
        return cache[key]

    def features(prep, indices, config):
        if learner == "ridge":
            return prep.matrix(subset(raw, indices), meta.batch.iloc[indices])
        from .gnn import graphs

        return graphs(
            subset(raw, indices),
            prep.graph_values(subset(raw, indices), meta.batch.iloc[indices]),
            config["threshold"],
            prep.transforms,
        )

    for c, config in enumerate(configs):
        oof = np.full(len(meta), np.nan)
        epochs = []
        for inner_number, inner in enumerate(fold["inner"], 1):
            it, iv = np.array(inner["train"]), np.array(inner["validation"])
            if learner == "gine":
                from . import gnn

                early, stop = np.array(inner["early_train"]), np.array(
                    inner["early_stop"]
                )
                prep = preprocessing(early)
                model, mean, std, best, history = gnn.fit(
                    features(prep, early, config),
                    meta.age.iloc[early].to_numpy(),
                    config,
                    settings["training"],
                    seed + inner_number,
                    features(prep, stop, config),
                    meta.age.iloc[stop].to_numpy(),
                )
                epochs.append(best)
                for row in history:
                    histories.append(
                        dict(
                            candidate=c,
                            inner_fold=inner_number,
                            stage="early_stopping",
                            selected_epoch=best,
                            **row,
                        )
                    )
                del model
                prep = preprocessing(it)
                model, mean, std, _, history = gnn.fit(
                    features(prep, it, config),
                    meta.age.iloc[it].to_numpy(),
                    config,
                    settings["training"],
                    seed + inner_number,
                    epochs=best,
                )
                prediction = gnn.predict(model, features(prep, iv, config), mean, std)
                del model
            else:
                prep = preprocessing(it)
                model = Ridge(alpha=config["alpha"], solver="lsqr", tol=1e-6).fit(
                    features(prep, it, config), meta.age.iloc[it]
                )
                prediction = model.predict(features(prep, iv, config))
            oof[iv] = prediction
        if not np.isfinite(oof[train]).all():
            raise AssertionError("Inner OOF coverage failure")
        score = float(np.mean(abs(oof[train] - meta.age.iloc[train].to_numpy())))
        candidates.append(
            dict(
                candidate=c,
                inner_oof_MAE=score,
                config=json.dumps(config, sort_keys=True),
            )
        )
        candidate_predictions.append(oof[train])
        candidate_epochs.append(epochs)
    selected = int(np.argmin([row["inner_oof_MAE"] for row in candidates]))
    config = configs[selected]
    pd.DataFrame(candidates).assign(selected=lambda d: d.candidate.eq(selected)).to_csv(
        output / "selection.csv", index=False
    )
    calibration = pd.DataFrame(
        {"row_index": train, "inner_oof_prediction": candidate_predictions[selected]}
    )
    calibration.to_csv(output / "calibration_private.csv", index=False)
    global_bias = BiasCorrection("global").fit(
        meta.age.iloc[train], candidate_predictions[selected], meta.cohort.iloc[train]
    )
    bias = BiasCorrection(settings.get("calibration", "cohort")).fit(
        meta.age.iloc[train], candidate_predictions[selected], meta.cohort.iloc[train]
    )
    prep = preprocessing(train)
    record = dict(
        learner=learner,
        config=config,
        preprocessor=prep,
        calibration=bias,
        global_calibration=global_bias,
        train_indices=train,
        test_indices=test,
        selected_candidate=selected,
    )
    if learner == "ridge":
        model = Ridge(alpha=config["alpha"], solver="lsqr", tol=1e-6).fit(
            features(prep, train, config), meta.age.iloc[train]
        )
        prediction = model.predict(features(prep, test, config))
        record["model"] = model
        joblib.dump(record, output / "model_private.joblib")
    else:
        import torch
        from . import gnn

        fixed_epochs = max(1, int(np.median(candidate_epochs[selected])))
        model, mean, std, _, history = gnn.fit(
            features(prep, train, config),
            meta.age.iloc[train].to_numpy(),
            config,
            settings["training"],
            seed,
            epochs=fixed_epochs,
        )
        test_graphs = features(prep, test, config)
        train_graphs = features(prep, train, config)
        prediction = gnn.predict(model, test_graphs, mean, std)
        train_embeddings = gnn.predict(model, train_graphs, mean, std, embedding=True)
        test_embeddings = gnn.predict(model, test_graphs, mean, std, embedding=True)
        # Embedding coordinate systems differ by fold: never pool raw coordinates.
        pca = PCA(
            n_components=min(2, len(train_embeddings), train_embeddings.shape[1])
        ).fit(train_embeddings)
        projection = pca.transform(test_embeddings)
        np.savez_compressed(
            output / "embeddings_private.npz",
            train_indices=train,
            test_indices=test,
            train=train_embeddings,
            test=test_embeddings,
            test_projection=projection,
        )
        for row in history:
            histories.append(
                dict(
                    candidate=selected,
                    inner_fold=0,
                    stage="outer_refit",
                    selected_epoch=fixed_epochs,
                    **row,
                )
            )
        pd.DataFrame(histories).to_csv(output / "learning_history.csv", index=False)
        torch.save(
            {
                "state_dict": model.cpu().state_dict(),
                "config": config,
                "node_dim": test_graphs[0].x.shape[1],
                "edge_dim": test_graphs[0].edge_attr.shape[1],
                "age_mean": mean,
                "age_std": std,
                "epochs": fixed_epochs,
            },
            output / "checkpoint_private.pt",
        )
        record.update(
            age_mean=mean,
            age_std=std,
            epochs=fixed_epochs,
            parameters=sum(p.numel() for p in model.parameters()),
            embedding_pca=pca,
        )
        joblib.dump(record, output / "model_private.joblib")
    rows = meta.iloc[test][
        [
            "cohort",
            "scan_id",
            "participant_group",
            "batch",
            "is_cn",
            "clinical_diagnosis",
            "sex",
            "age",
        ]
    ].copy()
    rows["row_index"] = test
    rows["predicted_age"] = prediction
    rows["BAG"] = prediction - rows.age
    rows["cBAG_global"] = global_bias.apply(prediction, rows.age, rows.cohort)
    rows["cBAG"] = bias.apply(prediction, rows.age, rows.cohort)
    rows.to_csv(output / "predictions_private.csv", index=False)
    retention = probe(
        prep.matrix(subset(raw, train), meta.batch.iloc[train]),
        prep.matrix(subset(raw, test), meta.batch.iloc[test]),
        meta.iloc[train].reset_index(drop=True),
        meta.iloc[test].reset_index(drop=True),
    )
    (output / "retention_probes.json").write_text(json.dumps(retention, indent=2))
    (output / "preprocessing_audit_private.json").write_text(
        json.dumps(prep_audit, indent=2)
    )
    cn = rows[rows.is_cn]
    result = dict(
        status="complete",
        selected_config=config,
        selected_candidate=selected,
        calibration_kind=bias.kind,
        calibration_fallback_cohorts=bias.fallback_,
        n_training_CN=len(train),
        n_test=len(test),
        parameters=record.get("parameters"),
        epochs=record.get("epochs"),
        **metrics(cn.age.to_numpy(), cn.predicted_age.to_numpy()),
    )
    (output / "task.json").write_text(json.dumps(result, indent=2))
    return result


def run(prepared, output, settings, population="paired", resume=False, fraction=1.0):
    prepared, output = Path(prepared), Path(output)
    modalities = ("DWI", "fMRI") if population == "paired" else (population,)
    raw, meta = load(prepared, population, modalities)
    package = Path(__file__).parent
    dependencies = list(package.glob("*.py")) + [
        package.parent / "final" / name
        for name in [
            "harmonization.py",
            "foldwise_dwi.py",
            "foldwise_fmri.py",
            "run_foldwise_DWI_Ridge.py",
        ]
    ]
    code_hashes = {
        str(p.relative_to(package.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in dependencies
    }
    fingerprint = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()
    dataset = hashlib.sha256((prepared / "manifest.json").read_bytes()).hexdigest()
    if output.exists() and not resume:
        raise ValueError("Output exists; use a new directory or explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        old = json.loads((output / "manifest.json").read_text())
        if old.get("code_sha256") != code_hashes:
            raise ValueError(
                "Source code changed; resume requires the exact original implementation"
            )
        if (
            old["settings_sha256"] != fingerprint
            or old["prepared_manifest_sha256"] != dataset
            or old["population"] != population
            or old["training_fraction"] != fraction
        ):
            raise ValueError("Resume configuration/data mismatch")
        plan = json.loads((output / "fold_plan_private.json").read_text())
    else:
        plan = make_plan(
            meta, settings["outer_folds"], settings["inner_folds"], settings["seed"]
        )
        if fraction != 1:
            # Rebuild inner plans on the supported participant subsample, keeping outer tests fixed.
            from .splits import stopping_split
            from run_foldwise_DWI_Ridge import supported_splits

            for fold in plan["folds"]:
                tr = subsample_groups(
                    meta, fold["train_CN"], fraction, settings["seed"] + fold["fold"]
                )
                folds, _ = supported_splits(
                    meta.iloc[tr].reset_index(drop=True),
                    settings["inner_folds"],
                    settings["seed"] + 100 * fold["fold"],
                )
                fold["train_CN"] = tr.tolist()
                fold["inner"] = []
                for number, (a, b) in enumerate(folds, 1):
                    fit, stop = stopping_split(
                        meta, tr[a], settings["seed"] + 1000 * fold["fold"] + number
                    )
                    fold["inner"].append(
                        dict(
                            train=tr[a].tolist(),
                            validation=tr[b].tolist(),
                            early_train=fit.tolist(),
                            early_stop=stop.tolist(),
                        )
                    )
        (output / "fold_plan_private.json").write_text(json.dumps(plan, indent=2))
    audit_plan(plan, meta)
    meta.to_csv(output / "metadata_private.csv", index=False)
    manifest = dict(
        status="running",
        code_sha256=code_hashes,
        population=population,
        training_fraction=fraction,
        settings=settings,
        settings_sha256=fingerprint,
        prepared_manifest_sha256=dataset,
        limitations=[
            "Development reanalysis after prior OOF inspection; external validation remains necessary.",
            "Source labels, acquisition dates, ROI order and scanner protocols remain unverified.",
            "Cohort-aware calibration is prespecified for this development run, not chosen using outer-test scores.",
        ],
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summaries = []
    model_modalities = {m: [m] for m in modalities}
    if population == "paired":
        model_modalities["fusion"] = list(modalities)
    source_dir = output / "source_snapshot"
    source_dir.mkdir(exist_ok=True)
    for source in dependencies:
        destination = source_dir / source.relative_to(package.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    for fold in plan["folds"]:
        for mode in settings["harmonization_modes"]:
            for name, mods in model_modalities.items():
                for learner in settings["learners"]:
                    path = output / f'fold{fold["fold"]}' / mode / name / learner
                    start = time.perf_counter()
                    if (path / "task.json").exists():
                        result = json.loads((path / "task.json").read_text())
                    else:
                        result = evaluate_task(
                            {m: raw[m] for m in mods},
                            meta,
                            fold,
                            learner,
                            mode,
                            settings,
                            path,
                            settings["seed"] + fold["fold"],
                        )
                    summaries.append(
                        dict(
                            fold=fold["fold"],
                            mode=mode,
                            modality=name,
                            learner=learner,
                            seconds_this_invocation=time.perf_counter() - start,
                            **result,
                        )
                    )
                    print(
                        f'fold={fold["fold"]} {mode} {name} {learner} MAE={result["MAE"]:.3f}',
                        flush=True,
                    )
                    pd.DataFrame(summaries).to_csv(
                        output / "task_summary.csv", index=False
                    )
    manifest["status"] = "complete"
    manifest["completed_tasks"] = len(summaries)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return pd.DataFrame(summaries)
