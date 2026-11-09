"""Frozen participant fold plans shared by all modalities/models."""
import hashlib
import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from run_foldwise_DWI_Ridge import supported_splits


def identity_hash(meta):
    return hashlib.sha256(
        meta[["cohort", "scan_id", "participant_group", "batch", "is_cn"]]
        .to_csv(index=False)
        .encode()
    ).hexdigest()


def stopping_split(meta, indices, seed):
    local = meta.iloc[indices].reset_index(drop=True)
    for offset in range(1000):
        a, b = next(
            GroupShuffleSplit(
                n_splits=1, test_size=0.2, random_state=seed + offset
            ).split(local, groups=local.participant_group)
        )
        support = local.iloc[a].groupby("batch").size()
        if all(support.get(x, 0) >= 2 for x in local.batch.unique()):
            train, stop = indices[a], indices[b]
            if not set(meta.participant_group.iloc[train]).isdisjoint(
                meta.participant_group.iloc[stop]
            ):
                raise AssertionError("Stopping leakage")
            return train, stop
    raise ValueError("No supported early-stopping partition; inspect sparse batches")


def make_plan(meta, outer_folds=5, inner_folds=3, seed=42):
    outer, actual = supported_splits(meta, outer_folds, seed)
    plan = dict(
        identity_sha256=identity_hash(meta),
        seed=seed,
        actual_outer_seed=actual,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        folds=[],
    )
    for fold, (all_train, test) in enumerate(outer, 1):
        train = all_train[meta.is_cn.iloc[all_train].to_numpy()]
        inner, inner_seed = supported_splits(
            meta.iloc[train].reset_index(drop=True), inner_folds, seed + 100 * fold
        )
        entries = []
        for inner_fold, (a, b) in enumerate(inner, 1):
            fit, valid = train[a], train[b]
            early_fit, stop = stopping_split(meta, fit, seed + fold * 1000 + inner_fold)
            entries.append(
                dict(
                    train=fit.tolist(),
                    validation=valid.tolist(),
                    early_train=early_fit.tolist(),
                    early_stop=stop.tolist(),
                )
            )
        plan["folds"].append(
            dict(
                fold=fold,
                train_all=all_train.tolist(),
                train_CN=train.tolist(),
                test=test.tolist(),
                inner_seed=inner_seed,
                inner=entries,
            )
        )
    return plan


def audit_plan(plan, meta):
    if identity_hash(meta) != plan["identity_sha256"]:
        raise ValueError("Fold plan does not match prepared data order/identity")
    seen = []
    for fold in plan["folds"]:
        train, test = set(fold["train_CN"]), set(fold["test"])
        if not meta.iloc[list(train)].is_cn.all():
            raise AssertionError("Non-CN normative fitting")

        def disjoint(a, b):
            if not set(meta.participant_group.iloc[list(a)]).isdisjoint(
                meta.participant_group.iloc[list(b)]
            ):
                raise AssertionError("Participant overlap")

        disjoint(fold["train_all"], test)
        seen += list(test)
        held = []
        for inner in fold["inner"]:
            a, b = set(inner["train"]), set(inner["validation"])
            held += list(b)
            if (a | b) != train or a & b:
                raise AssertionError("Invalid inner partition")
            disjoint(a, b)
            disjoint(inner["early_train"], inner["early_stop"])
            if set(inner["early_train"]) | set(inner["early_stop"]) != a:
                raise AssertionError("Stopping split not within inner training")
        if sorted(held) != sorted(train):
            raise AssertionError("Incomplete inner OOF coverage")
    if sorted(seen) != list(range(len(meta))):
        raise AssertionError("Incomplete outer OOF coverage")
    return True


def subsample_groups(meta, indices, fraction, seed):
    if fraction == 1:
        return np.asarray(indices)
    if not 0 < fraction <= 1:
        raise ValueError("Fraction outside (0,1]")
    rng = np.random.default_rng(seed)
    local = meta.iloc[indices]
    groups = local.participant_group.unique()
    target = int(np.ceil(len(groups) * fraction))
    chosen = set()
    for _, part in local.groupby("batch"):
        candidates = part.participant_group.unique()
        if len(candidates) < 2:
            raise ValueError("Insufficient participants in batch")
        chosen.update(rng.choice(candidates, 2, replace=False))
    if len(chosen) > target:
        raise ValueError("Requested fraction cannot support all batches")
    remaining = [x for x in groups if x not in chosen]
    chosen.update(rng.choice(remaining, target - len(chosen), replace=False))
    return np.asarray(indices)[local.participant_group.isin(chosen).to_numpy()]
