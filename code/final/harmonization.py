"""Shared train/apply ComBat implementation for predictive evaluation.

The default representation is DWI log1p. Subclasses supply modality transforms.
Only training ages may inform additive site estimates. No held-out age, sex,
diagnosis, genotype or metabolic covariate is accepted at transformation time.
"""
import contextlib
import io
import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from sklearn.preprocessing import StandardScaler


class CombatPreprocessor:
    """Frozen ComBat + scaler; batch-only or training-age additive mode."""

    def __init__(self, harmonize=True, age_trained_mean_only=False):
        self.harmonize = harmonize
        self.age_trained_mean_only = age_trained_mean_only
        if age_trained_mean_only and not harmonize:
            raise ValueError("Mean-only mode requires harmonization")

    def _features(self, X):
        if not np.isfinite(X).all() or (X < 0).any():
            raise ValueError("Invalid DWI edges; no test-driven imputation")
        return np.log1p(X)

    def fit(self, X, batches, training_ages=None):
        X = np.asarray(X, dtype=float)
        batches = np.asarray(batches, dtype=str)
        if X.ndim != 2 or len(batches) != len(X):
            raise ValueError("Invalid raw training features or batches")
        self.n_input_features_ = X.shape[1]
        self.batches_, counts = np.unique(batches, return_counts=True)
        if len(self.batches_) < 2 or np.min(counts) < 2:
            raise ValueError(
                "Known-batch comparison requires >=2 batches and >=2 training rows per batch"
            )
        z = self._features(X)
        residual = z.copy()
        for batch in self.batches_:
            ix = batches == batch
            residual[ix] -= z[ix].mean(axis=0)
        self.mask_ = (z.std(axis=0) > 1e-10) & (np.mean(residual**2, axis=0) > 1e-12)
        if self.mask_.sum() < 2:
            raise ValueError("Insufficient training-variable edges")
        if self.harmonize:
            covars = pd.DataFrame({"batch": batches})
            continuous = []
            if self.age_trained_mean_only:
                ages = np.asarray(training_ages, dtype=float)
                if ages.shape != (len(X),) or not np.isfinite(ages).all():
                    raise ValueError("Mean-only fit requires training-only ages")
                design = np.column_stack(
                    [(batches == b).astype(float) for b in self.batches_] + [ages]
                )
                if np.linalg.matrix_rank(design) != design.shape[1]:
                    raise ValueError(
                        "Training age is confounded with batch; design is rank deficient"
                    )
                covars["training_age"] = ages
                continuous = ["training_age"]
            elif training_ages is not None:
                raise ValueError("Batch-only fit does not accept ages")
            with contextlib.redirect_stdout(io.StringIO()):
                result = neuroCombat(
                    dat=z[:, self.mask_].T,
                    covars=covars,
                    batch_col="batch",
                    categorical_cols=[],
                    continuous_cols=continuous,
                    eb=True,
                    parametric=True,
                    mean_only=self.age_trained_mean_only,
                )
            estimates = result["estimates"]
            self.mean_ = np.asarray(estimates["stand.mean"])[:, 0].copy()
            self.scale_ = np.sqrt(
                np.asarray(estimates["var.pooled"]).reshape(-1)
            ).copy()
            self.gamma_ = np.asarray(estimates["gamma.star"]).copy()
            self.delta_ = np.asarray(estimates["delta.star"]).copy()
            self.batches_ = np.asarray(estimates["batches"], dtype=str)
            if not np.isfinite(self.delta_).all() or (self.delta_ <= 0).any():
                raise ValueError("Invalid ComBat batch variances")
            trained = self._adjust(z[:, self.mask_], batches)
            self.training_transform_max_error_ = float(
                np.max(np.abs(trained - result["data"].T))
            )
            if not np.allclose(trained, result["data"].T, rtol=1e-7, atol=1e-8):
                raise AssertionError("Train/apply ComBat mismatch")
        else:
            trained = z[:, self.mask_]
            self.training_transform_max_error_ = 0.0
        self.scaler_ = StandardScaler().fit(trained)
        return self

    def _adjust(self, z, batches):
        levels = {b: i for i, b in enumerate(self.batches_)}
        unknown = set(batches) - set(levels)
        if unknown:
            raise ValueError(f"Unseen batches unsupported: {sorted(unknown)}")
        index = np.array([levels[b] for b in batches])
        if self.age_trained_mean_only:
            # With delta=1: age terms cancel, x_H = x - sigma * gamma_batch.
            if not np.allclose(self.delta_, 1):
                raise AssertionError("Mean-only mode must have unit batch scales")
            return z - self.gamma_[index] * self.scale_
        standardized = (z - self.mean_) / self.scale_
        return (
            (standardized - self.gamma_[index]) / np.sqrt(self.delta_[index])
        ) * self.scale_ + self.mean_

    def transform(self, X, batches, standardize=True):
        X = np.asarray(X, dtype=float)
        batches = np.asarray(batches, dtype=str)
        if (
            X.ndim != 2
            or X.shape[1] != self.n_input_features_
            or len(batches) != len(X)
        ):
            raise ValueError("Input alignment/shape mismatch")
        z = self._features(X)[:, self.mask_]
        if self.harmonize:
            z = self._adjust(z, batches)
        return self.scaler_.transform(z) if standardize else z
