"""Importable sklearn transforms for reproducible, serializable fusion models."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class ModalityBlockWeight(BaseEstimator, TransformerMixin):
    """Keep standardized DWI features and multiply fMRI features by a weight.

    Must follow StandardScaler: scaling again afterwards cancels the weight.
    The weight is a relative feature scale, not a percentage contribution.
    """

    def __init__(self, fmri_weight=1.0, n_dwi_features=3486):
        self.fmri_weight = fmri_weight
        self.n_dwi_features = n_dwi_features

    def fit(self, X, y=None):
        X = check_array(X)
        if not np.isfinite(self.fmri_weight) or self.fmri_weight < 0:
            raise ValueError("fMRI weight must be finite and nonnegative")
        if X.shape[1] != 2 * self.n_dwi_features:
            raise ValueError("Expected concatenated equal-sized DWI and fMRI blocks")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X, copy=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from training data")
        X[:, self.n_dwi_features:] *= self.fmri_weight
        return X
