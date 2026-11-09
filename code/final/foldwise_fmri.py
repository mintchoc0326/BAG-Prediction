"""Signed Pearson-correlation fMRI preprocessing; shared foldwise ComBat core."""
import numpy as np
from harmonization import CombatPreprocessor


class FmriPreprocessor(CombatPreprocessor):
    """Fixed Fisher-z transform before training-only feature QC and ComBat.

    Negative correlations are retained. Exact +/-1 values are clipped using a
    fixed epsilon, not a sample-derived threshold. No demographics in transform.
    """

    def _features(self, X):
        if not np.isfinite(X).all() or np.max(np.abs(X)) > 1 + 1e-6:
            raise ValueError(
                "Invalid fMRI correlations; expected finite values within [-1,1]"
            )
        return np.arctanh(np.clip(X, -1 + 1e-7, 1 - 1e-7))
