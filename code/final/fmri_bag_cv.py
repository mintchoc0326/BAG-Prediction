"""Deterministic participant-grouped folds shared by fMRI brain-age models."""

import numpy as np
from sklearn.model_selection import GroupKFold as SklearnGroupKFold


class GroupKFold(SklearnGroupKFold):
    """Balance scan counts with explicit ties by group name and fold number.

    Many participants have the same number of scans. Explicitly ordering
    these ties avoids relying on NumPy's unstable default count sorting.
    """

    def _iter_test_indices(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Participant groups are required.")
        unique_groups, group_indices = np.unique(groups, return_inverse=True)
        if len(unique_groups) < self.n_splits:
            raise ValueError("Fewer participant groups than folds.")
        group_sizes = np.bincount(group_indices)
        order = np.lexsort((unique_groups, -group_sizes))
        fold_sizes = np.zeros(self.n_splits, dtype=int)
        group_folds = np.empty(len(unique_groups), dtype=int)
        for group_index in order:
            fold = int(np.argmin(fold_sizes))
            group_folds[group_index] = fold
            fold_sizes[fold] += group_sizes[group_index]
        scan_folds = group_folds[group_indices]
        for fold in range(self.n_splits):
            yield np.flatnonzero(scan_folds == fold)
