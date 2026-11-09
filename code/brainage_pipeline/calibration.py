"""Training-reference BAG calibration; application never estimates coefficients."""
import numpy as np


class BiasCorrection:
    def __init__(self, kind="cohort"):
        if kind not in {"global", "cohort"}:
            raise ValueError("Unknown calibration type")
        self.kind = kind

    def fit(self, age, oof_prediction, cohorts):
        age = np.asarray(age, dtype=float)
        prediction = np.asarray(oof_prediction, dtype=float)
        cohorts = np.asarray(cohorts, dtype=str)
        if not np.isfinite(age).all() or not np.isfinite(prediction).all():
            raise ValueError("Invalid training calibration")
        self.global_coef_ = np.polyfit(age, prediction - age, 1)
        self.coef_ = {}
        self.fallback_ = []
        for cohort in np.unique(cohorts):
            ix = cohorts == cohort
            if self.kind == "cohort" and ix.sum() >= 10 and np.std(age[ix]) > 1e-6:
                self.coef_[cohort] = np.polyfit(age[ix], prediction[ix] - age[ix], 1)
            else:
                self.coef_[cohort] = self.global_coef_.copy()
                self.fallback_.append(cohort)
        return self

    def apply(self, prediction, age, cohorts):
        age = np.asarray(age, dtype=float)
        cohorts = np.asarray(cohorts, dtype=str)
        missing = set(cohorts) - set(self.coef_)
        if missing:
            raise ValueError("Unseen calibration cohorts: " + str(missing))
        trend = np.array([np.polyval(self.coef_[c], a) for c, a in zip(cohorts, age)])
        return np.asarray(prediction) - age - trend
