"""Independent, frozen training-only transforms for each modality."""
import numpy as np
from harmonization import CombatPreprocessor
from foldwise_fmri import FmriPreprocessor

MODES = {"none": (False, False), "batch": (True, False), "age_mean": (True, True)}


class MultiPreprocessor:
    def __init__(self, mode="age_mean"):
        if mode not in MODES:
            raise ValueError("Unknown harmonization mode")
        self.mode = mode

    def fit(self, inputs, batches, training_ages):
        self.transforms = {}
        h, a = MODES[self.mode]
        for modality, X in inputs.items():
            cls = CombatPreprocessor if modality == "DWI" else FmriPreprocessor
            self.transforms[modality] = cls(h, a).fit(
                X, batches, training_ages if a else None
            )
        return self

    def transform(self, inputs, batches):
        if set(inputs) != set(self.transforms):
            raise ValueError("Modalities differ from training")
        return {
            m: prep.transform(inputs[m], batches) for m, prep in self.transforms.items()
        }

    def matrix(self, inputs, batches):
        features = self.transform(inputs, batches)
        return np.concatenate(list(features.values()), axis=1)

    def graph_values(self, inputs, batches):
        """Restore edge positions after training-only QC; zero is the scaled mean."""
        features = self.transform(inputs, batches)
        full = {}
        for m, prep in self.transforms.items():
            a = np.zeros_like(inputs[m], dtype=np.float32)
            a[:, prep.mask_] = features[m]
            full[m] = a
        return full
