# Private input specification

`python code/run_pipeline.py prepare --root INPUT_ROOT --output OUTPUT` reads the following files. Inputs are not shipped. Prepare uses archived eligible-scan universes, not raw DICOM/NIfTI discovery. Existing exclusions in those universes remain a limitation.

## Raw connectome manifests

| Modality | Path relative to INPUT_ROOT | Required columns |
| --- | --- | --- |
| DWI | `results/final/DWI/metadata/metadata_used_for_joint_combat.csv` | `DATASET`, `__id__`, `AGE`, `SEX`, `DATASET_SITE`, `filepath` |
| fMRI | `results/final/fMRI/evaluation_inputs/fMRI_metadata_aligned.csv` | `DATASET`, `connectome_key`, `Age`, `Sex`, `DATASET_SITE`, `file_path` |

Despite the DWI manifest filename, the `filepath` column must point to raw connectome matrices, not globally adjusted features. These filenames identify the existing eligible-scan registry only.

- `DATASET`: cohort name; ADNI, ADRC, HABS or ADDecode (normalized to uppercase).
- `__id__` / `connectome_key`: unique scan/visit key within each cohort.
- Age: finite numeric years; sex: harmonizable labels such as F/M or Female/Male.
- `DATASET_SITE`: acquisition batch identifier; sufficient CN participant support is required.
- Matrix path: absolute, or relative to INPUT_ROOT. A relocated dataset needs valid paths.

Each matrix is a headerless numeric 84-by-84 CSV with identical, verified ROI ordering. The loader symmetrizes it and extracts 3,486 upper-triangle edges. DWI values must be finite and nonnegative. fMRI values must be finite Pearson correlations in [-1,1]. The fixed DWI log1p and fMRI clipped Fisher-z transforms are applied in the training workflow.

## Cohort clinical linkage tables

| Cohort | Path relative to INPUT_ROOT | Participant-ID column |
| --- | --- | --- |
| ADNI | `data/final/ADNI/ADNI_harmonized_metadata(in).csv` | `PTID` |
| HABS | `data/final/HABS/HABS_harmonized_metadata(in).csv` | `Subject` |
| ADRC | `data/final/ADRC/ADRC_harmonized_metadata_merged.csv` | `PTID` |
| ADDecode | `data/final/ADDecode/AD_DECODE_harmonized_metadata.csv` | `ID` |

Each source also needs `CONNECTOME_KEY_USED_FOR_INTERSECTION` and `DX_Label_harmonized`. Only cohorts present in the modality manifest require a source table. Join keys are normalized to uppercase and stripped of surrounding whitespace. Duplicate clinical scan keys are excluded as ambiguous; the loader never selects their first row.

Current CN reference labels are NORMAL, CN, CU, HC, COGNITIVELY NORMAL and COGNITIVELY UNIMPAIRED. These mappings describe implemented eligibility; they do not replace source diagnosis verification. Source assessment/acquisition dates still require review.

## Pairing and support

DWI and fMRI are paired one-to-one on cohort and scan key. Participant group, acquisition batch, diagnosis and sex must agree; ages must agree within the configured tolerance (default 0.01 years). Remaining pairs require minimum CN participant support per batch (default five at preparation). Inner/outer training and stopping subsets have additional feasibility checks. Different failure reasons are recorded explicitly.

The current development registry retains 919 DWI scans, 575 fMRI scans and 318 paired scans. Of 575 common keys, 250 fail age agreement and seven further pairs fail CN batch support. Matched CN evaluation uses 237 scans / 189 participants. These are archived-run counts, not requirements imposed on a new dataset.

## Prepared outputs

Preparation saves raw edge arrays, canonical metadata, eligibility/pairing exclusions, sample counts and source hashes. Participant-level files contain `_private` in their names and remain under ignored `results/`. Train never expects globally harmonized arrays. Preserve manifest hashes and use a new preparation directory when inputs change.
