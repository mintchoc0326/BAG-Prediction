# BAG Connectome Harmonization

This README documents the fMRI and DWI harmonization workflows used in the BAG project.

The two modalities are documented separately for reproducibility.

## Contents

1. fMRI Harmonization
2. DWI Harmonization
3. Pre/Post-Harmonization Evaluation
4. Reproducibility Notes

# 1. fMRI Harmonization

## 1.1 Overview

The fMRI harmonization workflow includes three cohorts:

- ADNI
- HABS
- ADRC

Each fMRI connectome is represented as an 84 × 84 functional connectivity matrix.

Because the connectivity matrices are symmetric, only the upper triangular portion excluding the diagonal is used for harmonization. This results in 3,486 connectivity edges per scan.

The goal of harmonization is to reduce technical variability associated with both dataset and acquisition site while preserving biologically relevant variation.

---

## 1.2 Input Connectomes and Metadata

### ADNI

Connectome directory:

`/mnt/newStor/paros/paros_WORK/avery/data/harmonization/ADNI/connectomes/fMRI`

Metadata file:

`/mnt/newStor/paros/paros_WORK/avery/data/harmonization/ADNI/ADNI_harmonized_metadata(in).csv`

### HABS

Connectome directory:

`/mnt/newStor/paros/paros_WORK/avery/data/harmonization/HABS/connectomes/fMRI`

Metadata file:

`/mnt/newStor/paros/paros_WORK/avery/data/harmonization/HABS/HABS_harmonized_metadata(in).csv`

### ADRC

Connectome directory:

`/mnt/newStor/paros/paros_WORK/avery/data/harmonization/ADRC/connectomes/fMRI`

Metadata file:

`/mnt/newStor/paros/paros_WORK/avery/data/harmonization/ADRC/ADRC_harmonized_metadata_merged.csv`

### Site Information

Additional site information used during harmonization is obtained from:

`/mnt/newStor/paros/paros_WORK/avery/data/harmonization/all_covars_aligned.csv`

---

## 1.3 Variables and Covariates

### Technical batch variable

The main technical batch variable is:

`DATASET_SITE`

It is constructed by combining the dataset and acquisition site:

`DATASET_SITE = DATASET + "__" + SITE`

For example:

- `ADNI__003`
- `ADNI__027`

This allows site effects within individual datasets and differences across datasets to be modeled jointly in the same NeuroCombat harmonization.

### Biological covariates

The biological covariates used during harmonization are:

- Age — continuous
- Sex — categorical
- Diagnosis — categorical, when appropriate for the available data and model design

These covariates are included so that biologically relevant variation can be preserved while technical batch effects are adjusted.

---

## 1.4 Preprocessing and Quality Control

Before NeuroCombat harmonization, the following preprocessing and quality-control steps are performed.

### Step 1 — Connectome Discovery

The script searches the cohort-specific directories for candidate fMRI connectome files.

Non-connectome files, including time-series, metadata, demographic, and DWI files, are excluded.

### Step 2 — Matrix Quality Control

Each connectome must:

- be readable;
- contain numerical values;
- be a two-dimensional square matrix;
- have dimensions of 84 × 84; and
- contain only finite values.

### Step 3 — Subject and Scan ID Standardization

Connectome filenames and metadata identifiers are converted into standardized scan IDs so that each connectome can be matched to the correct metadata.

The standardized IDs are cohort-specific.

Examples include:

- ADNI: `R####_y#`
- HABS: `H####_y#`
- ADRC: `D####`

### Step 4 — Metadata Matching

Each valid connectome is matched with its corresponding metadata.

The matched information includes:

- Dataset
- Site
- Age
- Sex
- Diagnosis

Only scans with the required metadata are retained for harmonization.

### Step 5 — Fisher-Z Transformation

The fMRI connectomes contain Pearson correlation coefficients.

Before harmonization, the connectivity values are transformed using the Fisher-Z transformation:

`z = arctanh(r)`

where:

- `r` = original Pearson correlation coefficient
- `z` = Fisher-Z transformed connectivity value

### Step 6 — Feature Extraction

The upper triangular portion of each 84 × 84 connectivity matrix is extracted, excluding the diagonal.

This converts each connectome into a vector containing 3,486 connectivity edges.

The resulting feature matrix has the structure:

`number of scans × 3,486 edges`

### Step 7 — Batch Quality Control

After constructing `DATASET_SITE`, technical batches are checked for sample size.

Singleton batches containing only one scan are excluded before NeuroCombat because reliable batch-effect parameters cannot be estimated from a single observation.

Small batches are retained but flagged for interpretation.

## 1.5 NeuroCombat Harmonization

Harmonization is performed using NeuroCombat.

The harmonization model is applied jointly across ADNI, HABS, and ADRC.

The technical batch variable is:

`DATASET_SITE`

The biological covariates are:

- Age
- Sex
- Diagnosis, when appropriate

The purpose of using `DATASET_SITE` rather than dataset alone is to account for two sources of technical variation simultaneously:

1. within-dataset differences among acquisition sites; and
2. between-dataset differences among ADNI, HABS, and ADRC.

Therefore, all eligible scans are entered into a single joint harmonization model rather than performing separate within-dataset and between-dataset harmonization steps.

NeuroCombat is configured to adjust both the location and scale of the connectivity features while preserving the modeled biological covariates.

---

## 1.6 Harmonization Code

Main fMRI harmonization script:

`run_fMRI_harmonization.py`

Repository location:

`[TO BE ADDED]`

Server code location:

`[TO BE ADDED]`

---

## 1.7 Harmonization Outputs

Main fMRI results directory:

`/mnt/newStor/paros/paros_WORK/avery/results/ADNI_HABS_ADRC_fMRI_joint_site_harmonization`

Harmonized connectome directory:

`/mnt/newStor/paros/paros_WORK/avery/results/ADNI_HABS_ADRC_fMRI_joint_site_harmonization/harmonized_connectomes`

The harmonized connectomes are saved separately by cohort:

- ADNI
- HABS
- ADRC

After NeuroCombat, the harmonized connectivity values are converted from Fisher-Z space back to correlation space.

The inverse Fisher-Z transformation is:

`r = tanh(z)`

The 3,486 harmonized connectivity edges are then reconstructed into symmetric 84 × 84 matrices.

The diagonal elements are set to 1.

A manifest file is also generated to record the relationship between each scan and its saved harmonized connectome.

## 1.8 Pre- vs. Post-Harmonization Diagnostics

The effectiveness of harmonization is evaluated by comparing the connectome features before and after NeuroCombat.

The main goal is to determine whether technical dataset/site information is reduced after harmonization while biologically relevant information is preserved.

### PCA Visualization

Principal Component Analysis (PCA) is performed before and after harmonization.

The samples are colored by dataset or acquisition site to visually examine whether technical clustering is reduced after NeuroCombat.

### Dataset Classification

A classifier is trained to predict the source dataset from the connectome features.

Results:

| Metric | Before Harmonization | After Harmonization |
|---|---:|---:|
| Balanced Accuracy | TBD | TBD |
| ROC AUC | TBD | TBD |

### Within-Cohort Site Classification

Site classification is evaluated within individual cohorts when sufficient site sample sizes are available.

Results:

| Cohort | Before Harmonization | After Harmonization |
|---|---:|---:|
| ADNI | TBD | TBD |
| HABS | TBD | TBD |
| ADRC | N/A / TBD | N/A / TBD |

### Biological Information Preservation

Biological-variable classification is also compared before and after harmonization.

This analysis is used to determine whether biologically relevant connectivity information remains available after removal of technical effects.

Results:

| Variable | Before Harmonization | After Harmonization |
|---|---:|---:|
| Sex | TBD | TBD |
| Diagnosis | TBD | TBD |

The final values will be updated after the harmonization evaluation is finalized.

## 1.9 Exclusions and Relevant Notes

The following exclusions and preprocessing decisions are applied in the fMRI workflow:

- Only valid 84 × 84 numerical connectivity matrices are included.
- Time-series and other non-connectome CSV files are excluded.
- Connectomes that cannot be matched to metadata are excluded.
- Scans missing required harmonization covariates are excluded.
- Singleton `DATASET_SITE` batches are excluded before NeuroCombat.
- Small technical batches are retained but should be interpreted cautiously.
- Pearson correlation coefficients are Fisher-Z transformed before harmonization.
- Harmonization is performed on the 3,486 upper-triangular connectivity edges rather than directly on the 84 × 84 matrices.
- ADNI, HABS, and ADRC are included in one joint NeuroCombat model.
- `DATASET_SITE` is used to model both within-dataset site effects and between-dataset technical differences.
