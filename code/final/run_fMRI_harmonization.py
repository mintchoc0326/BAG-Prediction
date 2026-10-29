from IPython.display import display
# %%
# GPU
import socket
import torch

print("Host:", socket.gethostname())
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# %%
# ============================================================
# CELL 1 — IMPORTS
# ============================================================

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuroCombat import neuroCombat

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %% [markdown]
# # File path

# %%
# ============================================================
# CELL 2 — DATASET PATHS AND ANALYSIS CONFIGURATION
# ============================================================

DATASETS = {

    "ADNI": {
        "connectome_dir": (
            '/mnt/newStor/paros/paros_WORK/avery/data/final/ADNI/connectomes/fMRI'
        ),

        "metadata_path": (
            '/mnt/newStor/paros/paros_WORK/avery/data/final/ADNI/ADNI_harmonized_metadata(in).csv'
        ),
    },

    "HABS": {
        "connectome_dir": (
            '/mnt/newStor/paros/paros_WORK/avery/data/final/HABS/connectomes/fMRI'
        ),

        "metadata_path": (
            '/mnt/newStor/paros/paros_WORK/avery/data/final/HABS/HABS_harmonized_metadata(in).csv'
        ),
    },

    "ADRC": {
        "connectome_dir": (
            '/mnt/newStor/paros/paros_WORK/avery/data/final/ADRC/connectomes/fMRI'
        ),

        "metadata_path": (
            '/mnt/newStor/paros/paros_WORK/avery/data/final/ADRC/ADRC_harmonized_metadata_merged.csv'
        ),
    },
}


# ============================================================
# Combined covariate file
# SITE will come from this file
# ============================================================

ALL_COVARS_PATH = (
    '/mnt/newStor/paros/paros_WORK/avery/data/final/all_covars_aligned.csv'
)


OUTPUT_DIR = Path(
    "/mnt/newStor/paros/paros_WORK/avery/results/"
    "final/fMRI"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_N_ROIS = 84

USE_FISHER_Z = True


# ============================================================
# IMPORTANT:
#
# OLD:
# BATCH_COL = "DATASET"
#
# NEW:
# each DATASET × SITE combination is one technical batch
# ============================================================

BATCH_COL = "DATASET_SITE"


# Biological covariates that NeuroCombat should PRESERVE

CONTINUOUS_COLS = [
    "Age",
]

CATEGORICAL_COLS = [
    "Sex",
    "Diagnosis",
]


print("Configured datasets:")

for cohort, paths in DATASETS.items():

    print(
        cohort,
        "\n  connectomes:",
        paths["connectome_dir"],
        "\n  metadata:",
        paths["metadata_path"],
    )


print("\nBatch column:", BATCH_COL)

print("\nSite source:")
print(ALL_COVARS_PATH)

# %%
# ============================================================
# Validate configured paths
# ============================================================

for cohort, paths in DATASETS.items():

    connectome_path = Path(
        paths["connectome_dir"]
    )

    metadata_path = Path(
        paths["metadata_path"]
    )

    print("\n" + "=" * 70)
    print(cohort)
    print("=" * 70)

    print(
        "Connectome directory exists:",
        connectome_path.is_dir(),
    )

    print(
        "Metadata file exists:",
        metadata_path.is_file(),
    )

    if connectome_path.is_dir():

        csv_files = sorted(
            connectome_path.rglob("*.csv")
        )

        print(
            "CSV files in connectome directory:",
            len(csv_files),
        )

        print(
            "Example filenames:"
        )

        for path in csv_files[:10]:
            print(" ", path.name)

# %% [markdown]
# # Helper functions

# %%
def load_table_auto(path):
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix == ".txt":
        return pd.read_csv(path, sep="\t", low_memory=False)

    raise ValueError(f"Unsupported metadata format: {path}")


def first_existing_column(df, candidates):
    return next(
        (column for column in candidates if column in df.columns),
        None,
    )


def clean_text(series):
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
    )


def normalize_sex_value(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    mapping = {
        "1": "M",
        "1.0": "M",
        "MALE": "M",
        "M": "M",

        "2": "F",
        "2.0": "F",
        "FEMALE": "F",
        "F": "F",
    }

    return mapping.get(value, value)


def normalize_diagnosis_value(value):
    if pd.isna(value):
        return np.nan

    value = (
        str(value)
        .strip()
        .upper()
        .replace(" ", "_")
    )

    mapping = {
        "NORMAL": "CN",
        "CONTROL": "CN",
        "HEALTHY": "CN",
        "HC": "CN",
        "SMC": "CN",

        "EMCI": "MCI",
        "LMCI": "MCI",

        "DEMENTIA": "AD",
        "ALZHEIMER'S_DISEASE": "AD",
        "ALZHEIMERS_DISEASE": "AD",
    }

    return mapping.get(value, value)


def normalize_visit_series(series):
    visit = clean_text(series)

    visit = visit.replace({
        "init": "y0",
        "baseline": "y0",
        "bl": "y0",
        "m0": "y0",
        "m00": "y0",
        "0": "y0",

        "2": "y2",
        "m24": "y2",
        "24": "y2",

        "4": "y4",
        "m48": "y4",
        "48": "y4",
    })

    extracted = visit.str.extract(
        r"(y\d+)",
        expand=False,
    )

    return extracted.fillna(visit)

# %% [markdown]
# # Scan/Session ID functions

# %%
# ============================================================
# Scan/session ID and connectome-file helpers
# ============================================================

def normalize_scan_id(
    value,
    cohort,
):
    """
    Convert filenames or metadata IDs into canonical keys.

    Examples:
        ADNI: R0074_y0
        HABS: H4369_y0
        ADRC: D0007

    ADRC accepted input examples:
        D0007
        ADRC0007
        func_connectome_corr_D0007.csv
        func_connectome_corr_ADRC0007.csv

    All ADRC IDs are standardized to:
        D0007
    """

    if pd.isna(value):
        return np.nan

    original_text = str(value).strip()
    text = Path(original_text).stem

    # --------------------------------------------------------
    # Remove common prefixes
    # --------------------------------------------------------

    text = re.sub(
        r"^(func_connectome_corr_|"
        r"func_connectome_|"
        r"connectome_corr_|"
        r"connectome_|"
        r"corr_|"
        r"matrix_|"
        r"fc_|"
        r"fmri_)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"_(conn_plain|connectome|corr|"
        r"correlation|matrix|fc|fmri)$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace(
        "_Y",
        "_y",
    )

    # --------------------------------------------------------
    # ADNI
    # --------------------------------------------------------

    if cohort == "ADNI":

        match = re.search(
            r"(R\d+)_y(\d+)",
            text,
            flags=re.IGNORECASE,
        )

        if match is None:
            return np.nan

        subject = match.group(1).upper()
        visit = f"y{int(match.group(2))}"

        return f"{subject}_{visit}"

    # --------------------------------------------------------
    # HABS
    # --------------------------------------------------------

    elif cohort == "HABS":

        match = re.search(
            r"(H\d+)_y(\d+)",
            text,
            flags=re.IGNORECASE,
        )

        if match is None:
            return np.nan

        subject = match.group(1).upper()
        visit = f"y{int(match.group(2))}"

        return f"{subject}_{visit}"

    # --------------------------------------------------------
    # ADRC
    #
    # Accept either:
    #     D0007
    #     ADRC0007
    #
    # Standardize BOTH to:
    #     D0007
    #
    # ADRC does not use _y0 / _y2 / _y4 here.
    # --------------------------------------------------------

    elif cohort == "ADRC":

        match = re.search(
            r"(?:ADRC|D)(\d+)",
            text,
            flags=re.IGNORECASE,
        )

        if match is None:
            return np.nan

        subject_digits = match.group(1)

        return (
            "D"
            + subject_digits.zfill(4)
        )

    else:

        raise ValueError(
            f"Unexpected cohort: {cohort}"
        )


def extract_connectome_key(
    filename,
    cohort,
):
    """
    Extract the standardized connectome key.

    ADNI and HABS retain the original strict filename filter.
    ADRC files are accepted based on parseable IDs and matrix shape.
    """
    name = Path(filename).name
    lower_name = name.lower()

    excluded_tokens = [
        "time_series",
        "timeseries",
        "timecourse",
        "covar",
        "metadata",
        "demographic",
        "flair",
        "dwi",
        "dmri",
    ]

    if any(
        token in lower_name
        for token in excluded_tokens
    ):
        return np.nan

    if cohort in {
        "ADNI",
        "HABS",
    }:

        valid_name = (
            "connectome" in lower_name
            or "corr" in lower_name
            or lower_name.endswith(
                "_conn_plain.csv"
            )
        )

        if not valid_name:
            return np.nan

    elif cohort == "ADRC":
        # ADRC matrix filenames may not include
        # "connectome" or "corr".
        pass

    else:
        raise ValueError(
            f"Unexpected cohort: {cohort}"
        )

    return normalize_scan_id(
        name,
        cohort,
    )


def inspect_candidate_matrix(
    file_path,
):
    """
    Read a candidate CSV and determine whether it is a valid
    square numeric connectivity matrix.

    Returns:
        matrix or None
        reason string
    """
    try:
        candidate = pd.read_csv(
            file_path,
            header=None,
        )

    except pd.errors.EmptyDataError:
        return None, "empty_file"

    except Exception as error:
        return None, f"read_error: {error}"

    candidate = candidate.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Remove a possible text header/index row or column
    candidate = candidate.dropna(
        axis=0,
        how="all",
    )

    candidate = candidate.dropna(
        axis=1,
        how="all",
    )

    matrix = candidate.to_numpy(
        dtype=float
    )

    if matrix.ndim != 2:
        return None, f"not_2D_{matrix.shape}"

    if matrix.shape[0] != matrix.shape[1]:
        return None, f"not_square_{matrix.shape}"

    if matrix.shape[0] != EXPECTED_N_ROIS:
        return (
            None,
            f"unexpected_size_{matrix.shape}",
        )

    if not np.isfinite(matrix).all():
        return None, "contains_nan_or_infinity"

    return matrix, "valid"


def discover_connectome_files(
    connectome_dir,
    cohort,
):
    """
    Find valid connectome files for one cohort.
    """
    rows = []
    excluded_rows = []

    connectome_dir = Path(
        connectome_dir
    )

    if not connectome_dir.is_dir():
        raise FileNotFoundError(
            "Connectome directory does not exist: "
            f"{connectome_dir}"
        )

    for file_path in sorted(
        connectome_dir.rglob("*.csv")
    ):

        filename = file_path.name

        connectome_key = extract_connectome_key(
            filename,
            cohort,
        )

        if pd.isna(connectome_key):

            excluded_rows.append({
                "DATASET": cohort,
                "filename": filename,
                "file_path": str(file_path),
                "reason": (
                    "could_not_parse_or_excluded"
                ),
            })

            continue

        # Verify that it is an actual 84 × 84 matrix.
        matrix, reason = inspect_candidate_matrix(
            file_path
        )

        if matrix is None:

            excluded_rows.append({
                "DATASET": cohort,
                "filename": filename,
                "file_path": str(file_path),
                "reason": reason,
            })

            continue

        rows.append({
            "DATASET": cohort,
            "connectome_key": connectome_key,
            "file_path": str(file_path),
            "filename": filename,
        })

    files = pd.DataFrame(
        rows
    )

    excluded_files = pd.DataFrame(
        excluded_rows
    )

    if files.empty:

        print(
            f"\nNo valid files found for {cohort}."
        )

        if not excluded_files.empty:

            print(
                "\nFirst excluded filenames:"
            )

            display(
                excluded_files[
                    [
                        "filename",
                        "reason",
                    ]
                ].head(30)
            )

        raise ValueError(
            "No valid standard fMRI connectome "
            f"files found for {cohort}: "
            f"{connectome_dir}"
        )

    duplicate_mask = files[
        "connectome_key"
    ].duplicated(
        keep=False
    )

    if duplicate_mask.any():

        duplicate_rows = files.loc[
            duplicate_mask,
            [
                "connectome_key",
                "filename",
                "file_path",
            ],
        ].sort_values(
            [
                "connectome_key",
                "filename",
            ]
        )

        print(
            "\nDuplicate connectome keys:"
        )

        display(
            duplicate_rows
        )

        raise RuntimeError(
            f"{cohort} contains duplicate "
            "connectome keys."
        )

    files = (
        files
        .sort_values(
            "connectome_key"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"{cohort}: valid connectomes = "
        f"{len(files)}"
    )

    print(
        f"{cohort}: excluded CSV files = "
        f"{len(excluded_files)}"
    )

    if not excluded_files.empty:

        print(
            "\nExample excluded files:"
        )

        display(
            excluded_files.head(10)
        )

    return files

# %% [markdown]
# # Automatically find the scan-ID column

# %%
def choose_scan_column(
    df,
    cohort,
    preferred=None,
):
    """
    Find the metadata column containing the most valid
    scan/session IDs such as R0074_y0 or H4369_y2.
    """
    preferred = preferred or []

    candidates = [
        column
        for column in preferred
        if column in df.columns
    ]

    keyword_candidates = [
        column
        for column in df.columns
        if column not in candidates
        and any(
            token in str(column).lower()
            for token in [
                "dwi",
                "fmri",
                "run",
                "scan",
                "connectome",
                "session",
            ]
        )
    ]

    candidates.extend(keyword_candidates)

    best_column = None
    best_score = (-1, -1)

    for column in candidates:

        normalized = df[column].map(
            lambda value: normalize_scan_id(
                value,
                cohort,
            )
        )

        valid = normalized.notna()

        score = (
            int(valid.sum()),
            int(normalized[valid].nunique()),
        )

        if score > best_score:
            best_column = column
            best_score = score

    if best_column is None or best_score[0] == 0:
        return None

    print(
        f"{cohort} scan/session column selected: "
        f"{best_column}"
    )

    print(
        f"Valid rows: {best_score[0]} | "
        f"Unique scan IDs: {best_score[1]}"
    )

    return best_column

# %% [markdown]
# # Final metadata cleaning

# %%
# ============================================================
# FINAL METADATA CLEANING
# ============================================================

def finalize_metadata(df):

    output = df.copy()


    output["connectome_key"] = (
        output["connectome_key"]
        .astype("string")
        .str.strip()
    )


    # --------------------------------------------------------
    # Do NOT create DATASET_SITE here.
    #
    # Site will be replaced using all_covars_aligned.csv
    # after all three cohorts have been loaded.
    # --------------------------------------------------------

    keep_columns = [
        "DATASET",
        "connectome_key",
        "Age",
        "Sex",
        "Diagnosis",
    ]


    output = (
        output[
            keep_columns
        ]
        .copy()
    )


    output = output.dropna(
        subset=[
            "connectome_key"
        ]
    )


    duplicate_mask = (
        output.duplicated(
            subset=[
                "DATASET",
                "connectome_key",
            ],
            keep=False,
        )
    )


    if duplicate_mask.any():

        duplicates = (
            output.loc[
                duplicate_mask
            ]
            .sort_values(
                [
                    "DATASET",
                    "connectome_key",
                ]
            )
        )

        raise RuntimeError(
            "Duplicate metadata session keys found:\n"
            + duplicates.head(50).to_string(
                index=False
            )
        )


    return (
        output
        .reset_index(drop=True)
    )

# %% [markdown]
# # ADNI metadata matching

# %%
def harmonize_adni_metadata(df_raw):
    df = df_raw.copy()

    df["DATASET"] = "ADNI"

    # First choice:
    # exact scan-level column containing R####_y#
    scan_column = choose_scan_column(
        df,
        cohort="ADNI",
        preferred=[
            "fMRI",
            "FMRI",
            "connectome_key",
            "DWI",
            "runno",
            "scan_id",
            "Scan_ID",
            "Subject_ID",
        ],
    )

    if scan_column is not None:

        df["connectome_key"] = df[scan_column].map(
            lambda value: normalize_scan_id(
                value,
                "ADNI",
            )
        )

        print(
            "ADNI connectome-key source:",
            scan_column,
        )

    # Second choice:
    # exact subject key plus visit label
    elif {
        "DWI_subject_key",
        "DWI_visit_label",
    }.issubset(df.columns):

        subject = (
            df["DWI_subject_key"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        visit = normalize_visit_series(
            df["DWI_visit_label"]
        )

        df["connectome_key"] = (
            subject
            + "_"
            + visit
        )

        print(
            "ADNI connectome-key source: "
            "DWI_subject_key + DWI_visit_label"
        )

    # Final fallback:
    # RID plus Visit/VISCODE
    else:

        if "RID" not in df.columns:
            raise ValueError(
                "ADNI metadata does not contain RID."
            )

        rid = (
            pd.to_numeric(
                df["RID"],
                errors="coerce",
            )
            .astype("Int64")
            .astype("string")
            .str.zfill(4)
        )

        visit_column = first_existing_column(
            df,
            [
                "Visit",
                "DWI_visit_label",
                "VISCODE_x",
                "VISCODE",
            ],
        )

        if visit_column is None:
            raise ValueError(
                "ADNI metadata has no usable visit column."
            )

        visit = normalize_visit_series(
            df[visit_column]
        )

        df["connectome_key"] = (
            "R"
            + rid
            + "_"
            + visit
        )

        print(
            "ADNI connectome-key source: "
            f"RID + {visit_column}"
        )

    age_column = first_existing_column(
        df,
        [
            "AGE",
            "Age",
            "AGE_session_covars",
            "age_visit_corrected",
        ],
    )

    if age_column is None:
        raise ValueError(
            "No ADNI age column found."
        )

    df["Age"] = pd.to_numeric(
        df[age_column],
        errors="coerce",
    )

    sex_column = first_existing_column(
        df,
        [
            "PTGENDER",
            "SEX_from_covars",
            "Sex",
            "SEX",
        ],
    )

    if sex_column is None:
        raise ValueError(
            "No ADNI sex column found."
        )

    df["Sex"] = df[sex_column].map(
        normalize_sex_value
    )

    diagnosis_column = first_existing_column(
        df,
        [
            "DX_Label_harmonized",
            "GROUP_broad_from_covars",
            "GROUP_from_covars",
            "Research Group",
            "DX_bl",
        ],
    )

    if diagnosis_column is None:
        df["Diagnosis"] = "Unknown"
    else:
        df["Diagnosis"] = df[
            diagnosis_column
        ].map(normalize_diagnosis_value)

    site_column = first_existing_column(
        df,
        [
            "SITE_from_covars",
            "SITE",
            "site",
            "COLPROT",
        ],
    )

    if site_column is None:
        df["Site"] = "ADNI"
    else:
        df["Site"] = (
            df[site_column]
            .astype("string")
            .str.strip()
        )

    return finalize_metadata(df)

# %% [markdown]
# # HABS metadata matching

# %%
def harmonize_habs_metadata(df_raw):
    df = df_raw.copy()

    df["DATASET"] = "HABS"

    scan_column = choose_scan_column(
        df,
        cohort="HABS",
        preferred=[
            "fMRI",
            "FMRI",
            "DWI",
            "dwi",
            "connectome_key",
            "runno",
            "RunNo",
            "RUNNO",
            "scan_id",
            "Scan_ID",
            "Subject_ID",
            "subject_match",
        ],
    )

    if scan_column is not None:

        df["connectome_key"] = df[scan_column].map(
            lambda value: normalize_scan_id(
                value,
                "HABS",
            )
        )

        print(
            "HABS connectome-key source:",
            scan_column,
        )

    elif {
        "DWI_subject",
        "DWI_visit",
    }.issubset(df.columns):

        subject = (
            df["DWI_subject"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        visit = normalize_visit_series(
            df["DWI_visit"]
        )

        df["connectome_key"] = (
            subject
            + "_"
            + visit
        )

        print(
            "HABS connectome-key source: "
            "DWI_subject + DWI_visit"
        )

    elif {
        "Subject",
        "Year",
    }.issubset(df.columns):

        subject_digits = (
            df["Subject"]
            .astype("string")
            .str.extract(
                r"(\d+)",
                expand=False,
            )
            .str.zfill(4)
        )

        year = pd.to_numeric(
            df["Year"],
            errors="coerce",
        ).astype("Int64")

        df["connectome_key"] = (
            "H"
            + subject_digits
            + "_y"
            + year.astype("string")
        )

        print(
            "HABS connectome-key source: "
            "Subject + Year"
        )

    else:
        raise ValueError(
            "Could not construct HABS session IDs. "
            "Expected an H####_y# column, "
            "DWI_subject + DWI_visit, "
            "or Subject + Year."
        )

    age_column = first_existing_column(
        df,
        [
            "Age",
            "AGE",
            "age",
        ],
    )

    if age_column is None:
        raise ValueError(
            "No HABS age column found."
        )

    df["Age"] = pd.to_numeric(
        df[age_column],
        errors="coerce",
    )

    sex_column = first_existing_column(
        df,
        [
            "Sex",
            "SEX",
            "sex",
        ],
    )

    if sex_column is None:
        raise ValueError(
            "No HABS sex column found."
        )

    df["Sex"] = df[sex_column].map(
        normalize_sex_value
    )

    diagnosis_column = first_existing_column(
        df,
        [
            "DX_Label_harmonized",
            "Diagnosis",
            "Dementia",
            "Research Group",
        ],
    )

    if diagnosis_column is None:
        df["Diagnosis"] = "Unknown"
    else:
        df["Diagnosis"] = df[
            diagnosis_column
        ].map(normalize_diagnosis_value)

    site_column = first_existing_column(
        df,
        [
            "Interview_Site",
            "SITE",
            "Site",
            "site",
        ],
    )

    if site_column is None:
        df["Site"] = "HABS"
    else:
        df["Site"] = (
            df[site_column]
            .astype("string")
            .str.strip()
        )

    return finalize_metadata(df)

# %% [markdown]
# # ADRC metadata matching

# %%
def harmonize_adrc_metadata(df_raw):
    """
    Standardize ADRC metadata so it can be combined with
    ADNI and HABS.

    Canonical ADRC connectome key:
        D####

    Examples:
        ADRC0007 -> D0007
        D0007    -> D0007

    Output is finalized by finalize_metadata().
    """

    df = df_raw.copy()

    df["DATASET"] = "ADRC"

    print("ADRC metadata shape:")
    print(df.shape)

    print("\nADRC metadata columns:")
    print(df.columns.tolist())

    # ========================================================
    # 1. Subject / connectome key
    # ========================================================

    subject_column = "PTID"

    if subject_column not in df.columns:

        raise ValueError(
            f"Expected ADRC subject column "
            f"'{subject_column}', but it was not found."
        )

    # --------------------------------------------------------
    # Convert both:
    #
    #     ADRC0007
    #     D0007
    #
    # into:
    #
    #     D0007
    # --------------------------------------------------------

    df["connectome_key"] = (
        df[subject_column]
        .map(
            lambda value:
            normalize_scan_id(
                value,
                "ADRC",
            )
        )
    )

    print(
        "\nSelected ADRC subject column:",
        subject_column,
    )

    print(
        "Valid ADRC metadata keys:",
        int(
            df["connectome_key"]
            .notna()
            .sum()
        ),
    )

    print(
        "Invalid ADRC metadata keys:",
        int(
            df["connectome_key"]
            .isna()
            .sum()
        ),
    )

    print(
        "\nExample ADRC metadata keys:"
    )

    print(
        df["connectome_key"]
        .dropna()
        .head(10)
        .tolist()
    )

    # ========================================================
    # 2. Age
    # ========================================================

    age_column = "VISIT_AGE"

    if age_column not in df.columns:

        raise ValueError(
            f"Expected ADRC age column "
            f"'{age_column}', but it was not found."
        )

    df["Age"] = pd.to_numeric(
        df[age_column],
        errors="coerce",
    )

    print(
        "\nSelected ADRC age column:",
        age_column,
    )

    print(
        "Missing ADRC ages:",
        int(
            df["Age"]
            .isna()
            .sum()
        ),
    )

    print(
        "\nADRC age summary:"
    )

    print(
        df["Age"].describe()
    )

    # ========================================================
    # 3. Sex
    # ========================================================

    sex_candidates = [
        "Sex",
        "SEX",
        "sex",
        "Gender",
        "GENDER",
        "gender",
        "SUBJECT_SEX",
        "BIOLOGICAL_SEX",
    ]

    sex_column = next(
        (
            column
            for column in sex_candidates
            if column in df.columns
        ),
        None,
    )

    if sex_column is None:

        possible_sex_columns = [
            column
            for column in df.columns
            if (
                "sex" in str(column).lower()
                or
                "gender" in str(column).lower()
            )
        ]

        print(
            "\nPossible ADRC sex columns:"
        )

        print(
            possible_sex_columns
        )

        if len(possible_sex_columns) == 1:

            sex_column = (
                possible_sex_columns[0]
            )

        else:

            raise ValueError(
                "Could not uniquely identify "
                "the ADRC sex column."
            )

    df["Sex"] = (
        df[sex_column]
        .map(
            normalize_sex_value
        )
    )

    print(
        "\nSelected ADRC sex column:",
        sex_column,
    )

    print(
        "ADRC sex counts:"
    )

    print(
        df["Sex"]
        .value_counts(
            dropna=False
        )
    )

    # ========================================================
    # 4. Diagnosis
    # ========================================================

    diagnosis_candidates = [
        "DX_Label_harmonized",
        "Diagnosis",
        "DIAGNOSIS",
        "diagnosis",
        "DX",
        "Dx",
        "dx",
        "Group",
        "GROUP",
        "group",
        "Research Group",
        "Cognitive_Status",
        "cognitive_status",
        "Clinical_Status",
        "clinical_status",
        "COG_STATUS",
    ]

    diagnosis_column = next(
        (
            column
            for column
            in diagnosis_candidates
            if column in df.columns
        ),
        None,
    )

    if diagnosis_column is None:

        possible_diagnosis_columns = [
            column
            for column in df.columns
            if any(
                token
                in str(column).lower()
                for token in [
                    "diagn",
                    "dx",
                    "group",
                    "cognitive",
                    "clinical_status",
                    "cog_status",
                ]
            )
        ]

        print(
            "\nPossible ADRC diagnosis columns:"
        )

        print(
            possible_diagnosis_columns
        )

        if len(
            possible_diagnosis_columns
        ) == 1:

            diagnosis_column = (
                possible_diagnosis_columns[0]
            )

    if diagnosis_column is None:

        print(
            "\nWarning: no unique ADRC diagnosis "
            "column was found. Setting Diagnosis "
            "to 'Unknown'."
        )

        df["Diagnosis"] = "Unknown"

    else:

        df["Diagnosis"] = (
            df[diagnosis_column]
            .map(
                normalize_diagnosis_value
            )
        )

        print(
            "\nSelected ADRC diagnosis column:",
            diagnosis_column,
        )

        print(
            "ADRC diagnosis counts:"
        )

        print(
            df["Diagnosis"]
            .value_counts(
                dropna=False
            )
        )

    # ========================================================
    # 5. Site
    # ========================================================

    site_candidates = [
        "SITE",
        "Site",
        "site",
        "Scanner_Site",
        "scanner_site",
        "Interview_Site",
        "Center",
        "CENTER",
        "center",
    ]

    site_column = next(
        (
            column
            for column in site_candidates
            if column in df.columns
        ),
        None,
    )

    if site_column is None:

        possible_site_columns = [
            column
            for column in df.columns
            if any(
                token
                in str(column).lower()
                for token in [
                    "site",
                    "center",
                    "scanner",
                ]
            )
        ]

        if len(
            possible_site_columns
        ) == 1:

            site_column = (
                possible_site_columns[0]
            )

    if site_column is None:

        df["Site"] = "ADRC"

        print(
            "\nNo ADRC site column found. "
            "Using Site='ADRC'."
        )

    else:

        df["Site"] = (
            df[site_column]
            .astype("string")
            .str.strip()
        )

        print(
            "\nSelected ADRC site column:",
            site_column,
        )

    # ========================================================
    # 6. Remove invalid keys
    # ========================================================

    invalid_key_mask = (
        df["connectome_key"]
        .isna()
    )

    if invalid_key_mask.any():

        print(
            "\nRemoving ADRC metadata rows "
            "without a valid connectome key:",
            int(
                invalid_key_mask.sum()
            ),
        )

        df = (
            df.loc[
                ~invalid_key_mask
            ]
            .copy()
        )

    # ========================================================
    # 7. Handle duplicate subject keys
    # ========================================================

    duplicate_mask = (
        df["connectome_key"]
        .duplicated(
            keep=False
        )
    )

    if duplicate_mask.any():

        duplicate_columns = [
            "connectome_key",
            subject_column,
            age_column,
            sex_column,
        ]

        if diagnosis_column is not None:

            duplicate_columns.append(
                diagnosis_column
            )

        if site_column is not None:

            duplicate_columns.append(
                site_column
            )

        print(
            "\nDuplicate ADRC metadata keys:"
        )

        display(
            df.loc[
                duplicate_mask,
                duplicate_columns,
            ]
            .sort_values(
                "connectome_key"
            )
        )

        # Prefer rows with the most complete
        # biological covariates.
        df["_required_nonmissing"] = (
            df[
                [
                    "Age",
                    "Sex",
                    "Diagnosis",
                    "Site",
                ]
            ]
            .notna()
            .sum(axis=1)
        )

        df = (
            df
            .sort_values(
                [
                    "connectome_key",
                    "_required_nonmissing",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .drop_duplicates(
                subset=[
                    "connectome_key"
                ],
                keep="first",
            )
            .drop(
                columns=[
                    "_required_nonmissing"
                ]
            )
            .copy()
        )

        print(
            "Kept one ADRC metadata row "
            "per connectome key."
        )

    # ========================================================
    # 8. Final validation
    # ========================================================

    required_columns = [
        "DATASET",
        "connectome_key",
        "Age",
        "Sex",
        "Diagnosis",
        "Site",
    ]

    missing_required_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_required_columns:

        raise ValueError(
            "ADRC metadata is missing standardized "
            f"columns: {missing_required_columns}"
        )

    print(
        "\nADRC standardized metadata rows:",
        len(df),
    )

    print(
        "\nMissing standardized values:"
    )

    print(
        df[
            required_columns
        ]
        .isna()
        .sum()
    )

    print(
        "\nExample standardized ADRC metadata:"
    )

    display(
        df[
            required_columns
        ]
        .head(10)
    )

    return finalize_metadata(
        df
    )

# %% [markdown]
# # Load ADNI, ADRC, and HABS

# %%
# ============================================================
# Load connectome files and metadata for all three cohorts
# ============================================================

all_file_tables = []
all_metadata_tables = []


for cohort, paths in DATASETS.items():

    print("\n" + "=" * 70)
    print("LOADING", cohort)
    print("=" * 70)

    files_cohort = discover_connectome_files(
        connectome_dir=paths[
            "connectome_dir"
        ],
        cohort=cohort,
    )

    metadata_path = Path(
        paths["metadata_path"]
    )

    if not metadata_path.is_file():

        raise FileNotFoundError(
            f"{cohort} metadata_path must point "
            "to an actual metadata file, but found:\n"
            f"{metadata_path}"
        )

    metadata_raw = load_table_auto(
        metadata_path
    )

    if cohort == "ADNI":

        metadata_cohort = (
            harmonize_adni_metadata(
                metadata_raw
            )
        )

    elif cohort == "HABS":

        metadata_cohort = (
            harmonize_habs_metadata(
                metadata_raw
            )
        )

    elif cohort == "ADRC":

        metadata_cohort = (
            harmonize_adrc_metadata(
                metadata_raw
            )
        )

    else:

        raise ValueError(
            f"Unexpected cohort: {cohort}"
        )

    print(
        "Valid connectome files:",
        len(files_cohort),
    )

    print(
        "Canonical metadata rows:",
        len(metadata_cohort),
    )

    print(
        "\nExample connectome keys:"
    )

    print(
        files_cohort[
            "connectome_key"
        ].head(10).tolist()
    )

    print(
        "\nExample metadata keys:"
    )

    print(
        metadata_cohort[
            "connectome_key"
        ].head(10).tolist()
    )

    all_file_tables.append(
        files_cohort
    )

    all_metadata_tables.append(
        metadata_cohort
    )


files = pd.concat(
    all_file_tables,
    ignore_index=True,
)

metadata = pd.concat(
    all_metadata_tables,
    ignore_index=True,
)


print(
    "\nCombined file inventory shape:"
)

print(
    files.shape
)

print(
    "\nFiles by dataset:"
)

print(
    files[
        "DATASET"
    ].value_counts()
)

print(
    "\nCombined canonical metadata shape:"
)

print(
    metadata.shape
)

print(
    "\nMetadata rows by dataset:"
)

print(
    metadata[
        "DATASET"
    ].value_counts()
)

# %% [markdown]
# # Load site information from all_covars_aligned.csv

# %%
# ============================================================
# NEW CELL — LOAD SITE INFORMATION FROM ALL_COVARS
#
# Purpose:
# Use all_covars_aligned.csv as the single authoritative
# source of SITE across cohorts.
# ============================================================

all_covars = pd.read_csv(
    ALL_COVARS_PATH,
    low_memory=False,
)


print("=" * 72)
print("ALL COVARIATES")
print("=" * 72)

print(
    "Shape:",
    all_covars.shape
)

print(
    "\nColumns:"
)

print(
    all_covars.columns.tolist()
)


required_columns = [
    "subject_id",
    "SITE",
]


missing_columns = [
    column
    for column in required_columns
    if column not in all_covars.columns
]


if missing_columns:

    raise KeyError(
        "all_covars_aligned.csv is missing: "
        f"{missing_columns}"
    )


print("\nMissing SITE:")

print(
    all_covars["SITE"]
    .isna()
    .sum()
)


print("\nUnique SITE values:")

print(
    all_covars["SITE"]
    .nunique()
)


print("\nExample rows:")

display(
    all_covars[
        [
            "subject_id",
            "SITE",
        ]
    ].head(20)
)

# %% [markdown]
# # Convert all_covars IDs to the same connectome keys

# %%
# ============================================================
# NEW CELL — STANDARDIZE ALL_COVARS SITE KEYS
# ============================================================

def infer_dataset_from_covariate_id(value):

    if pd.isna(value):
        return np.nan

    text = (
        str(value)
        .strip()
        .upper()
    )

    # --------------------------------------------------------
    # ADNI
    # Example:
    # R0074_y0
    # --------------------------------------------------------

    if re.search(
        r"R\d+_Y\d+",
        text,
    ):
        return "ADNI"

    # --------------------------------------------------------
    # HABS
    # Example:
    # H4369_y0
    # --------------------------------------------------------

    if re.search(
        r"H\d+_Y\d+",
        text,
    ):
        return "HABS"

    # --------------------------------------------------------
    # ADRC
    #
    # Accept both:
    # D0007
    # ADRC0007
    # --------------------------------------------------------

    if re.search(
        r"^(?:ADRC|D)\d+$",
        text,
    ):
        return "ADRC"

    return np.nan


site_covars = all_covars[
    [
        "subject_id",
        "SITE",
    ]
].copy()


site_covars["DATASET"] = (
    site_covars[
        "subject_id"
    ]
    .map(
        infer_dataset_from_covariate_id
    )
)


# Remove other cohorts that may exist in all_covars
site_covars = (
    site_covars[
        site_covars["DATASET"]
        .isin(
            [
                "ADNI",
                "HABS",
                "ADRC",
            ]
        )
    ]
    .copy()
)


site_covars["connectome_key"] = (

    site_covars.apply(

        lambda row:

        normalize_scan_id(
            row["subject_id"],
            row["DATASET"],
        ),

        axis=1,
    )
)


site_covars["Site"] = (
    site_covars["SITE"]
    .astype("string")
    .str.strip()
)


site_covars = (
    site_covars[
        [
            "DATASET",
            "connectome_key",
            "Site",
        ]
    ]
    .dropna(
        subset=[
            "DATASET",
            "connectome_key",
            "Site",
        ]
    )
    .drop_duplicates()
    .reset_index(drop=True)
)


print("=" * 72)
print("STANDARDIZED SITE COVARIATES")
print("=" * 72)


print(
    "\nRows:",
    len(site_covars)
)


print("\nRows by cohort:")

print(
    site_covars[
        "DATASET"
    ]
    .value_counts()
)


print("\nNumber of sites by cohort:")

print(
    site_covars
    .groupby(
        "DATASET"
    )["Site"]
    .nunique()
)


print("\nExamples:")

display(
    site_covars.head(20)
)

# %% [markdown]
# # Check duplicate site keys

# %%
# ============================================================
# NEW CELL — CHECK SITE KEY DUPLICATES
# ============================================================

site_duplicate_mask = (
    site_covars.duplicated(
        subset=[
            "DATASET",
            "connectome_key",
        ],
        keep=False,
    )
)


if site_duplicate_mask.any():

    conflicting_sites = (
        site_covars.loc[
            site_duplicate_mask
        ]
        .sort_values(
            [
                "DATASET",
                "connectome_key",
            ]
        )
    )


    print(
        "Duplicate site assignments found:"
    )

    display(
        conflicting_sites
    )


    raise RuntimeError(
        "Some scans have multiple SITE assignments "
        "in all_covars_aligned.csv."
    )


print(
    "No duplicate site assignments."
)

# %% [markdown]
# # Match metadata to connectomes

# %%
matched = files.merge(
    metadata,
    how="inner",
    on=[
        "DATASET",
        "connectome_key",
    ],
    validate="one_to_one",
)

file_keys = files[
    [
        "DATASET",
        "connectome_key",
    ]
].drop_duplicates()

metadata_keys = metadata[
    [
        "DATASET",
        "connectome_key",
    ]
].drop_duplicates()


connectome_without_metadata = file_keys.merge(
    metadata_keys,
    how="left",
    on=[
        "DATASET",
        "connectome_key",
    ],
    indicator=True,
)

connectome_without_metadata = (
    connectome_without_metadata[
        connectome_without_metadata["_merge"]
        == "left_only"
    ]
    .drop(columns="_merge")
)

connectome_without_metadata["problem"] = (
    "connectome_without_metadata"
)


metadata_without_connectome = metadata_keys.merge(
    file_keys,
    how="left",
    on=[
        "DATASET",
        "connectome_key",
    ],
    indicator=True,
)

metadata_without_connectome = (
    metadata_without_connectome[
        metadata_without_connectome["_merge"]
        == "left_only"
    ]
    .drop(columns="_merge")
)

metadata_without_connectome["problem"] = (
    "metadata_without_connectome"
)


unmatched = pd.concat(
    [
        connectome_without_metadata,
        metadata_without_connectome,
    ],
    ignore_index=True,
)


print("MATCH COUNTS")
print(
    matched[
        "DATASET"
    ].value_counts()
)

print("\nCONNECTOME FILE COUNTS")
print(
    files[
        "DATASET"
    ].value_counts()
)

print("\nMETADATA COUNTS")
print(
    metadata[
        "DATASET"
    ].value_counts()
)

print("\nExample matched rows:")
display(
    matched[
        [
            "DATASET",
            "connectome_key",
            "Age",
            "Sex",
            "Diagnosis",
            "file_path",
        ]
    ].head(10)
)

print("\nExample unmatched rows:")
display(
    unmatched.head(20)
)


files.to_csv(
    OUTPUT_DIR / "01_connectome_inventory.csv",
    index=False,
)

metadata.to_csv(
    OUTPUT_DIR / "02_canonical_metadata.csv",
    index=False,
)

matched.to_csv(
    OUTPUT_DIR / "03_exact_matches.csv",
    index=False,
)

unmatched.to_csv(
    OUTPUT_DIR / "04_unmatched_keys.csv",
    index=False,
)

# %%
# ============================================================
# MATCH METADATA TO CONNECTOMES
# + ADD SITE FROM ALL_COVARS
# + ADRC FALLBACK SITE
# ============================================================

matched = files.merge(
    metadata,
    how="inner",
    on=[
        "DATASET",
        "connectome_key",
    ],
    validate="one_to_one",
)


print("=" * 72)
print("CONNECTOME / METADATA MATCH")
print("=" * 72)

print(
    matched[
        "DATASET"
    ]
    .value_counts()
)


# ============================================================
# Add SITE from all_covars_aligned.csv
#
# This provides the authoritative site labels for
# ADNI and HABS.
# ============================================================

matched = matched.merge(
    site_covars,
    how="left",
    on=[
        "DATASET",
        "connectome_key",
    ],
    validate="one_to_one",
)


# ============================================================
# ADRC FALLBACK
#
# all_covars_aligned.csv currently has no ADRC site rows.
#
# Your ADRC metadata also did not contain a separate site
# variable, so ADRC was previously treated as one acquisition
# site. Restore that behavior here.
#
# IMPORTANT:
# This is appropriate only if these ADRC scans truly belong
# to one acquisition site/scanner group.
# ============================================================

adrc_missing_site_mask = (
    matched["DATASET"]
    .eq("ADRC")
    &
    matched["Site"]
    .isna()
)


print(
    "\nADRC scans without SITE from all_covars:",
    int(
        adrc_missing_site_mask.sum()
    )
)


matched.loc[
    adrc_missing_site_mask,
    "Site"
] = "ADRC"


# ============================================================
# SITE MATCHING SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("SITE MATCHING AFTER ADRC FALLBACK")
print("=" * 72)


site_summary = (
    matched
    .groupby(
        "DATASET"
    )
    .agg(
        N_scans=(
            "connectome_key",
            "size"
        ),

        N_with_site=(
            "Site",
            "count"
        ),

        N_sites=(
            "Site",
            "nunique"
        ),
    )
)


display(
    site_summary
)


missing_site = (
    matched.loc[
        matched["Site"].isna(),
        [
            "DATASET",
            "connectome_key",
            "file_path",
        ]
    ]
    .copy()
)


print(
    "\nRemaining scans missing SITE:",
    len(
        missing_site
    )
)


if not missing_site.empty:

    display(
        missing_site.head(50)
    )

    raise ValueError(
        "Some matched scans still have no SITE."
    )


# ============================================================
# Expected sanity check
# ============================================================

print(
    "\nTotal matched scans:",
    len(matched)
)


print(
    "\nDataset counts:"
)

print(
    matched[
        "DATASET"
    ]
    .value_counts()
)

# %% [markdown]
# # Create the joint batch

# %%
# ============================================================
# CREATE JOINT DATASET × SITE TECHNICAL BATCH
#
# This is the key one-step harmonization design:
#
# ADNI site effects
# HABS site effects
# ADRC acquisition group
#
# are estimated in ONE NeuroCombat model.
# ============================================================

matched["Site"] = (
    matched["Site"]
    .astype("string")
    .str.strip()
)


matched["DATASET_SITE"] = (
    matched["DATASET"]
    .astype("string")
    .str.strip()
    .str.upper()

    + "__"

    + matched["Site"]
    .astype("string")
    .str.strip()
)


print("=" * 72)
print("JOINT TECHNICAL BATCHES")
print("=" * 72)


print(
    "Total scans:",
    len(
        matched
    )
)


print(
    "Number of cohorts:",
    matched[
        "DATASET"
    ].nunique()
)


print(
    "Number of technical batches:",
    matched[
        "DATASET_SITE"
    ].nunique()
)


print("\nDataset counts:")

print(
    matched[
        "DATASET"
    ]
    .value_counts()
)


print("\nBatch counts:")

batch_counts = (
    matched[
        "DATASET_SITE"
    ]
    .value_counts()
    .sort_index()
)


print(
    batch_counts.to_string()
)

# %% [markdown]
# # Check small site batches

# %%
# ============================================================
# TECHNICAL BATCH SIZE QC
# ============================================================

batch_qc = (
    matched[
        "DATASET_SITE"
    ]
    .value_counts()
    .rename_axis(
        "DATASET_SITE"
    )
    .reset_index(
        name="N"
    )
)


print("=" * 72)
print("TECHNICAL BATCH SIZE QC")
print("=" * 72)


print(
    "Total batches:",
    len(batch_qc)
)


# ============================================================
# Singleton batches
# ============================================================

singleton_batches = (
    batch_qc[
        batch_qc["N"] < 2
    ]
    .copy()
)


print(
    "\nSingleton batches (N < 2):",
    len(
        singleton_batches
    )
)


if not singleton_batches.empty:

    display(
        singleton_batches
    )


# ============================================================
# Small but usable batches
# ============================================================

small_batches = (
    batch_qc[
        (
            batch_qc["N"] >= 2
        )
        &
        (
            batch_qc["N"] < 5
        )
    ]
    .copy()
)


print(
    "\nSmall batches (2 <= N < 5):",
    len(
        small_batches
    )
)


if not small_batches.empty:

    display(
        small_batches
    )

    print(
        "\nThese batches will be retained, "
        "but their ComBat estimates may be less stable."
    )

# %% [markdown]
# # Connectome matrix functions

# %%
def load_connectome_matrix(path):
    if os.path.getsize(path) == 0:
        raise ValueError("empty file")

    matrix = pd.read_csv(
        path,
        header=None,
    ).to_numpy(dtype=float)

    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError(
            f"non-square matrix: {matrix.shape}"
        )

    expected_shape = (
        EXPECTED_N_ROIS,
        EXPECTED_N_ROIS,
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            f"ROI mismatch: expected {expected_shape}, "
            f"got {matrix.shape}"
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "matrix contains NaN or infinity"
        )

    matrix = (
        matrix
        + matrix.T
    ) / 2.0

    np.fill_diagonal(
        matrix,
        1.0,
    )

    return matrix


def fisher_z(matrix):
    clipped = np.clip(
        matrix,
        -0.999999,
        0.999999,
    )

    transformed = np.arctanh(
        clipped
    )

    np.fill_diagonal(
        transformed,
        0.0,
    )

    return transformed


def flatten_upper(matrix):
    indices = np.triu_indices_from(
        matrix,
        k=1,
    )

    return matrix[indices]

# %% [markdown]
# # Build the raw edge-feature matrix

# %%
all_edges = []
sample_ids = []
matrix_qc_rows = []

for row in matched.itertuples(index=False):

    sample_id = (
        f"{row.DATASET}_{row.connectome_key}"
        .lower()
    )

    try:
        matrix = load_connectome_matrix(
            row.file_path
        )

        if USE_FISHER_Z:
            matrix = fisher_z(
                matrix
            )

        edges = flatten_upper(
            matrix
        )

        all_edges.append(
            edges
        )

        sample_ids.append(
            sample_id
        )

        matrix_qc_rows.append({
            "sample_id": sample_id,
            "DATASET": row.DATASET,
            "connectome_key": row.connectome_key,
            "file_path": row.file_path,
            "status": "loaded",
            "reason": "",
            "matrix_shape": str(matrix.shape),
        })

    except Exception as error:
        print(
            "\nSkipping:",
            sample_id
            )
        print("Reason:", error)

        matrix_qc_rows.append({
            "sample_id": sample_id,
            "DATASET": row.DATASET,
            "connectome_key": row.connectome_key,
            "file_path": row.file_path,
            "status": "skipped",
            "reason": str(error),
            "matrix_shape": "",
            })


if len(all_edges) == 0:
    raise RuntimeError(
        "No valid matched connectomes were loaded."
    )


X_raw = pd.DataFrame(
    all_edges,
    index=sample_ids,
    columns=[
        f"edge_{index}"
        for index in range(
            len(all_edges[0])
        )
    ],
)

matrix_qc = pd.DataFrame(
    matrix_qc_rows
)


print("X_raw shape:")
print(X_raw.shape)

print("\nExpected number of edges:")
print(
    EXPECTED_N_ROIS
    * (EXPECTED_N_ROIS - 1)
    // 2
)

print("\nLoaded matrices by dataset:")
print(
    matrix_qc.loc[
        matrix_qc["status"] == "loaded",
        "DATASET",
    ].value_counts()
)

print("\nSkipped matrix examples:")
display(
    matrix_qc[
        matrix_qc["status"] == "skipped"
    ].head(20)
)


matrix_qc.to_csv(
    OUTPUT_DIR / "05_matrix_loading_qc.csv",
    index=False,
)

# %% [markdown]
# # Check invalid connectomes

# %%
# ============================================================
# Check invalid / skipped connectome matrices
# Run this AFTER matrix_qc has been created
# ============================================================

required_qc_columns = {
    "sample_id",
    "DATASET",
    "connectome_key",
    "file_path",
    "status",
    "reason",
    "matrix_shape",
}

missing_qc_columns = (
    required_qc_columns
    - set(matrix_qc.columns)
)

if missing_qc_columns:
    raise KeyError(
        "matrix_qc is missing required columns: "
        f"{sorted(missing_qc_columns)}"
    )


# Select matrices that failed during loading or validation
invalid_connectomes = (
    matrix_qc.loc[
        matrix_qc["status"].astype(str).str.lower()
        != "loaded"
    ]
    .copy()
    .reset_index(drop=True)
)


print("=" * 70)
print("INVALID CONNECTOME MATRIX CHECK")
print("=" * 70)

print("\nTotal matched connectomes checked:")
print(len(matrix_qc))

print("\nSuccessfully loaded connectomes:")
print(
    int(
        (
            matrix_qc["status"].astype(str).str.lower()
            == "loaded"
        ).sum()
    )
)

print("\nInvalid/skipped connectomes:")
print(len(invalid_connectomes))


if invalid_connectomes.empty:

    print("\nNo invalid connectome matrices were found.")

else:

    # Normalize empty reasons
    invalid_connectomes["reason"] = (
        invalid_connectomes["reason"]
        .fillna("Unknown reason")
        .astype(str)
        .str.strip()
        .replace("", "Unknown reason")
    )

    print("\nInvalid connectomes by dataset:")
    print(
        invalid_connectomes[
            "DATASET"
        ].value_counts(dropna=False)
    )

    print("\nReasons for exclusion:")
    print(
        invalid_connectomes[
            "reason"
        ].value_counts(dropna=False)
    )

    print("\nInvalid connectome details:")

    display(
        invalid_connectomes[
            [
                "sample_id",
                "DATASET",
                "connectome_key",
                "matrix_shape",
                "reason",
                "file_path",
            ]
        ]
    )


# Save the full invalid-connectome table
invalid_connectomes.to_csv(
    OUTPUT_DIR / "06_invalid_connectome_matrices.csv",
    index=False,
)

print(
    "\nSaved invalid-connectome report to:"
)
print(
    OUTPUT_DIR / "06_invalid_connectome_matrices.csv"
)

# %% [markdown]
# # Check matched and unmatched Data Diagnosis

# %%
# ============================================================
# MATCHED / UNMATCHED DATA DIAGNOSTIC
#
# Run AFTER:
#   files
#   metadata
#
# have been created.
#
# This checks EXACT connectome_key matching and tries to explain
# why each unmatched scan is unmatched.
# ============================================================

import re
import numpy as np
import pandas as pd


# ============================================================
# 1. HELPER — EXTRACT SUBJECT-LEVEL ID AND VISIT
# ============================================================

def split_subject_and_visit(
    connectome_key,
    dataset,
):
    """
    Convert scan-level keys into:

        subject_base
        visit

    Examples
    --------

    ADNI:
        R0178_y4
        -> R0178, y4

    HABS:
        H4789_y2
        -> H4789, y2

    ADRC:
        D0052
        -> D0052, NaN

    Legacy ADRC IDs such as ADRC0052 are also accepted
    and standardized to D0052.
    """

    if pd.isna(connectome_key):
        return np.nan, np.nan

    text = (
        str(connectome_key)
        .strip()
        .upper()
    )

    # --------------------------------------------------------
    # ADNI
    # --------------------------------------------------------

    if dataset == "ADNI":

        match = re.fullmatch(
            r"(R\d+)_Y(\d+)",
            text,
        )

        if match is not None:

            subject = (
                match.group(1)
            )

            visit = (
                "y"
                + str(
                    int(
                        match.group(2)
                    )
                )
            )

            return (
                subject,
                visit,
            )

    # --------------------------------------------------------
    # HABS
    # --------------------------------------------------------

    if dataset == "HABS":

        match = re.fullmatch(
            r"(H\d+)_Y(\d+)",
            text,
        )

        if match is not None:

            subject = (
                match.group(1)
            )

            visit = (
                "y"
                + str(
                    int(
                        match.group(2)
                    )
                )
            )

            return (
                subject,
                visit,
            )

    # --------------------------------------------------------
    # ADRC
    #
    # D0052       -> D0052
    # ADRC0052    -> D0052
    #
    # No visit is used for ADRC.
    # --------------------------------------------------------

    if dataset == "ADRC":

        match = re.fullmatch(
            r"(?:ADRC|D)(\d+)",
            text,
        )

        if match is not None:

            subject = (
                "D"
                + match.group(1).zfill(4)
            )

            return (
                subject,
                np.nan,
            )

    # --------------------------------------------------------
    # Could not parse
    # --------------------------------------------------------

    return np.nan, np.nan
    """
    Convert scan-level keys into:
        subject_base
        visit

    Examples
    --------
    ADNI:
        R0178_y4 -> R0178, y4

    HABS:
        H4789_y2 -> H4789, y2

    ADRC:
        ADRC0052 -> ADRC0052, None
    """

    if pd.isna(connectome_key):
        return np.nan, np.nan

    text = str(connectome_key).strip().upper()


    # --------------------------------------------------------
    # ADNI
    # --------------------------------------------------------

    if dataset == "ADNI":

        match = re.fullmatch(
            r"(R\d+)_Y(\d+)",
            text,
        )

        if match is not None:

            subject = match.group(1)

            visit = (
                "y"
                + str(
                    int(
                        match.group(2)
                    )
                )
            )

            return subject, visit


    # --------------------------------------------------------
    # HABS
    # --------------------------------------------------------

    if dataset == "HABS":

        match = re.fullmatch(
            r"(H\d+)_Y(\d+)",
            text,
        )

        if match is not None:

            subject = match.group(1)

            visit = (
                "y"
                + str(
                    int(
                        match.group(2)
                    )
                )
            )

            return subject, visit


    # --------------------------------------------------------
    # ADRC
    # --------------------------------------------------------

    if dataset == "ADRC":

        match = re.search(
            r"(ADRC\d+)",
            text,
        )

        if match is not None:

            return (
                match.group(1),
                np.nan,
            )


    # --------------------------------------------------------
    # Could not parse
    # --------------------------------------------------------

    return np.nan, np.nan


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_file_columns = {
    "DATASET",
    "connectome_key",
    "file_path",
}

required_metadata_columns = {
    "DATASET",
    "connectome_key",
}


missing_file_columns = (
    required_file_columns
    -
    set(files.columns)
)

missing_metadata_columns = (
    required_metadata_columns
    -
    set(metadata.columns)
)


if missing_file_columns:

    raise KeyError(
        "files is missing columns: "
        f"{sorted(missing_file_columns)}"
    )


if missing_metadata_columns:

    raise KeyError(
        "metadata is missing columns: "
        f"{sorted(missing_metadata_columns)}"
    )


# ============================================================
# 3. MAKE UNIQUE KEY TABLES
# ============================================================

file_keys = (
    files[
        [
            "DATASET",
            "connectome_key",
            "file_path",
            "filename",
        ]
    ]
    .drop_duplicates(
        subset=[
            "DATASET",
            "connectome_key",
        ]
    )
    .copy()
)


metadata_columns_to_keep = [
    column
    for column in [
        "DATASET",
        "connectome_key",
        "Age",
        "Sex",
        "Diagnosis",
    ]
    if column in metadata.columns
]


metadata_keys = (
    metadata[
        metadata_columns_to_keep
    ]
    .drop_duplicates(
        subset=[
            "DATASET",
            "connectome_key",
        ]
    )
    .copy()
)


# ============================================================
# 4. EXACT MATCHES
# ============================================================

matched_diagnostic = (
    file_keys.merge(

        metadata_keys,

        how="inner",

        on=[
            "DATASET",
            "connectome_key",
        ],

        validate="one_to_one",
    )
)


# ============================================================
# 5. CONNECTOME EXISTS, BUT METADATA DOES NOT
# ============================================================

connectome_without_metadata = (

    file_keys.merge(

        metadata_keys[
            [
                "DATASET",
                "connectome_key",
            ]
        ],

        how="left",

        on=[
            "DATASET",
            "connectome_key",
        ],

        indicator=True,
    )
)


connectome_without_metadata = (

    connectome_without_metadata.loc[
        connectome_without_metadata[
            "_merge"
        ]
        ==
        "left_only"
    ]
    .drop(
        columns="_merge"
    )
    .copy()
)


# ============================================================
# 6. METADATA EXISTS, BUT CONNECTOME DOES NOT
# ============================================================

metadata_without_connectome = (

    metadata_keys.merge(

        file_keys[
            [
                "DATASET",
                "connectome_key",
            ]
        ],

        how="left",

        on=[
            "DATASET",
            "connectome_key",
        ],

        indicator=True,
    )
)


metadata_without_connectome = (

    metadata_without_connectome.loc[
        metadata_without_connectome[
            "_merge"
        ]
        ==
        "left_only"
    ]
    .drop(
        columns="_merge"
    )
    .copy()
)


# ============================================================
# 7. ADD SUBJECT-LEVEL ID AND VISIT
# ============================================================

for table in [
    file_keys,
    metadata_keys,
    connectome_without_metadata,
    metadata_without_connectome,
]:

    parsed = table.apply(

        lambda row:
            split_subject_and_visit(
                row["connectome_key"],
                row["DATASET"],
            ),

        axis=1,
    )


    table["subject_base"] = [
        value[0]
        for value in parsed
    ]


    table["visit"] = [
        value[1]
        for value in parsed
    ]


# ============================================================
# 8. BUILD SUBJECT-LEVEL LOOKUP TABLES
# ============================================================

file_subjects_by_dataset = {}

metadata_subjects_by_dataset = {}


for cohort in [
    "ADNI",
    "HABS",
    "ADRC",
]:

    file_subjects_by_dataset[
        cohort
    ] = set(

        file_keys.loc[
            file_keys["DATASET"] == cohort,
            "subject_base",
        ]
        .dropna()
        .astype(str)
    )


    metadata_subjects_by_dataset[
        cohort
    ] = set(

        metadata_keys.loc[
            metadata_keys["DATASET"] == cohort,
            "subject_base",
        ]
        .dropna()
        .astype(str)
    )


# ============================================================
# 9. EXPLAIN CONNECTOME WITHOUT METADATA
# ============================================================

def explain_connectome_without_metadata(row):

    dataset = row["DATASET"]

    subject = row["subject_base"]

    visit = row["visit"]


    # Could not parse scan ID
    if pd.isna(subject):

        return (
            "ID parsing problem"
        )


    # Same participant exists in metadata
    if (
        subject
        in
        metadata_subjects_by_dataset[
            dataset
        ]
    ):

        if dataset in [
            "ADNI",
            "HABS",
        ]:

            available_visits = (

                metadata_keys.loc[
                    (
                        metadata_keys[
                            "DATASET"
                        ]
                        ==
                        dataset
                    )
                    &
                    (
                        metadata_keys[
                            "subject_base"
                        ]
                        ==
                        subject
                    ),
                    "visit",
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )


            return (
                "Participant exists in metadata, "
                "but exact visit/session is missing. "
                f"Connectome visit={visit}; "
                f"metadata visits={available_visits}"
            )


        return (
            "Participant exists in metadata, "
            "but exact scan key differs"
        )


    # Subject itself is not represented
    return (
        "Participant not found in metadata"
    )


connectome_without_metadata[
    "likely_reason"
] = (

    connectome_without_metadata
    .apply(
        explain_connectome_without_metadata,
        axis=1,
    )
)


# ============================================================
# 10. EXPLAIN METADATA WITHOUT CONNECTOME
# ============================================================

def explain_metadata_without_connectome(row):

    dataset = row["DATASET"]

    subject = row["subject_base"]

    visit = row["visit"]


    if pd.isna(subject):

        return (
            "ID parsing problem"
        )


    # Participant has some connectome, but not this visit
    if (
        subject
        in
        file_subjects_by_dataset[
            dataset
        ]
    ):

        if dataset in [
            "ADNI",
            "HABS",
        ]:

            available_visits = (

                file_keys.loc[
                    (
                        file_keys[
                            "DATASET"
                        ]
                        ==
                        dataset
                    )
                    &
                    (
                        file_keys[
                            "subject_base"
                        ]
                        ==
                        subject
                    ),
                    "visit",
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )


            return (
                "Participant has a connectome, "
                "but not for this metadata visit. "
                f"Metadata visit={visit}; "
                f"connectome visits={available_visits}"
            )


        return (
            "Participant has a connectome, "
            "but exact scan key differs"
        )


    return (
        "No usable fMRI connectome found "
        "for this participant"
    )


metadata_without_connectome[
    "likely_reason"
] = (

    metadata_without_connectome
    .apply(
        explain_metadata_without_connectome,
        axis=1,
    )
)


# ============================================================
# 11. SUMMARY TABLE
# ============================================================

summary_rows = []


for cohort in [
    "ADNI",
    "HABS",
    "ADRC",
]:

    n_files = int(
        (
            file_keys[
                "DATASET"
            ]
            ==
            cohort
        ).sum()
    )


    n_metadata = int(
        (
            metadata_keys[
                "DATASET"
            ]
            ==
            cohort
        ).sum()
    )


    n_matched = int(
        (
            matched_diagnostic[
                "DATASET"
            ]
            ==
            cohort
        ).sum()
    )


    n_connectome_only = int(
        (
            connectome_without_metadata[
                "DATASET"
            ]
            ==
            cohort
        ).sum()
    )


    n_metadata_only = int(
        (
            metadata_without_connectome[
                "DATASET"
            ]
            ==
            cohort
        ).sum()
    )


    match_rate_connectomes = (

        n_matched
        /
        n_files

        if n_files > 0
        else np.nan
    )


    match_rate_metadata = (

        n_matched
        /
        n_metadata

        if n_metadata > 0
        else np.nan
    )


    summary_rows.append({

        "DATASET":
            cohort,

        "Valid_connectomes":
            n_files,

        "Metadata_rows":
            n_metadata,

        "Matched":
            n_matched,

        "Connectome_without_metadata":
            n_connectome_only,

        "Metadata_without_connectome":
            n_metadata_only,

        "Connectome_match_rate":
            match_rate_connectomes,

        "Metadata_match_rate":
            match_rate_metadata,
    })


matching_summary = pd.DataFrame(
    summary_rows
)


# TOTAL row
matching_summary.loc[
    len(
        matching_summary
    )
] = {

    "DATASET":
        "TOTAL",

    "Valid_connectomes":
        matching_summary[
            "Valid_connectomes"
        ].sum(),

    "Metadata_rows":
        matching_summary[
            "Metadata_rows"
        ].sum(),

    "Matched":
        matching_summary[
            "Matched"
        ].sum(),

    "Connectome_without_metadata":
        matching_summary[
            "Connectome_without_metadata"
        ].sum(),

    "Metadata_without_connectome":
        matching_summary[
            "Metadata_without_connectome"
        ].sum(),

    "Connectome_match_rate":
        (
            matching_summary[
                "Matched"
            ].sum()
            /
            matching_summary[
                "Valid_connectomes"
            ].sum()
        ),

    "Metadata_match_rate":
        (
            matching_summary[
                "Matched"
            ].sum()
            /
            matching_summary[
                "Metadata_rows"
            ].sum()
        ),
}


# ============================================================
# 12. DISPLAY SUMMARY
# ============================================================

print("=" * 80)
print("CONNECTOME / METADATA MATCHING SUMMARY")
print("=" * 80)


summary_display = (
    matching_summary.copy()
)


summary_display[
    "Connectome_match_rate"
] = (

    summary_display[
        "Connectome_match_rate"
    ]
    .map(
        lambda value:
            f"{value:.1%}"
            if pd.notna(value)
            else ""
    )
)


summary_display[
    "Metadata_match_rate"
] = (

    summary_display[
        "Metadata_match_rate"
    ]
    .map(
        lambda value:
            f"{value:.1%}"
            if pd.notna(value)
            else ""
    )
)


display(
    summary_display
)


# ============================================================
# 13. SHOW MATCHED SCANS
# ============================================================

print("\n" + "=" * 80)
print("MATCHED SCANS")
print("=" * 80)


print(
    "Total matched:",
    len(
        matched_diagnostic
    )
)


display(
    matched_diagnostic
    .sort_values(
        [
            "DATASET",
            "connectome_key",
        ]
    )
)


# ============================================================
# 14. SHOW CONNECTOME WITHOUT METADATA
# ============================================================

print("\n" + "=" * 80)
print("CONNECTOMES WITHOUT MATCHING METADATA")
print("=" * 80)


print(
    "Total:",
    len(
        connectome_without_metadata
    )
)


print("\nCounts by cohort:")

print(
    connectome_without_metadata[
        "DATASET"
    ]
    .value_counts()
)


print("\nReason summary:")

print(
    connectome_without_metadata[
        "likely_reason"
    ]
    .value_counts()
)


display(
    connectome_without_metadata[
        [
            "DATASET",
            "connectome_key",
            "subject_base",
            "visit",
            "likely_reason",
            "filename",
            "file_path",
        ]
    ]
    .sort_values(
        [
            "DATASET",
            "connectome_key",
        ]
    )
)


# ============================================================
# 15. SHOW METADATA WITHOUT CONNECTOME
# ============================================================

print("\n" + "=" * 80)
print("METADATA WITHOUT MATCHING CONNECTOME")
print("=" * 80)


print(
    "Total:",
    len(
        metadata_without_connectome
    )
)


print("\nCounts by cohort:")

print(
    metadata_without_connectome[
        "DATASET"
    ]
    .value_counts()
)


print("\nReason summary:")

print(
    metadata_without_connectome[
        "likely_reason"
    ]
    .value_counts()
)


columns_to_show = [
    column
    for column in [
        "DATASET",
        "connectome_key",
        "subject_base",
        "visit",
        "Age",
        "Sex",
        "Diagnosis",
        "likely_reason",
    ]
    if column
    in metadata_without_connectome.columns
]


display(
    metadata_without_connectome[
        columns_to_show
    ]
    .sort_values(
        [
            "DATASET",
            "connectome_key",
        ]
    )
)


# ============================================================
# 16. REASON SUMMARY BY COHORT
# ============================================================

print("\n" + "=" * 80)
print("UNMATCHED REASONS BY COHORT")
print("=" * 80)


connectome_reason_summary = (

    connectome_without_metadata
    .groupby(
        [
            "DATASET",
            "likely_reason",
        ]
    )
    .size()
    .rename(
        "N"
    )
    .reset_index()
)


metadata_reason_summary = (

    metadata_without_connectome
    .groupby(
        [
            "DATASET",
            "likely_reason",
        ]
    )
    .size()
    .rename(
        "N"
    )
    .reset_index()
)


print(
    "\nConnectome → no metadata:"
)

display(
    connectome_reason_summary
)


print(
    "\nMetadata → no connectome:"
)

display(
    metadata_reason_summary
)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

matching_summary.to_csv(
    OUTPUT_DIR
    /
    "matching_summary.csv",

    index=False,
)


matched_diagnostic.to_csv(
    OUTPUT_DIR
    /
    "matched_scans_diagnostic.csv",

    index=False,
)


connectome_without_metadata.to_csv(
    OUTPUT_DIR
    /
    "connectomes_without_metadata_diagnostic.csv",

    index=False,
)


metadata_without_connectome.to_csv(
    OUTPUT_DIR
    /
    "metadata_without_connectomes_diagnostic.csv",

    index=False,
)


connectome_reason_summary.to_csv(
    OUTPUT_DIR
    /
    "connectome_unmatched_reason_summary.csv",

    index=False,
)


metadata_reason_summary.to_csv(
    OUTPUT_DIR
    /
    "metadata_unmatched_reason_summary.csv",

    index=False,
)


print(
    "\nDiagnostic tables saved to:"
)

print(
    OUTPUT_DIR
)

# %% [markdown]
# # Align features and metadata

# %%
# ============================================================
# ALIGN FEATURES AND METADATA
# + REMOVE SINGLETON TECHNICAL BATCHES
# ============================================================


# ============================================================
# 1. Create sample IDs
# ============================================================

matched["sample_id"] = (
    matched["DATASET"]
    .astype(str)

    + "_"

    + matched[
        "connectome_key"
    ]
    .astype(str)
).str.lower()


loaded_sample_ids = set(
    matrix_qc.loc[
        matrix_qc[
            "status"
        ]
        ==
        "loaded",

        "sample_id",
    ]
)


metadata_aligned = (
    matched.loc[
        matched[
            "sample_id"
        ]
        .isin(
            loaded_sample_ids
        )
    ]
    .copy()
)


metadata_aligned = (
    metadata_aligned
    .set_index(
        "sample_id"
    )
)


# ============================================================
# 2. Align feature matrix and metadata
# ============================================================

common_ids = (
    X_raw.index
    .intersection(
        metadata_aligned.index
    )
)


X_raw = (
    X_raw
    .loc[
        common_ids
    ]
    .copy()
)


metadata_aligned = (
    metadata_aligned
    .loc[
        common_ids
    ]
    .copy()
)


assert (
    list(
        X_raw.index
    )
    ==
    list(
        metadata_aligned.index
    )
)


print("=" * 72)
print("INITIAL FEATURE / METADATA ALIGNMENT")
print("=" * 72)


print(
    "X_raw:",
    X_raw.shape
)


print(
    "metadata:",
    metadata_aligned.shape
)


print("\nDataset counts:")

print(
    metadata_aligned[
        "DATASET"
    ]
    .value_counts()
)


# ============================================================
# 3. Clean covariates
# ============================================================

metadata_aligned["Age"] = (
    pd.to_numeric(
        metadata_aligned[
            "Age"
        ],
        errors="coerce",
    )
)


for column in [
    "DATASET",
    "DATASET_SITE",
    "Site",
    "Sex",
    "Diagnosis",
]:

    metadata_aligned[column] = (
        metadata_aligned[
            column
        ]
        .astype("string")
        .str.strip()
    )


needed_columns = [
    BATCH_COL,
    "Age",
    "Sex",
    "Diagnosis",
]


# ============================================================
# 4. Missing covariates
# ============================================================

print("\n" + "=" * 72)
print("MISSING COVARIATES")
print("=" * 72)


print(
    metadata_aligned[
        needed_columns
    ]
    .isna()
    .sum()
)


valid_covariate_rows = (
    metadata_aligned[
        needed_columns
    ]
    .notna()
    .all(axis=1)
)


print(
    "\nRows removed because of "
    "missing required covariates:",
    int(
        (~valid_covariate_rows)
        .sum()
    )
)


metadata_aligned = (
    metadata_aligned.loc[
        valid_covariate_rows
    ]
    .copy()
)


X_raw = (
    X_raw.loc[
        metadata_aligned.index
    ]
    .copy()
)


# ============================================================
# 5. Identify singleton DATASET_SITE batches
# ============================================================

batch_counts = (
    metadata_aligned[
        BATCH_COL
    ]
    .value_counts()
)


singleton_batch_names = (
    batch_counts[
        batch_counts < 2
    ]
    .index
)


singleton_mask = (
    metadata_aligned[
        BATCH_COL
    ]
    .isin(
        singleton_batch_names
    )
)


print("\n" + "=" * 72)
print("SINGLETON TECHNICAL BATCH REMOVAL")
print("=" * 72)


print(
    "Singleton batches:"
)


if len(
    singleton_batch_names
) == 0:

    print(
        "None"
    )

else:

    print(
        batch_counts.loc[
            singleton_batch_names
        ]
    )


print(
    "\nNumber of scans removed:",
    int(
        singleton_mask.sum()
    )
)


if singleton_mask.any():

    display(
        metadata_aligned.loc[
            singleton_mask,
            [
                "DATASET",
                "connectome_key",
                "Site",
                BATCH_COL,
                "Age",
                "Sex",
                "Diagnosis",
            ]
        ]
    )


# ============================================================
# 6. Remove singleton batches from BOTH metadata and X_raw
# ============================================================

metadata_aligned = (
    metadata_aligned.loc[
        ~singleton_mask
    ]
    .copy()
)


X_raw = (
    X_raw.loc[
        metadata_aligned.index
    ]
    .copy()
)


# ============================================================
# 7. Feature QC
# ============================================================

finite_edges = (
    np.isfinite(
        X_raw.to_numpy()
    )
    .all(axis=0)
)


if not finite_edges.all():

    print(
        "\nDropping invalid edge columns:",
        int(
            (~finite_edges).sum()
        )
    )


    X_raw = (
        X_raw.loc[
            :,
            finite_edges
        ]
        .copy()
    )


# ============================================================
# 8. Final batch counts after singleton removal
# ============================================================

final_batch_counts = (
    metadata_aligned[
        BATCH_COL
    ]
    .value_counts()
)


print("\n" + "=" * 72)
print("FINAL HARMONIZATION INPUT")
print("=" * 72)


print(
    "X_raw:",
    X_raw.shape
)


print(
    "Metadata:",
    metadata_aligned.shape
)


print("\nDataset counts:")

print(
    metadata_aligned[
        "DATASET"
    ]
    .value_counts()
)


print("\nSites per cohort:")

print(
    metadata_aligned
    .groupby(
        "DATASET"
    )[
        "Site"
    ]
    .nunique()
)


print(
    "\nNumber of technical batches:",
    len(
        final_batch_counts
    )
)


print(
    "\nSmallest batch size:",
    final_batch_counts.min()
)


print("\nJoint batch counts:")

print(
    final_batch_counts
    .sort_index()
    .to_string()
)


# ============================================================
# 9. Hard sanity checks
# ============================================================

assert (
    X_raw.shape[0]
    ==
    len(
        metadata_aligned
    )
)


assert (
    metadata_aligned[
        "DATASET"
    ]
    .nunique()
    ==
    3
), (
    "ADNI, HABS and ADRC must all remain."
)


assert (
    final_batch_counts.min()
    >=
    2
), (
    "At least one singleton technical batch remains."
)


assert (
    np.isfinite(
        X_raw.to_numpy()
    )
    .all()
)


print(
    "\nAll pre-ComBat checks passed."
)

# %% [markdown]
# # Run ComBat

# %%
# ============================================================
# RUN ONE JOINT NEUROCOMBAT
#
# Within-cohort:
#   site/scanner differences inside ADNI/HABS/ADRC
#
# Between-cohort:
#   acquisition differences among ADNI/HABS/ADRC
#
# Both are modeled simultaneously through DATASET_SITE.
# ============================================================


covars = (

    metadata_aligned[
        [
            BATCH_COL,
            "Age",
            "Sex",
            "Diagnosis",
        ]
    ]
    .copy()
)


print("=" * 72)
print("JOINT COMBAT INPUT")
print("=" * 72)


print(
    "X_raw shape:",
    X_raw.shape
)


print(
    "Covariates shape:",
    covars.shape
)


print(
    "\nBatch column:",
    BATCH_COL
)


print(
    "\nNumber of batches:",
    covars[
        BATCH_COL
    ].nunique()
)


print("\nBatch counts:")

print(
    covars[
        BATCH_COL
    ]
    .value_counts()
    .sort_index()
)


print("\nDataset counts:")

print(
    metadata_aligned[
        "DATASET"
    ]
    .value_counts()
)


# ============================================================
# Alignment validation
# ============================================================

if len(X_raw) != len(covars):

    raise ValueError(
        "X_raw and covars have different "
        "numbers of subjects.\n"
        f"X_raw rows: {len(X_raw)}\n"
        f"covars rows: {len(covars)}"
    )


if (
    covars[
        [
            BATCH_COL,
            "Age",
            "Sex",
            "Diagnosis",
        ]
    ]
    .isna()
    .any()
    .any()
):

    raise ValueError(
        "Missing values remain in ComBat covariates."
    )


# ============================================================
# Feature QC
# ============================================================

X_raw = (
    X_raw
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
)


bad_edge_columns = (

    X_raw.columns[
        X_raw
        .isna()
        .any(axis=0)
    ]
)


print(
    "\nBad edge columns before ComBat:",
    len(
        bad_edge_columns
    ),
)


if len(
    bad_edge_columns
) > 0:

    X_raw = (
        X_raw.drop(
            columns=
            bad_edge_columns
        )
    )


print(
    "X_raw after cleaning:",
    X_raw.shape,
)


# ============================================================
# ONE NeuroCombat run
# ============================================================

combat_result = neuroCombat(

    # NeuroCombat uses features × subjects
    dat=(
        X_raw
        .T
        .to_numpy()
    ),

    covars=(
        covars
        .reset_index(
            drop=True
        )
    ),

    batch_col=BATCH_COL,

    continuous_cols=[
        "Age",
    ],

    categorical_cols=[
        "Sex",
        "Diagnosis",
    ],

    eb=True,

    parametric=True,

    mean_only=False,
)


# ============================================================
# Convert back to subjects × edges
# ============================================================

X_combat = pd.DataFrame(

    combat_result[
        "data"
    ].T,

    index=
        X_raw.index,

    columns=
        X_raw.columns,
)


print("\n" + "=" * 72)
print("COMBAT FINISHED")
print("=" * 72)


print(
    "X_combat:",
    X_combat.shape
)


print(
    "NaN:",
    int(
        X_combat
        .isna()
        .sum()
        .sum()
    )
)


print(
    "Infinity:",
    int(
        np.isinf(
            X_combat
            .to_numpy()
        )
        .sum()
    )
)


if not np.isfinite(
    X_combat.to_numpy()
).all():

    raise ValueError(
        "X_combat contains NaN or infinity."
    )


# %%
# ============================================================
# SAVE HARMONIZED fMRI CONNECTOMES
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_CONNECTOME_DIR = Path(
    '/mnt/newStor/paros/paros_WORK/avery/results/final/fMRI/harmonized_connectomes'
)

OUTPUT_CONNECTOME_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MATRIX SETTINGS
# ============================================================

N_ROIS = 84

upper_indices = np.triu_indices(
    N_ROIS,
    k=1,
)

expected_edges = (
    N_ROIS
    * (N_ROIS - 1)
    // 2
)


print("=" * 80)
print("SAVE HARMONIZED fMRI CONNECTOMES")
print("=" * 80)

print(
    "X_combat shape:",
    X_combat.shape,
)

print(
    "metadata rows:",
    len(metadata_aligned),
)

print(
    "Expected edges:",
    expected_edges,
)


# ============================================================
# CHECK ALIGNMENT
# ============================================================

assert (
    X_combat.shape[0]
    ==
    len(metadata_aligned)
), (
    "X_combat rows do not match metadata_aligned."
)


assert (
    X_combat.shape[1]
    ==
    expected_edges
), (
    f"Expected {expected_edges} edges, "
    f"but X_combat has {X_combat.shape[1]}."
)


# ============================================================
# SAVE EACH CONNECTOME
# ============================================================

saved_rows = []


for i in range(
    X_combat.shape[0]
):

    # --------------------------------------------------------
    # Harmonized Fisher-Z edge vector
    # --------------------------------------------------------

    z_edges = np.asarray(
        X_combat.iloc[i]
        if isinstance(X_combat, pd.DataFrame)
        else X_combat[i],
        dtype=float,
    )


    # --------------------------------------------------------
    # Convert Fisher-Z back to Pearson correlation
    #
    # z = arctanh(r)
    # therefore:
    # r = tanh(z)
    # --------------------------------------------------------

    correlation_edges = np.tanh(
        z_edges
    )


    # --------------------------------------------------------
    # Reconstruct 84 × 84 matrix
    # --------------------------------------------------------

    matrix = np.zeros(
        (
            N_ROIS,
            N_ROIS,
        ),
        dtype=float,
    )


    matrix[
        upper_indices
    ] = correlation_edges


    # Mirror upper triangle to lower triangle

    matrix[
        (
            upper_indices[1],
            upper_indices[0],
        )
    ] = correlation_edges


    # Correlation matrix diagonal

    np.fill_diagonal(
        matrix,
        1.0,
    )


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    row = metadata_aligned.iloc[
        i
    ]


    dataset = str(
        row[
            "DATASET"
        ]
    )


    connectome_key = str(
        row[
            "connectome_key"
        ]
    )


    # --------------------------------------------------------
    # Cohort folder
    # --------------------------------------------------------

    cohort_dir = (
        OUTPUT_CONNECTOME_DIR
        /
        dataset
    )


    cohort_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    output_path = (

        cohort_dir
        /
        f"{connectome_key}_harmonized_fMRI.csv"
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    pd.DataFrame(
        matrix
    ).to_csv(
        output_path,
        index=False,
        header=False,
    )


    saved_rows.append(
        {
            "DATASET":
                dataset,

            "connectome_key":
                connectome_key,

            "harmonized_connectome":
                str(
                    output_path
                ),
        }
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

harmonized_fmri_manifest = pd.DataFrame(
    saved_rows
)


manifest_path = (

    OUTPUT_CONNECTOME_DIR
    /
    "harmonized_fMRI_manifest.csv"
)


harmonized_fmri_manifest.to_csv(
    manifest_path,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("fMRI HARMONIZED CONNECTOMES SAVED")
print("=" * 80)


print(
    "Total:",
    len(
        harmonized_fmri_manifest
    )
)


print(
    "\nBy dataset:"
)


print(

    harmonized_fmri_manifest[
        "DATASET"
    ]
    .value_counts()
)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_CONNECTOME_DIR
)


print(
    "\nManifest:"
)

print(
    manifest_path
)
# %%

# %%
# ============================================================
# SAVE FINAL MATRICES FOR DOWNSTREAM EVALUATION
# ============================================================

EVALUATION_INPUT_DIR = (
    OUTPUT_DIR
    /
    "evaluation_inputs"
)

EVALUATION_INPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. Final raw edge-feature matrix
# ============================================================

raw_path = (
    EVALUATION_INPUT_DIR
    /
    "fMRI_X_raw_edges.csv"
)

X_raw.to_csv(
    raw_path,
    index=True,
)


# ============================================================
# 2. Final harmonized edge-feature matrix
# ============================================================

combat_path = (
    EVALUATION_INPUT_DIR
    /
    "fMRI_X_combat_edges.csv"
)

X_combat.to_csv(
    combat_path,
    index=True,
)


# ============================================================
# 3. Final aligned metadata
# ============================================================

metadata_path = (
    EVALUATION_INPUT_DIR
    /
    "fMRI_metadata_aligned.csv"
)

metadata_aligned.to_csv(
    metadata_path,
    index=True,
)


# ============================================================
# 4. Final consistency checks
# ============================================================

assert (
    X_raw.shape
    ==
    X_combat.shape
), (
    "X_raw and X_combat have different shapes."
)


assert (
    X_raw.shape[0]
    ==
    len(metadata_aligned)
), (
    "Feature matrices and metadata have different row counts."
)


assert (
    list(X_raw.index)
    ==
    list(X_combat.index)
), (
    "X_raw and X_combat sample IDs are not aligned."
)


assert (
    list(X_raw.index)
    ==
    list(metadata_aligned.index)
), (
    "Feature matrices and metadata sample IDs are not aligned."
)


print("\n" + "=" * 80)
print("EVALUATION INPUTS SAVED")
print("=" * 80)

print(
    "\nX_raw:",
    X_raw.shape
)

print(
    "X_combat:",
    X_combat.shape
)

print(
    "metadata_aligned:",
    metadata_aligned.shape
)


print("\nSaved files:")

print(
    raw_path
)

print(
    combat_path
)

print(
    metadata_path
)
# %%
