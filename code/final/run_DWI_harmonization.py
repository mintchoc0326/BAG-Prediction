from IPython.display import display
# %% [markdown]
# # File path

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
# CELL 1 — IMPORTS + PATHS
# ============================================================

from pathlib import Path
import re

import numpy as np
import pandas as pd

from neuroCombat import neuroCombat



# %%
# ============================================================
# ROOT
# ============================================================

ROOT = Path(
    "/mnt/newStor/paros/paros_WORK/avery"
)

DATA_DIR = (
    ROOT
    / "data"
    / "final"
)


# ============================================================
# DWI PLAIN CONNECTOME DIRECTORIES
# ============================================================

DWI_DIRS = {

    "ADNI": (
        DATA_DIR
        / "ADNI"
        / "connectomes"
        / "DWI"
        / "plain"
    ),

    "HABS": (
        DATA_DIR
        / "HABS"
        / "connectomes"
        / "DWI"
        / "plain"
    ),

    "ADRC": (
        DATA_DIR
        / "ADRC"
        / "connectomes"
        / "DWI"
        / "plain"
    ),

    "ADDECODE": (
        DATA_DIR
        / "ADDecode"
        / "connectomes"
        / "DWI"
        / "plain"
    ),
}


# ============================================================
# EXISTING PREP DIRECTORIES
#
# ADNI / HABS / ADDecode already have these.
# ADRC will be rebuilt from its own metadata.
# ============================================================

PREP_DIRS = {

    "ADNI": (
        DATA_DIR
        / "ADNI"
        / "connectomes"
        / "DWI"
        / "harmonize_prep"
    ),

    "HABS": (
        DATA_DIR
        / "HABS"
        / "connectomes"
        / "DWI"
        / "harmonize_prep"
    ),

    "ADRC": (
        DATA_DIR
        / "ADRC"
        / "connectomes"
        / "DWI"
        / "harmonize_prep_joint"
    ),

    "ADDECODE": (
        DATA_DIR
        / "ADDecode"
        / "connectomes"
        / "DWI"
        / "harmonize_prep"
    ),
}


# ============================================================
# ADRC METADATA
#
# If this prints False, only change THIS ONE LINE.
# ============================================================

ADRC_METADATA_PATH = (
    DATA_DIR
    / "ADRC"
    / "ADRC_harmonized_metadata_merged.csv"
)


# ============================================================
# RESULTS
# ============================================================

RESULTS_DIR = (
    ROOT
    / "results"
    / "final/DWI"
)

OUT_META = RESULTS_DIR / "metadata"
OUT_ARRAYS = RESULTS_DIR / "arrays"
OUT_FIGURES = RESULTS_DIR / "figures"


for folder in [
    OUT_META,
    OUT_ARRAYS,
    OUT_FIGURES,
]:

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# STATUS
# ============================================================

print("=" * 80)
print("DWI DIRECTORIES")
print("=" * 80)

for cohort, path in DWI_DIRS.items():

    print(
        f"{cohort:10s}",
        path.exists(),
        path,
    )


print("\n" + "=" * 80)
print("EXISTING PREP")
print("=" * 80)

for cohort in [
    "ADNI",
    "HABS",
    "ADDECODE",
]:

    prep = PREP_DIRS[cohort]

    covars = (
        prep
        / "covars_aligned.csv"
    )

    manifest = (
        prep
        / "features_manifest_aligned.csv"
    )

    print(f"\n[{cohort}]")

    print(
        "covars:",
        covars.exists(),
    )

    print(
        "manifest:",
        manifest.exists(),
    )


print("\n" + "=" * 80)
print("ADRC METADATA")
print("=" * 80)

print(
    ADRC_METADATA_PATH.exists(),
    ADRC_METADATA_PATH,
)

# %% [markdown]
# # General helper functions

# %%
# ============================================================
# CELL 2 — HELPER FUNCTIONS
# ============================================================


def normalize_string_id(value):
    """
    Generic ID normalization.
    """

    if pd.isna(value):
        return None

    value = (
        str(value)
        .strip()
        .lower()
    )

    value = re.sub(
        r"\s+",
        "",
        value,
    )

    return value


# ============================================================
# ADRC subject ID
#
# ADRC0040 -> ADRC0040
# ADRC_0040 -> ADRC0040
# D0040 -> ADRC0040
# ============================================================

def normalize_adrc_subject(value):

    if pd.isna(value):
        return None

    text = str(value)

    match = re.search(
        r"(?:ADRC|D)[_\-]?0*(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    number = int(
        match.group(1)
    )

    return (
        f"ADRC{number:04d}"
    )


# ============================================================
# Find first available column
# ============================================================

def find_column(
    df,
    candidates,
    required=True,
):

    for col in candidates:

        if col in df.columns:
            return col

    if required:

        raise ValueError(
            "Could not find any of these columns:\n"
            + str(candidates)
            + "\nAvailable columns:\n"
            + str(df.columns.tolist())
        )

    return None


# ============================================================
# Standardize Sex
# ============================================================

def standardize_sex(series):

    result = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return result.replace(
        {
            "MALE": "M",
            "FEMALE": "F",
            "1": "M",
            "2": "F",
            "NAN": pd.NA,
            "NONE": pd.NA,
            "": pd.NA,
        }
    )


# ============================================================
# Standardize diagnosis
# ============================================================

def standardize_diagnosis(series):

    result = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return result.replace(
        {
            "NORMAL": "CN",
            "CONTROL": "CN",
            "HEALTHY": "CN",
            "HC": "CN",
            "COGNITIVELY NORMAL": "CN",

            "ALZHEIMER DISEASE": "AD",
            "ALZHEIMER'S DISEASE": "AD",

            "NAN": pd.NA,
            "NONE": pd.NA,
            "": pd.NA,
        }
    )

# %% [markdown]
# # Build ADRC alignment from ADRC metadata

# %%
# ============================================================
# CELL 3 — BUILD ADRC PREP FROM ADRC METADATA
# ============================================================

ADRC_DWI_DIR = DWI_DIRS["ADRC"]
ADRC_PREP_DIR = PREP_DIRS["ADRC"]

ADRC_PREP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. LOAD ADRC METADATA
# ============================================================

if ADRC_METADATA_PATH.suffix.lower() in [
    ".xlsx",
    ".xls",
]:
    adrc_meta_raw = pd.read_excel(
        ADRC_METADATA_PATH
    )
else:
    adrc_meta_raw = pd.read_csv(
        ADRC_METADATA_PATH,
        low_memory=False,
    )


print("=" * 80)
print("ADRC METADATA")
print("=" * 80)

print(
    "Shape:",
    adrc_meta_raw.shape,
)

print(
    "Columns:",
    adrc_meta_raw.columns.tolist(),
)


# ============================================================
# 2. USE ACTUAL ADRC COLUMNS
# ============================================================

subject_col = "PTID"

# Prefer visit-specific age if available,
# otherwise use screening age.
if "VISIT_AGE" in adrc_meta_raw.columns:
    age_col = "VISIT_AGE"
else:
    age_col = "SUBJECT_AGE_SCREEN"

sex_col = "SUBJECT_SEX"

diagnosis_col = "DX_Label_harmonized"


# ------------------------------------------------------------
# Site
#
# Your ADRC metadata shown here does not contain a clear SITE
# column, so for now ADRC is treated as one acquisition group.
# If you later find scanner/site metadata, replace this.
# ------------------------------------------------------------

site_col = None


print("\nSelected ADRC columns:")

print(
    "Subject:",
    subject_col,
)

print(
    "Age:",
    age_col,
)

print(
    "Sex:",
    sex_col,
)

print(
    "Diagnosis:",
    diagnosis_col,
)

print(
    "Site:",
    site_col,
)


# ============================================================
# 3. STANDARDIZE ADRC METADATA
# ============================================================

adrc_meta = pd.DataFrame()


# ------------------------------------------------------------
# Subject ID
# ------------------------------------------------------------

adrc_meta["subject_id"] = (
    adrc_meta_raw[
        subject_col
    ]
    .astype("string")
    .str.strip()
)


# ------------------------------------------------------------
# Matching key
#
# IMPORTANT:
# Your metadata already contains cleaned matching columns.
# Prefer CONNECTOME_KEY_USED_FOR_INTERSECTION, then
# CONNECTOME_KEY_CLEAN, then match_id, then PTID_clean.
# ------------------------------------------------------------

match_col_candidates = [
    "CONNECTOME_KEY_USED_FOR_INTERSECTION",
    "CONNECTOME_KEY_CLEAN",
    "match_id",
    "PTID_clean",
]

match_col = None

for col in match_col_candidates:
    if col in adrc_meta_raw.columns:
        if adrc_meta_raw[col].notna().any():
            match_col = col
            break


if match_col is None:
    raise ValueError(
        "Could not find a usable ADRC connectome matching column."
    )


print(
    "\nUsing ADRC matching column:",
    match_col,
)


adrc_meta["MATCH_KEY"] = (
    adrc_meta_raw[
        match_col
    ]
    .astype("string")
    .str.strip()
    .str.lower()
)


# ------------------------------------------------------------
# Age
# ------------------------------------------------------------

adrc_meta["AGE"] = pd.to_numeric(
    adrc_meta_raw[
        age_col
    ],
    errors="coerce",
)


# ------------------------------------------------------------
# Sex
# ------------------------------------------------------------

adrc_meta["SEX"] = standardize_sex(
    adrc_meta_raw[
        sex_col
    ]
)


# ------------------------------------------------------------
# Diagnosis
# ------------------------------------------------------------

adrc_meta["GROUP"] = (
    standardize_diagnosis(
        adrc_meta_raw[
            diagnosis_col
        ]
    )
)


# ------------------------------------------------------------
# Site
# ------------------------------------------------------------

adrc_meta["SITE"] = "ADRC"


# ------------------------------------------------------------
# Keep original matched filename if available
# ------------------------------------------------------------

if "MATCHED_CONNECTOME_FILE" in adrc_meta_raw.columns:

    adrc_meta[
        "MATCHED_CONNECTOME_FILE"
    ] = (
        adrc_meta_raw[
            "MATCHED_CONNECTOME_FILE"
        ]
        .astype("string")
        .str.strip()
    )


# ============================================================
# 4. REMOVE INVALID METADATA ROWS
# ============================================================

valid_meta = (
    adrc_meta[
        [
            "MATCH_KEY",
            "AGE",
            "SEX",
        ]
    ]
    .notna()
    .all(axis=1)
)


print(
    "\nMetadata rows before filtering:",
    len(adrc_meta),
)

print(
    "Rows removed for missing MATCH_KEY/AGE/SEX:",
    int(
        (~valid_meta).sum()
    ),
)


adrc_meta = (
    adrc_meta.loc[
        valid_meta
    ]
    .reset_index(drop=True)
)


print(
    "Metadata rows after filtering:",
    len(adrc_meta),
)


# ============================================================
# 5. DISCOVER ADRC DWI FILES
# ============================================================

candidate_files = sorted([
    f
    for f in ADRC_DWI_DIR.rglob("*")
    if (
        f.is_file()
        and f.suffix.lower()
        in [
            ".csv",
            ".txt",
        ]
    )
])


print("\n" + "=" * 80)
print("ADRC DWI FILES")
print("=" * 80)

print(
    "Candidate files:",
    len(candidate_files),
)


print(
    "\nFirst 20 files:"
)

for f in candidate_files[:20]:
    print(
        f.name
    )


# ============================================================
# 6. BUILD DWI FILE TABLE
# ============================================================

dwi_rows = []
failed_rows = []


for filepath in candidate_files:

    try:

        matrix = (
            pd.read_csv(
                filepath,
                header=None,
            )
            .to_numpy(
                dtype=float
            )
        )


        if matrix.ndim != 2:

            raise ValueError(
                f"not 2D: {matrix.shape}"
            )


        if (
            matrix.shape[0]
            != matrix.shape[1]
        ):

            raise ValueError(
                f"not square: {matrix.shape}"
            )


        if not np.isfinite(
            matrix
        ).all():

            raise ValueError(
                "contains NaN or infinity"
            )


    except Exception as exc:

        failed_rows.append(
            {
                "filepath": str(
                    filepath
                ),

                "reason": str(
                    exc
                ),
            }
        )

        continue


    # --------------------------------------------------------
    # Build filename matching forms
    # --------------------------------------------------------

    filename = (
        filepath.name
    )

    stem = (
        filepath.stem
    )


    dwi_rows.append(
        {
            "filepath": str(
                filepath
            ),

            "filename": (
                filename
            ),

            "filename_lower": (
                filename.lower()
            ),

            "stem_lower": (
                stem.lower()
            ),

            "matrix_n": (
                matrix.shape[0]
            ),
        }
    )


adrc_dwi = pd.DataFrame(
    dwi_rows
)


failed_df = pd.DataFrame(
    failed_rows
)


print(
    "\nValid DWI matrices:",
    len(adrc_dwi),
)


if len(adrc_dwi) == 0:

    raise ValueError(
        "No valid ADRC DWI matrices found."
    )


print(
    "\nMatrix sizes:"
)

print(
    adrc_dwi[
        "matrix_n"
    ]
    .value_counts()
)


# ============================================================
# 7. MATCH USING EXISTING ADRC MATCH KEY
# ============================================================

matched_rows = []
unmatched_meta_rows = []
ambiguous_rows = []


for idx, row in adrc_meta.iterrows():

    match_key = (
        str(
            row["MATCH_KEY"]
        )
        .strip()
        .lower()
    )


    if (
        match_key == ""
        or match_key == "nan"
    ):

        continue


    # --------------------------------------------------------
    # Strategy 1:
    # MATCHED_CONNECTOME_FILE if metadata already gives filename
    # --------------------------------------------------------

    candidates = pd.DataFrame()


    if (
        "MATCHED_CONNECTOME_FILE"
        in adrc_meta.columns
        and pd.notna(
            row.get(
                "MATCHED_CONNECTOME_FILE"
            )
        )
    ):

        expected_file = (
            str(
                row[
                    "MATCHED_CONNECTOME_FILE"
                ]
            )
            .strip()
            .lower()
        )


        candidates = (
            adrc_dwi.loc[
                adrc_dwi[
                    "filename_lower"
                ]
                ==
                expected_file
            ]
        )


    # --------------------------------------------------------
    # Strategy 2:
    # MATCH_KEY appears in filename/stem
    # --------------------------------------------------------

    if len(candidates) == 0:

        mask = (
            adrc_dwi[
                "filename_lower"
            ]
            .str.contains(
                re.escape(
                    match_key
                ),
                regex=True,
                na=False,
            )
            |
            adrc_dwi[
                "stem_lower"
            ]
            .str.contains(
                re.escape(
                    match_key
                ),
                regex=True,
                na=False,
            )
        )


        candidates = (
            adrc_dwi.loc[
                mask
            ]
        )


    # --------------------------------------------------------
    # Resolve
    # --------------------------------------------------------

    if len(candidates) == 1:

        dwi_row = (
            candidates.iloc[0]
        )


        merged_row = (
            row.to_dict()
        )


        merged_row[
            "filepath"
        ] = (
            dwi_row[
                "filepath"
            ]
        )


        merged_row[
            "matrix_n"
        ] = (
            dwi_row[
                "matrix_n"
            ]
        )


        matched_rows.append(
            merged_row
        )


    elif len(candidates) == 0:

        unmatched_meta_rows.append(
            {
                "subject_id":
                    row[
                        "subject_id"
                    ],

                "MATCH_KEY":
                    match_key,
            }
        )


    else:

        ambiguous_rows.append(
            {
                "subject_id":
                    row[
                        "subject_id"
                    ],

                "MATCH_KEY":
                    match_key,

                "n_candidates":
                    len(
                        candidates
                    ),
            }
        )


# ============================================================
# 8. BUILD MATCHED TABLE
# ============================================================

adrc_matched = pd.DataFrame(
    matched_rows
)


unmatched_meta_df = pd.DataFrame(
    unmatched_meta_rows
)


ambiguous_df = pd.DataFrame(
    ambiguous_rows
)


print("\n" + "=" * 80)
print("ADRC ALIGNMENT RESULT")
print("=" * 80)


print(
    "Metadata rows:",
    len(adrc_meta),
)


print(
    "Valid DWI matrices:",
    len(adrc_dwi),
)


print(
    "Matched:",
    len(adrc_matched),
)


print(
    "Unmatched metadata:",
    len(unmatched_meta_df),
)


print(
    "Ambiguous:",
    len(ambiguous_df),
)


if len(
    unmatched_meta_df
) > 0:

    print(
        "\nFirst unmatched metadata rows:"
    )

    display(
        unmatched_meta_df.head(
            20
        )
    )


if len(
    ambiguous_df
) > 0:

    print(
        "\nFirst ambiguous matches:"
    )

    display(
        ambiguous_df.head(
            20
        )
    )


if len(
    adrc_matched
) == 0:

    raise ValueError(
        "ADRC metadata and DWI still produced "
        "0 matches. Inspect MATCH_KEY values "
        "and DWI filenames."
    )


# ============================================================
# 9. REMOVE DUPLICATE FILE/SUBJECT MATCHES
# ============================================================

print(
    "\nDuplicate subject IDs:",
    int(
        adrc_matched[
            "subject_id"
        ]
        .duplicated()
        .sum()
    ),
)


print(
    "Duplicate filepaths:",
    int(
        adrc_matched[
            "filepath"
        ]
        .duplicated()
        .sum()
    ),
)


adrc_matched = (
    adrc_matched
    .drop_duplicates(
        subset=[
            "subject_id"
        ],
        keep="first",
    )
    .drop_duplicates(
        subset=[
            "filepath"
        ],
        keep="first",
    )
    .reset_index(drop=True)
)


# ============================================================
# 10. CREATE __id__
# ============================================================

adrc_matched[
    "__id__"
] = (
    adrc_matched[
        "MATCH_KEY"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# 11. BUILD COVARS
# ============================================================

adrc_covars = (
    adrc_matched[
        [
            "subject_id",
            "SITE",
            "AGE",
            "SEX",
            "GROUP",
            "__id__",
        ]
    ]
    .copy()
)


# ============================================================
# 12. BUILD MANIFEST
# ============================================================

adrc_manifest = (
    adrc_matched[
        [
            "__id__",
            "filepath",
        ]
    ]
    .copy()
)


# ============================================================
# 13. FINAL CONSISTENCY CHECK
# ============================================================

if len(
    adrc_covars
) != len(
    adrc_manifest
):

    raise ValueError(
        "ADRC covars and manifest "
        "row counts differ."
    )


if set(
    adrc_covars[
        "__id__"
    ]
) != set(
    adrc_manifest[
        "__id__"
    ]
):

    raise ValueError(
        "ADRC covars and manifest "
        "__id__ values differ."
    )


# ============================================================
# 14. SAVE
# ============================================================

adrc_covars.to_csv(
    ADRC_PREP_DIR
    / "covars_aligned.csv",
    index=False,
)


adrc_manifest.to_csv(
    ADRC_PREP_DIR
    / "features_manifest_aligned.csv",
    index=False,
)


failed_df.to_csv(
    ADRC_PREP_DIR
    / "failed_dwi_files.csv",
    index=False,
)


unmatched_meta_df.to_csv(
    ADRC_PREP_DIR
    / "unmatched_metadata.csv",
    index=False,
)


ambiguous_df.to_csv(
    ADRC_PREP_DIR
    / "ambiguous_matches.csv",
    index=False,
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ADRC PREP COMPLETE")
print("=" * 80)


print(
    "Final subjects:",
    len(
        adrc_covars
    )
)


print(
    "covars shape:",
    adrc_covars.shape
)


print(
    "manifest shape:",
    adrc_manifest.shape
)


print(
    "\nDiagnosis counts:"
)


print(
    adrc_covars[
        "GROUP"
    ]
    .value_counts(
        dropna=False
    )
)


print(
    "\nAge summary:"
)


print(
    adrc_covars[
        "AGE"
    ]
    .describe()
)


print(
    "\nSaved to:"
)

print(
    ADRC_PREP_DIR
)

# %% [markdown]
# # Load all four cohorts into one standard format

# %%
# ============================================================
# CELL 4 — LOAD FOUR COHORTS
# ============================================================


def load_prepared_cohort(
    cohort,
):

    prep_dir = (
        PREP_DIRS[
            cohort
        ]
    )


    covars_path = (
        prep_dir
        / "covars_aligned.csv"
    )


    manifest_path = (
        prep_dir
        / "features_manifest_aligned.csv"
    )


    if not covars_path.exists():

        raise FileNotFoundError(
            covars_path
        )


    if not manifest_path.exists():

        raise FileNotFoundError(
            manifest_path
        )


    covars = pd.read_csv(
        covars_path,
        low_memory=False,
    )


    manifest = pd.read_csv(
        manifest_path,
        low_memory=False,
    )


    # ========================================================
    # Normalize __id__
    # ========================================================

    covars["__id__"] = (
        covars["__id__"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    manifest["__id__"] = (
        manifest["__id__"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    # ========================================================
    # Merge
    # ========================================================

    df = (
        covars.merge(

            manifest[
                [
                    "__id__",
                    "filepath",
                ]
            ],

            on="__id__",

            how="inner",

            validate="one_to_one",
        )
    )


    # ========================================================
    # Standardize column names
    # ========================================================

    age_col = find_column(
        df,
        [
            "AGE",
            "Age",
            "age",
        ],
    )


    sex_col = find_column(
        df,
        [
            "SEX",
            "Sex",
            "sex",
        ],
    )


    site_col = find_column(
        df,
        [
            "SITE",
            "Site",
            "site",
        ],
        required=False,
    )


    diagnosis_col = find_column(
        df,
        [
            "GROUP",
            "Group",
            "Diagnosis",
            "DIAGNOSIS",
            "DX",
        ],
        required=False,
    )


    subject_col = find_column(
        df,
        [
            "subject_id",
            "Subject",
            "SUBJECT",
            "ID",
        ],
        required=False,
    )


    # ========================================================
    # Output standard format
    # ========================================================

    result = pd.DataFrame()


    result["__id__"] = (
        df["__id__"]
    )


    if subject_col is not None:

        result["subject_id"] = (
            df[
                subject_col
            ].astype(str)
        )

    else:

        result["subject_id"] = (
            df["__id__"]
        )


    result["DATASET"] = (
        cohort
    )


    result["AGE"] = pd.to_numeric(
        df[
            age_col
        ],
        errors="coerce",
    )


    result["SEX"] = standardize_sex(
        df[
            sex_col
        ]
    )


    if site_col is None:

        result["SITE"] = (
            cohort
        )

    else:

        result["SITE"] = (
            df[
                site_col
            ]
            .astype("string")
            .str.strip()
        )


    if diagnosis_col is None:

        result["GROUP"] = (
            pd.NA
        )

    else:

        result["GROUP"] = (
            standardize_diagnosis(
                df[
                    diagnosis_col
                ]
            )
        )


    result["filepath"] = (
        df["filepath"]
    )


    print(
        f"{cohort:10s}",
        "covars=",
        len(covars),
        "manifest=",
        len(manifest),
        "matched=",
        len(result),
    )


    return result


# ============================================================
# LOAD
# ============================================================

adni = load_prepared_cohort(
    "ADNI"
)

habs = load_prepared_cohort(
    "HABS"
)

adrc = load_prepared_cohort(
    "ADRC"
)

addecode = load_prepared_cohort(
    "ADDECODE"
)


# ============================================================
# COMBINE
# ============================================================

metadata = pd.concat(
    [
        adni,
        habs,
        adrc,
        addecode,
    ],
    ignore_index=True,
)


print("\n" + "=" * 80)
print("FOUR-COHORT METADATA")
print("=" * 80)


print(
    "Total:",
    len(metadata)
)


print(
    "\nDataset counts:"
)


print(
    metadata[
        "DATASET"
    ]
    .value_counts()
)

# %% [markdown]
# # Create DATASET_SITE

# %%
# ============================================================
# CELL 5 — CREATE JOINT DATASET_SITE BATCH
# ============================================================

metadata = (
    metadata
    .copy()
)


# ============================================================
# Clean site
# ============================================================

metadata["SITE"] = (
    metadata["SITE"]
    .astype("string")
    .str.strip()
)


metadata["SITE"] = metadata[
    "SITE"
].replace(
    {
        "": pd.NA,
        "nan": pd.NA,
        "NaN": pd.NA,
        "None": pd.NA,
    }
)


# ============================================================
# Missing site fallback
# ============================================================

missing_site = (
    metadata[
        "SITE"
    ].isna()
)


if missing_site.any():

    print(
        "Missing SITE:",
        int(
            missing_site.sum()
        )
    )


    metadata.loc[
        missing_site,
        "SITE"
    ] = (
        metadata.loc[
            missing_site,
            "DATASET"
        ]
    )


# ============================================================
# JOINT BATCH
# ============================================================

metadata[
    "DATASET_SITE"
] = (

    metadata[
        "DATASET"
    ]
    .astype(str)
    .str.upper()

    + "__"

    + metadata[
        "SITE"
    ]
    .astype(str)
)


# ============================================================
# Summary
# ============================================================

print("=" * 80)
print("DATASET × SITE TECHNICAL BATCH")
print("=" * 80)


print(
    "Datasets:",
    metadata[
        "DATASET"
    ].nunique()
)


print(
    "Joint batches:",
    metadata[
        "DATASET_SITE"
    ].nunique()
)


print("\nSites per dataset:")


print(
    metadata
    .groupby(
        "DATASET"
    )["SITE"]
    .nunique()
)


print("\nBatch counts:")


print(
    metadata[
        "DATASET_SITE"
    ]
    .value_counts()
    .sort_index()
)

# %% [markdown]
# # Read all DWI matrices → 3486 edges

# %%
# ============================================================
# CELL 6 — LOAD DWI CONNECTOMES
# ============================================================


def read_connectome(
    filepath,
):

    matrix = (
        pd.read_csv(
            filepath,
            header=None,
        )
        .to_numpy(
            dtype=float
        )
    )


    if matrix.ndim != 2:

        raise ValueError(
            f"Not 2-D: {matrix.shape}"
        )


    if (
        matrix.shape[0]
        != matrix.shape[1]
    ):

        raise ValueError(
            f"Not square: {matrix.shape}"
        )


    if not np.isfinite(
        matrix
    ).all():

        raise ValueError(
            "Contains NaN or infinity"
        )


    return matrix


def matrix_to_edges(
    matrix,
):

    upper = (
        np.triu_indices_from(
            matrix,
            k=1,
        )
    )

    return (
        matrix[
            upper
        ]
    )


# ============================================================
# LOAD
# ============================================================

X_list = []
used_rows = []
failed_rows = []


expected_shape = None
expected_edges = None


for _, row in metadata.iterrows():

    try:

        matrix = read_connectome(
            row[
                "filepath"
            ]
        )


        # ----------------------------------------------------
        # Same atlas across all cohorts
        # ----------------------------------------------------

        if expected_shape is None:

            expected_shape = (
                matrix.shape
            )

        elif (
            matrix.shape
            != expected_shape
        ):

            raise ValueError(
                f"Atlas mismatch: "
                f"{matrix.shape} "
                f"vs expected "
                f"{expected_shape}"
            )


        edges = matrix_to_edges(
            matrix
        )


        if expected_edges is None:

            expected_edges = (
                len(edges)
            )

        elif (
            len(edges)
            != expected_edges
        ):

            raise ValueError(
                "Edge number mismatch"
            )


        X_list.append(
            edges
        )


        used_rows.append(
            row.to_dict()
        )


    except Exception as exc:

        failed_rows.append(
            {
                "DATASET":
                    row[
                        "DATASET"
                    ],

                "subject_id":
                    row[
                        "subject_id"
                    ],

                "filepath":
                    row[
                        "filepath"
                    ],

                "error":
                    str(exc),
            }
        )


# ============================================================
# FINAL MATRICES
# ============================================================

X_raw = np.asarray(
    X_list,
    dtype=float,
)


metadata_aligned = (
    pd.DataFrame(
        used_rows
    )
    .reset_index(
        drop=True
    )
)


failed_df = pd.DataFrame(
    failed_rows
)


print("=" * 80)
print("CONNECTOME LOADING")
print("=" * 80)


print(
    "Atlas shape:",
    expected_shape
)


print(
    "Number edges:",
    expected_edges
)


print(
    "X_raw:",
    X_raw.shape
)


print(
    "Metadata:",
    metadata_aligned.shape
)


print(
    "Failed:",
    len(failed_df)
)


print("\nLoaded by dataset:")


print(
    metadata_aligned[
        "DATASET"
    ]
    .value_counts()
)


if len(
    failed_df
) > 0:

    display(
        failed_df.head(20)
    )


failed_df.to_csv(
    OUT_META
    / "failed_connectomes.csv",
    index=False,
)

# %% [markdown]
# # Pre-ComBat QC

# %%
# ============================================================
# CELL 7 — PRE-COMBAT QC
# ============================================================

BATCH_COL = (
    "DATASET_SITE"
)


# ============================================================
# 1. Basic covariates
# ============================================================

metadata_aligned["AGE"] = pd.to_numeric(
    metadata_aligned[
        "AGE"
    ],
    errors="coerce",
)


metadata_aligned["SEX"] = (
    standardize_sex(
        metadata_aligned[
            "SEX"
        ]
    )
)


metadata_aligned["GROUP"] = (
    standardize_diagnosis(
        metadata_aligned[
            "GROUP"
        ]
    )
)


# ============================================================
# 2. Keep Age + Sex complete
# ============================================================

valid_subjects = (
    metadata_aligned[
        [
            BATCH_COL,
            "AGE",
            "SEX",
        ]
    ]
    .notna()
    .all(axis=1)
)


print(
    "Missing Batch/Age/Sex:",
    int(
        (~valid_subjects)
        .sum()
    ),
)


X_raw = (
    X_raw[
        valid_subjects.values
    ]
)


metadata_aligned = (
    metadata_aligned.loc[
        valid_subjects
    ]
    .reset_index(
        drop=True
    )
)


# ============================================================
# 3. Remove singleton DATASET_SITE batches
# ============================================================

batch_counts = (
    metadata_aligned[
        BATCH_COL
    ]
    .value_counts()
)


singleton_batches = (
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
        singleton_batches
    )
)


print(
    "Singleton batches:",
    len(
        singleton_batches
    )
)


print(
    "Subjects removed:",
    int(
        singleton_mask.sum()
    )
)


if singleton_mask.any():

    print(
        metadata_aligned.loc[
            singleton_mask,
            BATCH_COL
        ]
        .value_counts()
    )


X_raw = (
    X_raw[
        ~singleton_mask.values
    ]
)


metadata_aligned = (
    metadata_aligned.loc[
        ~singleton_mask
    ]
    .reset_index(
        drop=True
    )
)


# ============================================================
# 4. Remove invalid / constant edges
# ============================================================

finite_edges = (
    np.isfinite(
        X_raw
    )
    .all(axis=0)
)


variable_edges = (
    np.std(
        X_raw,
        axis=0,
    )
    > 0
)


good_edge_mask = (
    finite_edges
    &
    variable_edges
)


X_clean = (
    X_raw[
        :,
        good_edge_mask
    ]
)


print("\n" + "=" * 80)
print("FINAL QC INPUT")
print("=" * 80)


print(
    "Subjects:",
    X_clean.shape[0]
)


print(
    "Features:",
    X_clean.shape[1]
)


print(
    "Removed features:",
    int(
        (~good_edge_mask)
        .sum()
    )
)


print(
    "Technical batches:",
    metadata_aligned[
        BATCH_COL
    ].nunique()
)


print(
    "Smallest batch:",
    metadata_aligned[
        BATCH_COL
    ]
    .value_counts()
    .min()
)


print("\nDataset counts:")


print(
    metadata_aligned[
        "DATASET"
    ]
    .value_counts()
)

# %% [markdown]
# # DWI log1p

# %%
# ============================================================
# CELL 8 — DWI LOG1P TRANSFORM
# ============================================================


minimum_value = (
    np.min(
        X_clean
    )
)


print(
    "Minimum raw edge:",
    minimum_value
)


if minimum_value < 0:

    raise ValueError(
        "Negative DWI values detected. "
        "Do not apply log1p."
    )


X_before_combat = (
    np.log1p(
        X_clean
    )
)


print(
    "Before ComBat:",
    X_before_combat.shape
)


print(
    "Finite:",
    np.isfinite(
        X_before_combat
    ).all()
)

# %% [markdown]
# # Decide whether Diagnosis can safely be preserved

# %%
# ============================================================
# CELL 9 — COMBAT DESIGN CHECK
# ============================================================


# ============================================================
# Diagnosis completeness
# ============================================================

diagnosis_complete = (
    metadata_aligned[
        "GROUP"
    ]
    .notna()
    .all()
)


print("=" * 80)
print("DIAGNOSIS AVAILABILITY")
print("=" * 80)


print(
    "Diagnosis complete:",
    diagnosis_complete
)


print("\nMissing Diagnosis by dataset:")


print(
    metadata_aligned[
        "GROUP"
    ]
    .isna()
    .groupby(
        metadata_aligned[
            "DATASET"
        ]
    )
    .sum()
)


# ============================================================
# Test full design
# ============================================================

USE_DIAGNOSIS = False


if diagnosis_complete:

    design_test = pd.concat(
        [
            pd.get_dummies(
                metadata_aligned[
                    BATCH_COL
                ],
                prefix="B",
                dtype=float,
            ),

            pd.get_dummies(
                metadata_aligned[
                    "SEX"
                ],
                prefix="SEX",
                drop_first=True,
                dtype=float,
            ),

            pd.get_dummies(
                metadata_aligned[
                    "GROUP"
                ],
                prefix="DX",
                drop_first=True,
                dtype=float,
            ),

            metadata_aligned[
                [
                    "AGE"
                ]
            ].astype(float),
        ],

        axis=1,
    )


    design_matrix = (
        design_test
        .to_numpy(
            dtype=float
        )
    )


    rank = np.linalg.matrix_rank(
        design_matrix
    )


    columns = (
        design_matrix.shape[1]
    )


    print(
        "\nFull design columns:",
        columns
    )


    print(
        "Full design rank:",
        rank
    )


    if rank == columns:

        USE_DIAGNOSIS = True

        print(
            "\nAge + Sex + Diagnosis "
            "can be preserved."
        )

    else:

        print(
            "\nDiagnosis makes the "
            "design rank deficient."
        )

        print(
            "ComBat will preserve "
            "Age + Sex only."
        )


else:

    print(
        "\nDiagnosis incomplete."
    )

    print(
        "ComBat will preserve "
        "Age + Sex only."
    )


print(
    "\nUSE_DIAGNOSIS =",
    USE_DIAGNOSIS
)

# %% [markdown]
# # One joint NeuroCombat

# %%
# ============================================================
# CELL 10 — JOINT WITHIN + BETWEEN NEUROCOMBAT
# ============================================================


# ============================================================
# Covariates
# ============================================================

if USE_DIAGNOSIS:

    covars = (
        metadata_aligned[
            [
                BATCH_COL,
                "AGE",
                "SEX",
                "GROUP",
            ]
        ]
        .copy()
    )


    categorical_cols = [
        "SEX",
        "GROUP",
    ]


else:

    covars = (
        metadata_aligned[
            [
                BATCH_COL,
                "AGE",
                "SEX",
            ]
        ]
        .copy()
    )


    categorical_cols = [
        "SEX",
    ]


continuous_cols = [
    "AGE",
]


# ============================================================
# Summary
# ============================================================

print("=" * 80)
print("JOINT COMBAT INPUT")
print("=" * 80)


print(
    "Subjects × features:",
    X_before_combat.shape
)


print(
    "Technical batches:",
    covars[
        BATCH_COL
    ].nunique()
)


print(
    "Categorical:",
    categorical_cols
)


print(
    "Continuous:",
    continuous_cols
)


print("\nBatch counts:")


print(
    covars[
        BATCH_COL
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# NeuroCombat
# ============================================================

combat_result = neuroCombat(

    # features × subjects
    dat=(
        X_before_combat.T
    ),

    covars=(
        covars
        .reset_index(
            drop=True
        )
    ),

    batch_col=BATCH_COL,

    categorical_cols=(
        categorical_cols
    ),

    continuous_cols=(
        continuous_cols
    ),

    eb=True,

    parametric=True,

    mean_only=False,
)


# ============================================================
# Output
# ============================================================

X_combat = (
    combat_result[
        "data"
    ].T
)


print("\n" + "=" * 80)
print("COMBAT COMPLETE")
print("=" * 80)


print(
    "X_combat:",
    X_combat.shape
)


print(
    "NaN:",
    int(
        np.isnan(
            X_combat
        ).sum()
    )
)


print(
    "Inf:",
    int(
        np.isinf(
            X_combat
        ).sum()
    )
)


if not np.isfinite(
    X_combat
).all():

    raise ValueError(
        "ComBat produced invalid values."
    )

# %% [markdown]
# # Save results

# %%
# ============================================================
# CELL 11 — SAVE RESULTS
# ============================================================


np.save(
    OUT_ARRAYS
    / "DWI_raw_edges.npy",

    X_raw,
)


np.save(
    OUT_ARRAYS
    / "DWI_log1p_before_combat.npy",

    X_before_combat,
)


np.save(
    OUT_ARRAYS
    / "DWI_joint_combat.npy",

    X_combat,
)


np.save(
    OUT_ARRAYS
    / "good_edge_mask.npy",

    good_edge_mask,
)


metadata_aligned.to_csv(
    OUT_META
    / "metadata_used_for_joint_combat.csv",

    index=False,
)


covars.to_csv(
    OUT_META
    / "combat_covariates.csv",

    index=False,
)


print("=" * 80)
print("SAVED")
print("=" * 80)


print(
    "Results:",
    RESULTS_DIR
)


print(
    "Subjects:",
    len(
        metadata_aligned
    )
)


print(
    "Features:",
    X_combat.shape[1]
)


print(
    "Datasets:",
    metadata_aligned[
        "DATASET"
    ].nunique()
)


print(
    "Technical batches:",
    metadata_aligned[
        BATCH_COL
    ].nunique()
)


print(
    "Diagnosis preserved:",
    USE_DIAGNOSIS
)



# %%
# ============================================================
# SAVE HARMONIZED DWI CONNECTOMES
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_CONNECTOME_DIR = Path(
    '/mnt/newStor/paros/paros_WORK/avery/results/final/DWI/harmonized_connectomes'
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
print("SAVE HARMONIZED DWI CONNECTOMES")
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
    # Harmonized log1p edge vector
    # --------------------------------------------------------

    log_edges = np.asarray(
        X_combat[i],
        dtype=float,
    )


    # --------------------------------------------------------
    # Convert log1p values back
    #
    # log_edges = log(1 + x)
    #
    # therefore:
    #
    # x = exp(log_edges) - 1
    # --------------------------------------------------------

    original_scale_edges = np.expm1(
        log_edges
    )


    # --------------------------------------------------------
    # Small negative values can appear after ComBat.
    #
    # Structural connectivity should not be negative.
    # Clip tiny/negative values to zero.
    # --------------------------------------------------------

    original_scale_edges = np.clip(
        original_scale_edges,
        0,
        None,
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
    ] = original_scale_edges


    matrix[
        (
            upper_indices[1],
            upper_indices[0],
        )
    ] = original_scale_edges


    # DWI diagonal

    np.fill_diagonal(
        matrix,
        0.0,
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


    # Your DWI metadata uses __id__
    if "__id__" in metadata_aligned.columns:

        connectome_key = str(
            row[
                "__id__"
            ]
        )


    elif "subject_id" in metadata_aligned.columns:

        connectome_key = str(
            row[
                "subject_id"
            ]
        )


    else:

        connectome_key = (
            f"subject_{i:04d}"
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
        f"{connectome_key}_harmonized_DWI.csv"
    )


    # --------------------------------------------------------
    # Save matrix
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

harmonized_dwi_manifest = pd.DataFrame(
    saved_rows
)


manifest_path = (

    OUTPUT_CONNECTOME_DIR
    /
    "harmonized_DWI_manifest.csv"
)


harmonized_dwi_manifest.to_csv(
    manifest_path,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("DWI HARMONIZED CONNECTOMES SAVED")
print("=" * 80)


print(
    "Total:",
    len(
        harmonized_dwi_manifest
    )
)


print(
    "\nBy dataset:"
)


print(

    harmonized_dwi_manifest[
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
