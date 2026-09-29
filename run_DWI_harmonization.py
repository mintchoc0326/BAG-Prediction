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
    / "harmonization"
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
    / "DWI_4cohort_joint_ComBat"
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

# %% [markdown]
# # PCA before vs after, colored by DATASET_SITE

# %%
# ============================================================
# CELL 12 — PCA BY DATASET_SITE
# ============================================================

import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.decomposition import (
    PCA
)


# ============================================================
# Scale
# ============================================================

before_scaled = (
    StandardScaler()
    .fit_transform(
        X_before_combat
    )
)


after_scaled = (
    StandardScaler()
    .fit_transform(
        X_combat
    )
)


# ============================================================
# PCA
# ============================================================

pca_before = PCA(
    n_components=2
)

pca_after = PCA(
    n_components=2
)


Z_before = (
    pca_before
    .fit_transform(
        before_scaled
    )
)


Z_after = (
    pca_after
    .fit_transform(
        after_scaled
    )
)


# ============================================================
# Colors = DATASET_SITE
# ============================================================

batch_codes, batch_names = (
    pd.factorize(
        metadata_aligned[
            BATCH_COL
        ]
    )
)


# ============================================================
# Same axes
# ============================================================

all_pc1 = np.concatenate(
    [
        Z_before[:, 0],
        Z_after[:, 0],
    ]
)


all_pc2 = np.concatenate(
    [
        Z_before[:, 1],
        Z_after[:, 1],
    ]
)


xmin, xmax = (
    all_pc1.min(),
    all_pc1.max(),
)


ymin, ymax = (
    all_pc2.min(),
    all_pc2.max(),
)


mx = (
    0.05
    * (
        xmax
        - xmin
    )
)


my = (
    0.05
    * (
        ymax
        - ymin
    )
)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        12,
        5,
    ),
    sharex=True,
    sharey=True,
)


axes[0].scatter(
    Z_before[:, 0],
    Z_before[:, 1],
    c=batch_codes,
    s=25,
    alpha=0.75,
)


axes[0].set_title(
    "Before Joint ComBat"
)


axes[0].set_xlabel(
    "PC1"
)


axes[0].set_ylabel(
    "PC2"
)


axes[0].set_xlim(
    xmin - mx,
    xmax + mx,
)


axes[0].set_ylim(
    ymin - my,
    ymax + my,
)


axes[1].scatter(
    Z_after[:, 0],
    Z_after[:, 1],
    c=batch_codes,
    s=25,
    alpha=0.75,
)


axes[1].set_title(
    "After Joint ComBat"
)


axes[1].set_xlabel(
    "PC1"
)


axes[1].set_xlim(
    xmin - mx,
    xmax + mx,
)


axes[1].set_ylim(
    ymin - my,
    ymax + my,
)


plt.tight_layout()


plt.savefig(
    OUT_FIGURES
    / "PCA_DATASET_SITE_before_after.png",

    dpi=300,

    bbox_inches="tight",
)


plt.show()


print(
    "Before explained variance:",
    pca_before
    .explained_variance_ratio_
)


print(
    "After explained variance:",
    pca_after
    .explained_variance_ratio_
)

# %% [markdown]
# # Evaluation imports + alignment check

# %%
# ============================================================
# CELL 13 — EVALUATION SETUP
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# Convert to arrays
# ============================================================

X_before_eval = np.asarray(
    X_before_combat,
    dtype=float,
)

X_after_eval = np.asarray(
    X_combat,
    dtype=float,
)


eval_meta = (
    metadata_aligned
    .reset_index(drop=True)
    .copy()
)


# ============================================================
# Alignment checks
# ============================================================

assert X_before_eval.shape == X_after_eval.shape

assert (
    len(X_before_eval)
    == len(eval_meta)
)


print("=" * 80)
print("EVALUATION INPUT")
print("=" * 80)

print(
    "Before:",
    X_before_eval.shape
)

print(
    "After:",
    X_after_eval.shape
)

print(
    "Metadata:",
    eval_meta.shape
)


print(
    "\nDatasets:"
)

print(
    eval_meta[
        "DATASET"
    ]
    .value_counts()
)

# %% [markdown]
# # Define participant groups

# %%
# ============================================================
# CELL 14 — PARTICIPANT GROUPS FOR CV
# ============================================================

def get_participant_group(row):

    dataset = str(
        row["DATASET"]
    ).upper()

    subject = str(
        row["subject_id"]
    ).strip()


    # --------------------------------------------------------
    # ADNI longitudinal IDs:
    #
    # R0072_y0
    # R0072_y4
    #
    # -> R0072
    # --------------------------------------------------------

    if dataset == "ADNI":

        subject = (
            subject
            .split("_y")[0]
        )


    # Add dataset prefix so different cohorts
    # cannot accidentally share same group ID
    return (
        dataset
        + "__"
        + subject
    )


eval_meta[
    "PARTICIPANT_GROUP"
] = (
    eval_meta
    .apply(
        get_participant_group,
        axis=1,
    )
)


print(
    "Scans:",
    len(eval_meta)
)

print(
    "Unique participant groups:",
    eval_meta[
        "PARTICIPANT_GROUP"
    ].nunique()
)


print(
    "\nExample:"
)

display(
    eval_meta[
        [
            "DATASET",
            "subject_id",
            "PARTICIPANT_GROUP",
        ]
    ].head(20)
)

# %% [markdown]
# # General before/after classifier function

# %%
# ============================================================
# DIAGNOSIS × DATASET CHECK
# ============================================================

diagnosis_dataset_table = pd.crosstab(
    eval_meta["DATASET"],
    eval_meta["GROUP"],
    margins=True,
)

display(
    diagnosis_dataset_table
)


print("\n" + "=" * 80)
print("ROW PERCENTAGE — DIAGNOSIS DISTRIBUTION WITHIN EACH DATASET")
print("=" * 80)


diagnosis_dataset_pct = pd.crosstab(
    eval_meta["DATASET"],
    eval_meta["GROUP"],
    normalize="index",
) * 100


display(
    diagnosis_dataset_pct.round(1)
)

# %%
# ============================================================
# CELL 15 — GENERAL BEFORE / AFTER CLASSIFIER
# ============================================================

def evaluate_classifier_before_after(
    X_before,
    X_after,
    y,
    groups,
    label_name,
    n_splits=5,
):

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    X_before = np.asarray(
        X_before,
        dtype=float,
    )

    X_after = np.asarray(
        X_after,
        dtype=float,
    )

    y = np.asarray(
        y
    )

    groups = np.asarray(
        groups
    )


    # --------------------------------------------------------
    # Remove missing labels
    # --------------------------------------------------------

    valid = (
        pd.notna(y)
    )


    Xb = X_before[
        valid
    ]

    Xa = X_after[
        valid
    ]

    yv = y[
        valid
    ]

    gv = groups[
        valid
    ]


    print("\n" + "=" * 80)
    print(label_name.upper())
    print("=" * 80)

    print(
        "Subjects/scans:",
        len(yv)
    )

    print(
        "Classes:",
        pd.Series(
            yv
        ).value_counts().to_dict()
    )


    # --------------------------------------------------------
    # Need at least 2 classes
    # --------------------------------------------------------

    classes = np.unique(
        yv
    )

    if len(classes) < 2:

        raise ValueError(
            f"{label_name}: fewer than 2 classes."
        )


    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )


    model = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),

            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear"
                    if len(classes) == 2
                    else "lbfgs",
                ),
            ),
        ]
    )


    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    results = {
        "before_accuracy": [],
        "after_accuracy": [],

        "before_balanced_accuracy": [],
        "after_balanced_accuracy": [],

        "before_auc": [],
        "after_auc": [],
    }


    before_true_all = []
    before_prob_all = []

    after_true_all = []
    after_prob_all = []


    # ========================================================
    # SAME folds for before and after
    # ========================================================

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        cv.split(
            Xb,
            yv,
            groups=gv,
        ),
        start=1,
    ):

        Xb_train = Xb[
            train_idx
        ]

        Xb_test = Xb[
            test_idx
        ]

        Xa_train = Xa[
            train_idx
        ]

        Xa_test = Xa[
            test_idx
        ]

        y_train = yv[
            train_idx
        ]

        y_test = yv[
            test_idx
        ]


        # ====================================================
        # BEFORE
        # ====================================================

        model.fit(
            Xb_train,
            y_train,
        )

        pred_before = (
            model.predict(
                Xb_test
            )
        )

        prob_before = (
            model.predict_proba(
                Xb_test
            )
        )


        # ====================================================
        # AFTER
        # ====================================================

        model.fit(
            Xa_train,
            y_train,
        )

        pred_after = (
            model.predict(
                Xa_test
            )
        )

        prob_after = (
            model.predict_proba(
                Xa_test
            )
        )


        # ====================================================
        # Accuracy
        # ====================================================

        results[
            "before_accuracy"
        ].append(
            accuracy_score(
                y_test,
                pred_before,
            )
        )

        results[
            "after_accuracy"
        ].append(
            accuracy_score(
                y_test,
                pred_after,
            )
        )


        results[
            "before_balanced_accuracy"
        ].append(
            balanced_accuracy_score(
                y_test,
                pred_before,
            )
        )

        results[
            "after_balanced_accuracy"
        ].append(
            balanced_accuracy_score(
                y_test,
                pred_after,
            )
        )


        # ====================================================
        # AUC
        # ====================================================

        try:

            if len(classes) == 2:

                positive_class = (
                    model.classes_[1]
                )


                before_auc = (
                    roc_auc_score(
                        (
                            y_test
                            ==
                            positive_class
                        ).astype(int),

                        prob_before[
                            :,
                            1
                        ],
                    )
                )


                after_auc = (
                    roc_auc_score(
                        (
                            y_test
                            ==
                            positive_class
                        ).astype(int),

                        prob_after[
                            :,
                            1
                        ],
                    )
                )


            else:

                before_auc = (
                    roc_auc_score(
                        y_test,
                        prob_before,
                        multi_class="ovr",
                        average="weighted",
                        labels=model.classes_,
                    )
                )


                after_auc = (
                    roc_auc_score(
                        y_test,
                        prob_after,
                        multi_class="ovr",
                        average="weighted",
                        labels=model.classes_,
                    )
                )


        except ValueError:

            before_auc = np.nan
            after_auc = np.nan


        results[
            "before_auc"
        ].append(
            before_auc
        )

        results[
            "after_auc"
        ].append(
            after_auc
        )


        print(
            f"Fold {fold}: "
            f"BA before={results['before_balanced_accuracy'][-1]:.3f}, "
            f"after={results['after_balanced_accuracy'][-1]:.3f}"
        )


    # ========================================================
    # Summary
    # ========================================================

    summary = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Balanced Accuracy",
                "ROC AUC",
            ],

            "Before Mean": [
                np.nanmean(
                    results[
                        "before_accuracy"
                    ]
                ),

                np.nanmean(
                    results[
                        "before_balanced_accuracy"
                    ]
                ),

                np.nanmean(
                    results[
                        "before_auc"
                    ]
                ),
            ],

            "Before SD": [
                np.nanstd(
                    results[
                        "before_accuracy"
                    ]
                ),

                np.nanstd(
                    results[
                        "before_balanced_accuracy"
                    ]
                ),

                np.nanstd(
                    results[
                        "before_auc"
                    ]
                ),
            ],

            "After Mean": [
                np.nanmean(
                    results[
                        "after_accuracy"
                    ]
                ),

                np.nanmean(
                    results[
                        "after_balanced_accuracy"
                    ]
                ),

                np.nanmean(
                    results[
                        "after_auc"
                    ]
                ),
            ],

            "After SD": [
                np.nanstd(
                    results[
                        "after_accuracy"
                    ]
                ),

                np.nanstd(
                    results[
                        "after_balanced_accuracy"
                    ]
                ),

                np.nanstd(
                    results[
                        "after_auc"
                    ]
                ),
            ],
        }
    )


    print(
        "\nSummary:"
    )

    display(
        summary
    )


    return {
        "summary": summary,
        "fold_results": results,
    }

# %% [markdown]
# # Dataset classifier

# %%
# ============================================================
# CELL 16 — DATASET CLASSIFIER
#
# Measures BETWEEN-COHORT technical information
# ============================================================

dataset_result = (
    evaluate_classifier_before_after(

        X_before=
            X_before_eval,

        X_after=
            X_after_eval,

        y=
            eval_meta[
                "DATASET"
            ].values,

        groups=
            eval_meta[
                "PARTICIPANT_GROUP"
            ].values,

        label_name=
            "Dataset Classification",

        n_splits=5,
    )
)

# %% [markdown]
# # Within-cohort Site classifier

# %%
# ============================================================
# WITHIN-COHORT SITE CLASSIFICATION
#
# Evaluate site predictability separately inside each cohort.
#
# Expected after harmonization:
#   Balanced Accuracy should decrease strongly.
#
# No multiclass ROC AUC here.
# ============================================================


def evaluate_within_cohort_site(
    X_before,
    X_after,
    meta,
    cohort_name,
    min_site_size=5,
    n_splits=5,
):

    # --------------------------------------------------------
    # Select cohort
    # --------------------------------------------------------

    cohort_mask = (
        meta["DATASET"]
        ==
        cohort_name
    )


    cohort_meta = (
        meta.loc[
            cohort_mask
        ]
        .reset_index(drop=True)
        .copy()
    )


    Xb = (
        X_before[
            cohort_mask.values
        ]
    )


    Xa = (
        X_after[
            cohort_mask.values
        ]
    )


    print("\n" + "=" * 80)
    print(
        f"{cohort_name} WITHIN-COHORT SITE CLASSIFICATION"
    )
    print("=" * 80)


    # --------------------------------------------------------
    # Site counts
    # --------------------------------------------------------

    site_counts = (
        cohort_meta[
            "SITE"
        ]
        .value_counts()
    )


    print(
        "Original number of sites:",
        len(site_counts)
    )


    print(
        "\nOriginal site counts:"
    )

    print(
        site_counts
    )


    # --------------------------------------------------------
    # Keep usable sites
    # --------------------------------------------------------

    usable_sites = (
        site_counts[
            site_counts >= min_site_size
        ]
        .index
    )


    usable_mask = (
        cohort_meta[
            "SITE"
        ]
        .isin(
            usable_sites
        )
    )


    cohort_meta = (
        cohort_meta.loc[
            usable_mask
        ]
        .reset_index(drop=True)
    )


    Xb = (
        Xb[
            usable_mask.values
        ]
    )


    Xa = (
        Xa[
            usable_mask.values
        ]
    )


    n_sites = (
        cohort_meta[
            "SITE"
        ]
        .nunique()
    )


    print(
        "\nUsable sites:",
        n_sites
    )


    print(
        "Usable scans:",
        len(
            cohort_meta
        )
    )


    # --------------------------------------------------------
    # Cannot classify if <2 sites
    # --------------------------------------------------------

    if n_sites < 2:

        print(
            "Skipped: fewer than 2 usable sites."
        )

        return None


    # --------------------------------------------------------
    # Labels + participant groups
    # --------------------------------------------------------

    y = (
        cohort_meta[
            "SITE"
        ]
        .astype(str)
        .values
    )


    groups = (
        cohort_meta[
            "PARTICIPANT_GROUP"
        ]
        .values
    )


    # --------------------------------------------------------
    # Safe BA function
    # --------------------------------------------------------

    def safe_balanced_accuracy(
        y_true,
        y_pred,
    ):

        present_classes = (
            np.unique(
                y_true
            )
        )


        recalls = []


        for cls in present_classes:

            cls_mask = (
                y_true
                ==
                cls
            )


            recall = (
                (
                    y_pred[
                        cls_mask
                    ]
                    ==
                    cls
                )
                .mean()
            )


            recalls.append(
                recall
            )


        return float(
            np.mean(
                recalls
            )
        )


    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )


    before_acc = []
    after_acc = []

    before_ba = []
    after_ba = []

    fold_rows = []


    # --------------------------------------------------------
    # Same folds before vs after
    # --------------------------------------------------------

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        cv.split(
            Xb,
            y,
            groups=groups,
        ),
        start=1,
    ):

        y_train = (
            y[
                train_idx
            ]
        )

        y_test = (
            y[
                test_idx
            ]
        )


        # ----------------------------------------------------
        # BEFORE
        # ----------------------------------------------------

        model_before = Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),

                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )


        model_before.fit(
            Xb[
                train_idx
            ],
            y_train,
        )


        pred_before = (
            model_before.predict(
                Xb[
                    test_idx
                ]
            )
        )


        # ----------------------------------------------------
        # AFTER
        # ----------------------------------------------------

        model_after = Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),

                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )


        model_after.fit(
            Xa[
                train_idx
            ],
            y_train,
        )


        pred_after = (
            model_after.predict(
                Xa[
                    test_idx
                ]
            )
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        acc_before = (
            accuracy_score(
                y_test,
                pred_before,
            )
        )


        acc_after = (
            accuracy_score(
                y_test,
                pred_after,
            )
        )


        ba_before = (
            safe_balanced_accuracy(
                y_test,
                pred_before,
            )
        )


        ba_after = (
            safe_balanced_accuracy(
                y_test,
                pred_after,
            )
        )


        before_acc.append(
            acc_before
        )

        after_acc.append(
            acc_after
        )

        before_ba.append(
            ba_before
        )

        after_ba.append(
            ba_after
        )


        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        train_classes = set(
            np.unique(
                y_train
            )
        )

        test_classes = set(
            np.unique(
                y_test
            )
        )

        missing_in_train = (
            test_classes
            -
            train_classes
        )


        fold_rows.append(
            {
                "Fold":
                    fold,

                "Train N":
                    len(
                        train_idx
                    ),

                "Test N":
                    len(
                        test_idx
                    ),

                "Train Sites":
                    len(
                        train_classes
                    ),

                "Test Sites":
                    len(
                        test_classes
                    ),

                "Test Sites Missing In Train":
                    len(
                        missing_in_train
                    ),

                "Before Accuracy":
                    acc_before,

                "After Accuracy":
                    acc_after,

                "Before Balanced Accuracy":
                    ba_before,

                "After Balanced Accuracy":
                    ba_after,
            }
        )


        print(
            f"Fold {fold}: "
            f"BA {ba_before:.3f} -> {ba_after:.3f}"
        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Balanced Accuracy",
            ],

            "Before Mean": [
                np.mean(
                    before_acc
                ),

                np.mean(
                    before_ba
                ),
            ],

            "Before SD": [
                np.std(
                    before_acc
                ),

                np.std(
                    before_ba
                ),
            ],

            "After Mean": [
                np.mean(
                    after_acc
                ),

                np.mean(
                    after_ba
                ),
            ],

            "After SD": [
                np.std(
                    after_acc
                ),

                np.std(
                    after_ba
                ),
            ],
        }
    )


    fold_df = pd.DataFrame(
        fold_rows
    )


    print(
        "\nSummary:"
    )

    display(
        summary
    )


    print(
        "\nFold diagnostics:"
    )

    display(
        fold_df
    )


    return {
        "summary":
            summary,

        "fold_results":
            fold_df,

        "before_balanced_accuracy":
            before_ba,

        "after_balanced_accuracy":
            after_ba,
    }

# %%
# ============================================================
# RUN WITHIN-COHORT SITE CLASSIFIERS
# ============================================================

within_site_results = {}


for cohort in [
    "ADNI",
    "HABS",
    "ADRC",
    "ADDECODE",
]:

    result = evaluate_within_cohort_site(
        X_before=
            X_before_eval,

        X_after=
            X_after_eval,

        meta=
            eval_meta,

        cohort_name=
            cohort,

        min_site_size=5,

        n_splits=5,
    )


    if result is not None:

        within_site_results[
            cohort
        ] = result

# %% [markdown]
# # Sex classifier

# %%
# ============================================================
# CELL 18 — SEX CLASSIFIER
#
# Biological preservation control
# ============================================================

sex_result = (
    evaluate_classifier_before_after(

        X_before=
            X_before_eval,

        X_after=
            X_after_eval,

        y=
            eval_meta[
                "SEX"
            ].values,

        groups=
            eval_meta[
                "PARTICIPANT_GROUP"
            ].values,

        label_name=
            "Sex Classification",

        n_splits=5,
    )
)

# %% [markdown]
# # Diagnosis classifier

# %%
# ============================================================
# CELL 19 — HARMONIZED DIAGNOSIS EVALUATION
#
# Create a common diagnosis definition:
#
# CN
# MCI
# AD
#
# IMPORTANT:
# GROUP is NOT modified.
# This column is only used for evaluation.
# ============================================================


# ============================================================
# INITIALIZE DIAGNOSIS_EVAL AS STRING
# ============================================================

eval_meta["DIAGNOSIS_EVAL"] = pd.Series(
    pd.NA,
    index=eval_meta.index,
    dtype="string"
)


# ============================================================
# ADNI
# ============================================================

adni_mask = (
    eval_meta["DATASET"]
    == "ADNI"
)

adni_group = (
    eval_meta.loc[
        adni_mask,
        "GROUP"
    ]
    .astype("string")
    .str.strip()
    .str.upper()
)

adni_dx_map = {
    "CN": "CN",
    "EMCI": "MCI",
    "LMCI": "MCI",
    "MCI": "MCI",
    "AD": "AD",
}

eval_meta.loc[
    adni_mask,
    "DIAGNOSIS_EVAL"
] = (
    adni_group
    .map(adni_dx_map)
    .to_numpy()
)


# ============================================================
# ADRC
# ============================================================

adrc_mask = (
    eval_meta["DATASET"]
    == "ADRC"
)

adrc_group = (
    eval_meta.loc[
        adrc_mask,
        "GROUP"
    ]
    .astype("string")
    .str.strip()
    .str.upper()
)

adrc_dx_map = {
    "CN": "CN",
    "MCI": "MCI",
    "AD": "AD",
}

eval_meta.loc[
    adrc_mask,
    "DIAGNOSIS_EVAL"
] = (
    adrc_group
    .map(adrc_dx_map)
    .to_numpy()
)


# ============================================================
# ADDECODE
#
# GROUP contains APOE genotype, NOT diagnosis.
# Therefore exclude from diagnosis evaluation.
# ============================================================

addecode_mask = (
    eval_meta["DATASET"]
    == "ADDECODE"
)

eval_meta.loc[
    addecode_mask,
    "DIAGNOSIS_EVAL"
] = pd.NA


# ============================================================
# CHECK RESULT
# ============================================================

print("=" * 80)
print("HARMONIZED DIAGNOSIS LABELS")
print("=" * 80)

print("\nOverall:")

print(
    eval_meta["DIAGNOSIS_EVAL"]
    .value_counts(dropna=False)
)

print("\nDataset × Diagnosis:")

display(
    pd.crosstab(
        eval_meta["DATASET"],
        eval_meta["DIAGNOSIS_EVAL"],
        margins=True,
        dropna=False,
    )
)

# %% [markdown]
# # CN vs MCI biological preservation

# %%
# ============================================================
# CELL 20 — CN vs MCI CLASSIFIER
#
# Biological preservation evaluation
# ============================================================


dx_mask = (
    eval_meta[
        "DIAGNOSIS_EVAL"
    ]
    .isin(
        [
            "CN",
            "MCI",
        ]
    )
)


dx_meta = (
    eval_meta.loc[
        dx_mask
    ]
    .reset_index(
        drop=True
    )
)


X_before_dx = (
    X_before_eval[
        dx_mask.values
    ]
)


X_after_dx = (
    X_after_eval[
        dx_mask.values
    ]
)


print("=" * 80)
print("CN vs MCI CLASSIFICATION")
print("=" * 80)


print(
    "N =",
    len(
        dx_meta
    )
)


print(
    "\nDiagnosis counts:"
)

print(
    dx_meta[
        "DIAGNOSIS_EVAL"
    ]
    .value_counts()
)


print(
    "\nDataset × Diagnosis:"
)


display(
    pd.crosstab(
        dx_meta[
            "DATASET"
        ],

        dx_meta[
            "DIAGNOSIS_EVAL"
        ],
    )
)


# ============================================================
# Run classifier
# ============================================================

diagnosis_result = (
    evaluate_classifier_before_after(

        X_before=
            X_before_dx,

        X_after=
            X_after_dx,

        y=
            dx_meta[
                "DIAGNOSIS_EVAL"
            ].values,

        groups=
            dx_meta[
                "PARTICIPANT_GROUP"
            ].values,

        label_name=
            "CN vs MCI Classification",

        n_splits=5,
    )
)

# %% [markdown]
# # Summary figure

# %%
# ============================================================
# FINAL SUMMARY FIGURE
#
# Technical effect removal:
#   1. Dataset classification = BETWEEN-cohort
#   2. Site classification    = WITHIN-cohort
#
# Biological preservation:
#   3. Sex
#   4. CN vs MCI
#
# Metric:
#   Balanced Accuracy
# ============================================================


# ============================================================
# Helper — extract Balanced Accuracy
# ============================================================

def extract_ba(result):

    summary = result["summary"]

    row = (
        summary.loc[
            summary["Metric"]
            ==
            "Balanced Accuracy"
        ]
        .iloc[0]
    )

    return {
        "Before":
            float(
                row["Before Mean"]
            ),

        "After":
            float(
                row["After Mean"]
            ),

        "Before SD":
            float(
                row["Before SD"]
            ),

        "After SD":
            float(
                row["After SD"]
            ),
    }


# ============================================================
# BUILD SUMMARY
# ============================================================

summary_rows = []


# ============================================================
# PART 1 — TECHNICAL EFFECT REMOVAL
# ============================================================


# ------------------------------------------------------------
# 1. Dataset
# BETWEEN-cohort effect
# ------------------------------------------------------------

values = extract_ba(
    dataset_result
)


summary_rows.append(
    {
        "Classifier":
            "Dataset",

        "Effect":
            "Between-cohort",

        "Type":
            "Technical",

        **values,
    }
)


# ------------------------------------------------------------
# 2. Site
# WITHIN-cohort effects
#
# Automatically include every cohort that successfully
# produced a within-site classifier.
# ------------------------------------------------------------

preferred_order = [
    "ADNI",
    "HABS",
    "ADRC",
    "ADDECODE",
]


for cohort in preferred_order:

    if cohort not in within_site_results:
        continue


    values = extract_ba(
        within_site_results[
            cohort
        ]
    )


    summary_rows.append(
        {
            "Classifier":
                f"{cohort} Site",

            "Effect":
                "Within-cohort",

            "Type":
                "Technical",

            **values,
        }
    )


# ============================================================
# PART 2 — BIOLOGICAL SIGNAL PRESERVATION
# ============================================================


# ------------------------------------------------------------
# 3. Sex
# ------------------------------------------------------------

values = extract_ba(
    sex_result
)


summary_rows.append(
    {
        "Classifier":
            "Sex",

        "Effect":
            "Biological",

        "Type":
            "Biological",

        **values,
    }
)


# ------------------------------------------------------------
# 4. CN vs MCI
#
# This should be the NEW diagnosis_result,
# NOT the old heterogeneous GROUP classifier.
# ------------------------------------------------------------

if (
    "diagnosis_result"
    in globals()
):

    values = extract_ba(
        diagnosis_result
    )


    summary_rows.append(
        {
            "Classifier":
                "CN vs MCI",

            "Effect":
                "Biological",

            "Type":
                "Biological",

            **values,
        }
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


print("=" * 90)
print("DWI HARMONIZATION EVALUATION SUMMARY")
print("=" * 90)


display(
    summary_df
)


# ============================================================
# PRINT INTERPRETATION TABLE
# ============================================================

interpretation_df = (
    summary_df[
        [
            "Classifier",
            "Effect",
            "Before",
            "After",
        ]
    ]
    .copy()
)


interpretation_df[
    "Change"
] = (
    interpretation_df[
        "After"
    ]
    -
    interpretation_df[
        "Before"
    ]
)


interpretation_df[
    "Expected"
] = (
    interpretation_df[
        "Effect"
    ]
    .map(
        {
            "Between-cohort":
                "Decrease",

            "Within-cohort":
                "Decrease",

            "Biological":
                "Preserve",
        }
    )
)


print(
    "\nInterpretation:"
)

display(
    interpretation_df
)


# ============================================================
# PLOT
# ============================================================

x = np.arange(
    len(
        summary_df
    )
)

width = 0.34


fig, ax = plt.subplots(
    figsize=(
        max(
            11,
            len(
                summary_df
            )
            * 1.6
        ),
        6.5,
    )
)


# ============================================================
# BEFORE
# ============================================================

bars_before = ax.bar(

    x - width / 2,

    summary_df[
        "Before"
    ],

    width,

    yerr=
        summary_df[
            "Before SD"
        ],

    capsize=4,

    label=
        "Before Harmonization",
)


# ============================================================
# AFTER
# ============================================================

bars_after = ax.bar(

    x + width / 2,

    summary_df[
        "After"
    ],

    width,

    yerr=
        summary_df[
            "After SD"
        ],

    capsize=4,

    label=
        "After Harmonization",
)


# ============================================================
# ADD NUMBERS ABOVE BARS
# ============================================================

for bar, value, sd in zip(

    bars_before,

    summary_df[
        "Before"
    ],

    summary_df[
        "Before SD"
    ],
):

    ax.text(

        bar.get_x()
        +
        bar.get_width() / 2,

        value
        +
        sd
        +
        0.025,

        f"{value:.2f}",

        ha="center",
        va="bottom",

        fontsize=9,
    )


for bar, value, sd in zip(

    bars_after,

    summary_df[
        "After"
    ],

    summary_df[
        "After SD"
    ],
):

    ax.text(

        bar.get_x()
        +
        bar.get_width() / 2,

        value
        +
        sd
        +
        0.025,

        f"{value:.2f}",

        ha="center",
        va="bottom",

        fontsize=9,
    )


# ============================================================
# TECHNICAL / BIOLOGICAL SEPARATOR
# ============================================================

technical_count = int(
    (
        summary_df[
            "Type"
        ]
        ==
        "Technical"
    )
    .sum()
)


if (
    technical_count > 0
    and
    technical_count
    <
    len(
        summary_df
    )
):

    separator_x = (
        technical_count
        - 0.5
    )


    ax.axvline(

        separator_x,

        linestyle="--",

        linewidth=1.2,

        alpha=0.6,
    )


# ============================================================
# SECTION LABELS
# ============================================================

if technical_count > 0:

    technical_center = (
        technical_count
        - 1
    ) / 2


    ax.text(

        technical_center,

        1.105,

        "Technical Effect Removal\n(expected: decrease)",

        ha="center",
        va="center",

        fontsize=10,

        fontweight="bold",
    )


biological_count = (
    len(
        summary_df
    )
    -
    technical_count
)


if biological_count > 0:

    biological_center = (
        technical_count
        +
        (
            biological_count
            - 1
        )
        / 2
    )


    ax.text(

        biological_center,

        1.105,

        "Biological Signal Preservation\n(expected: similar)",

        ha="center",
        va="center",

        fontsize=10,

        fontweight="bold",
    )


# ============================================================
# AXIS
# ============================================================

ax.set_xticks(
    x
)


ax.set_xticklabels(
    summary_df[
        "Classifier"
    ],

    rotation=25,

    ha="right",
)


ax.set_ylabel(
    "Balanced Accuracy"
)


ax.set_xlabel(
    "Classification Target"
)


ax.set_ylim(
    0,
    1.18,
)


ax.set_title(
    "DWI Harmonization Evaluation\n"
    "Between-Cohort, Within-Cohort, and Biological Signal"
)


ax.legend(
    loc="upper right"
)


# ============================================================
# OPTIONAL DESCRIPTION BELOW EACH CLASSIFIER
# ============================================================

for i, effect in enumerate(
    summary_df[
        "Effect"
    ]
):

    if effect == "Between-cohort":

        label = "between"

    elif effect == "Within-cohort":

        label = "within"

    else:

        label = "preserve"


    ax.text(

        i,

        -0.13,

        label,

        transform=
            ax.get_xaxis_transform(),

        ha="center",

        fontsize=8,
    )


# ============================================================
# FINALIZE
# ============================================================

plt.tight_layout()


plt.savefig(

    OUT_FIGURES
    / "DWI_harmonization_final_summary.png",

    dpi=300,

    bbox_inches="tight",
)


plt.show()


