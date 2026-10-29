# %%
# ============================================================
# CELL 1 — IMPORTS + CONFIGURATION + LOAD HARMONIZED fMRI
# ============================================================

import json
import re
import warnings

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pearsonr

from sklearn.base import clone

from sklearn.impute import (
    SimpleImputer,
)

from sklearn.linear_model import (
    Ridge,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.preprocessing import (
    StandardScaler,
)


try:

    from sklearn.model_selection import (
        StratifiedGroupKFold,
    )

except ImportError:

    StratifiedGroupKFold = None


warnings.filterwarnings(
    "ignore"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

OUTER_SPLITS = 5

INNER_SPLITS = 4

AGE_BIN_NUMBER = 5


AGE_COL = "Age"

DIAGNOSIS_COL = "Diagnosis"

DATASET_COL = "DATASET"

ID_COL = "connectome_key"


RIDGE_ALPHA_GRID = [

    0.01,

    0.1,

    1.0,

    10.0,

    100.0,

    1000.0,

    10000.0,
]


CONTROL_LABELS = {

    "CN",

    "CONTROL",

    "HEALTHY",

    "HEALTHY CONTROL",

    "COGNITIVELY NORMAL",

    "COGNITIVELY UNIMPAIRED",

    "CU",

    "HC",
}


# ============================================================
# PATHS
# ============================================================

HARMONIZATION_RESULT_DIR = Path(

    "/mnt/newStor/paros/paros_WORK/avery/results/"
    "ADNI_HABS_ADRC_fMRI_joint_site_harmonization"
)


EVALUATION_INPUT_DIR = (

    HARMONIZATION_RESULT_DIR

    / "evaluation_inputs"
)


X_COMBAT_PATH = (

    EVALUATION_INPUT_DIR

    / "fMRI_X_combat_edges.csv"
)


METADATA_PATH = (

    EVALUATION_INPUT_DIR

    / "fMRI_metadata_aligned.csv"
)


BRAIN_AGE_OUTPUT_DIR = Path(

    "/mnt/newStor/paros/paros_WORK/avery/results/"
    "fMRI_brain_age"
)


BRAIN_AGE_OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True,
)


# ============================================================
# LOAD
# ============================================================

X_combat = pd.read_csv(

    X_COMBAT_PATH,

    index_col=0,
)


metadata_aligned = pd.read_csv(

    METADATA_PATH,

    index_col=0,
)


# ============================================================
# ALIGNMENT CHECK
# ============================================================

if (

    len(
        X_combat
    )

    !=

    len(
        metadata_aligned
    )

):

    raise ValueError(

        "X_combat and metadata_aligned "
        "have different row counts."
    )


if (

    list(
        X_combat.index
    )

    !=

    list(
        metadata_aligned.index
    )

):

    raise ValueError(

        "X_combat and metadata_aligned "
        "row IDs are not aligned."
    )


print("=" * 72)

print(
    "fMRI BRAIN-AGE INPUT"
)

print("=" * 72)


print(

    "X_combat:",

    X_combat.shape
)


print(

    "metadata_aligned:",

    metadata_aligned.shape
)


print(

    "Output directory:",

    BRAIN_AGE_OUTPUT_DIR
)


# %%
# ============================================================
# CELL 2 — HELPER FUNCTIONS
# ============================================================


def normalize_diagnosis(
    value,
):

    """
    Standardize diagnosis labels.
    """

    value = str(
        value
    ).strip().upper()


    value = re.sub(

        r"[_-]+",

        " ",

        value,
    )


    value = re.sub(

        r"\s+",

        " ",

        value,
    )


    return value


# ============================================================
# PARTICIPANT ID
# ============================================================

def extract_participant_id(
    connectome_key,
):

    """
    Convert visit-level IDs to participant-level IDs.

    Examples
    --------
    R4288_y0 -> R4288
    R4288_y4 -> R4288
    H4369_y2 -> H4369
    ADRC0040 -> ADRC0040
    """

    value = str(
        connectome_key
    ).strip()


    value = re.sub(

        r"(?i)(_y\d+|_m\d+|_bl|_baseline)$",

        "",

        value,
    )


    return value.upper()


# ============================================================
# AGE BINS
# ============================================================

def make_age_bins(
    age,
    requested_bins=5,
):

    """
    Create approximately balanced chronological-age bins.
    """

    age = np.asarray(

        age,

        dtype=float,
    )


    unique_age_number = len(

        np.unique(

            age[
                np.isfinite(
                    age
                )
            ]
        )
    )


    number_of_bins = max(

        2,

        min(

            requested_bins,

            unique_age_number,
        ),
    )


    try:

        return np.asarray(

            pd.qcut(

                age,

                q=
                    number_of_bins,

                labels=False,

                duplicates=
                    "drop",
            ),

            dtype=int,
        )


    except Exception:

        return np.zeros(

            len(
                age
            ),

            dtype=int,
        )


# ============================================================
# MODEL
# ============================================================

def build_brain_age_pipeline():

    """
    Edge-based brain-age model.

    Imputation and standardization are fitted
    inside each training fold.
    """

    return Pipeline(
        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "scaler",

                StandardScaler(),
            ),

            (
                "ridge",

                Ridge(),
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(

        y_true,

        dtype=float,
    )


    y_pred = np.asarray(

        y_pred,

        dtype=float,
    )


    valid = (

        np.isfinite(
            y_true
        )

        &

        np.isfinite(
            y_pred
        )
    )


    y_true = (
        y_true[
            valid
        ]
    )


    y_pred = (
        y_pred[
            valid
        ]
    )


    if (

        len(
            y_true
        )
        >=
        3

        and

        np.std(
            y_true
        )
        >
        0

        and

        np.std(
            y_pred
        )
        >
        0

    ):

        correlation, correlation_p = (
            pearsonr(

                y_true,

                y_pred,
            )
        )


    else:

        correlation = np.nan

        correlation_p = np.nan


    return {

        "N":
            int(
                len(
                    y_true
                )
            ),

        "MAE":
            mean_absolute_error(
                y_true,
                y_pred,
            ),

        "RMSE":
            np.sqrt(

                mean_squared_error(
                    y_true,
                    y_pred,
                )
            ),

        "R2":
            r2_score(
                y_true,
                y_pred,
            ),

        "Pearson_r":
            correlation,

        "Pearson_p":
            correlation_p,
    }


# ============================================================
# BAG AGE-BIAS CORRECTION
# ============================================================

def fit_bag_bias(
    age,
    predicted_age,
):

    """
    BAG = beta * age + alpha
    """

    age = np.asarray(

        age,

        dtype=float,
    )


    predicted_age = np.asarray(

        predicted_age,

        dtype=float,
    )


    raw_bag = (

        predicted_age

        -

        age
    )


    valid = (

        np.isfinite(
            age
        )

        &

        np.isfinite(
            raw_bag
        )
    )


    beta, alpha = np.polyfit(

        age[
            valid
        ],

        raw_bag[
            valid
        ],

        deg=1,
    )


    return (

        float(
            beta
        ),

        float(
            alpha
        ),
    )


def apply_bag_correction(

    age,

    predicted_age,

    beta,

    alpha,
):

    """
    BAG_raw =
        predicted age - chronological age

    cBAG =
        BAG_raw - (beta * chronological age + alpha)
    """

    age = np.asarray(

        age,

        dtype=float,
    )


    predicted_age = np.asarray(

        predicted_age,

        dtype=float,
    )


    raw_bag = (

        predicted_age

        -

        age
    )


    expected_bag = (

        beta

        *

        age

        +

        alpha
    )


    corrected_bag = (

        raw_bag

        -

        expected_bag
    )


    corrected_predicted_age = (

        age

        +

        corrected_bag
    )


    return (

        raw_bag,

        corrected_bag,

        corrected_predicted_age,
    )


# ============================================================
# GROUPED CV
# ============================================================

def get_grouped_cv(
    groups,
    requested_splits,
):

    unique_group_number = len(

        np.unique(
            groups
        )
    )


    effective_splits = min(

        requested_splits,

        unique_group_number,
    )


    if effective_splits < 2:

        raise ValueError(

            "At least two unique participants "
            "are required."
        )


    return GroupKFold(

        n_splits=
            effective_splits
    )


# ============================================================
# TRAINING-ONLY OOF PREDICTIONS
# ============================================================

def get_training_oof_predictions(

    estimator,

    X_train,

    y_train,

    groups_train,

    requested_splits,
):

    """
    Training-only OOF predictions.

    Used to estimate BAG bias without
    using outer validation subjects.
    """

    inner_cv = get_grouped_cv(

        groups=
            groups_train,

        requested_splits=
            requested_splits,
    )


    predictions = np.full(

        len(
            y_train
        ),

        np.nan,

        dtype=float,
    )


    for (

        train_index,

        validation_index,

    ) in inner_cv.split(

        X_train,

        y_train,

        groups_train,
    ):

        model = clone(
            estimator
        )


        model.fit(

            X_train[
                train_index
            ],

            y_train[
                train_index
            ],
        )


        predictions[
            validation_index
        ] = model.predict(

            X_train[
                validation_index
            ]
        )


    return predictions


print(
    "Helper functions defined."
)

# %%
# ============================================================
# CELL 3 — PREPARE HARMONIZED fMRI INPUTS
# ============================================================


# ============================================================
# Harmonized fMRI connectome edges
# ============================================================

X_brain_age_all = (

    X_combat

    .to_numpy(
        dtype=float
    )
)


# ============================================================
# Row-aligned metadata
# ============================================================

brain_age_metadata_all = (

    metadata_aligned

    .reset_index(
        drop=True
    )

    .copy()
)


print("=" * 72)

print(
    "BRAIN-AGE INPUT CHECK"
)

print("=" * 72)


print(

    "X_combat:",

    X_brain_age_all.shape
)


print(

    "metadata_aligned:",

    brain_age_metadata_all.shape
)


# ============================================================
# Required columns
# ============================================================

required_columns = [

    AGE_COL,

    DIAGNOSIS_COL,

    DATASET_COL,

    ID_COL,
]


missing_columns = [

    column

    for column in required_columns

    if column
    not in
    brain_age_metadata_all.columns
]


if missing_columns:

    raise KeyError(

        f"Missing columns: {missing_columns}\n"

        f"Available columns: "

        f"{brain_age_metadata_all.columns.tolist()}"
    )


# ============================================================
# Alignment
# ============================================================

if (

    len(
        brain_age_metadata_all
    )

    !=

    X_brain_age_all.shape[0]

):

    raise ValueError(

        "X_combat and metadata_aligned "
        "are not aligned.\n"

        f"X rows: "
        f"{X_brain_age_all.shape[0]}\n"

        f"Metadata rows: "
        f"{len(brain_age_metadata_all)}"
    )


# ============================================================
# Original row position
# ============================================================

brain_age_metadata_all[
    "brain_age_original_row"
] = np.arange(

    len(
        brain_age_metadata_all
    )
)


# ============================================================
# Clean Age
# ============================================================

brain_age_metadata_all[
    AGE_COL
] = pd.to_numeric(

    brain_age_metadata_all[
        AGE_COL
    ],

    errors="coerce",
)


# ============================================================
# Clean Diagnosis
# ============================================================

brain_age_metadata_all[
    "Diagnosis_brain_age"
] = (

    brain_age_metadata_all[
        DIAGNOSIS_COL
    ]

    .map(
        normalize_diagnosis
    )
)


# ============================================================
# Clean Dataset
# ============================================================

brain_age_metadata_all[
    DATASET_COL
] = (

    brain_age_metadata_all[
        DATASET_COL
    ]

    .astype(str)

    .str.strip()

    .str.upper()
)


# ============================================================
# Participant groups
# ============================================================

brain_age_metadata_all[
    "participant_group"
] = (

    brain_age_metadata_all[
        DATASET_COL
    ]
    .astype(str)

    +

    "__"

    +

    brain_age_metadata_all[
        ID_COL
    ]
    .map(
        extract_participant_id
    )
)


# ============================================================
# Replace Inf with NaN
# ============================================================

X_brain_age_all[

    ~np.isfinite(
        X_brain_age_all
    )

] = np.nan


# ============================================================
# Valid subjects for prediction
# ============================================================

valid_prediction_mask = (

    brain_age_metadata_all[
        AGE_COL
    ]
    .notna()
    .to_numpy()

    &

    ~np.all(

        np.isnan(
            X_brain_age_all
        ),

        axis=1,
    )
)


# ============================================================
# CN subjects for normative training
# ============================================================

control_mask = (

    brain_age_metadata_all[
        "Diagnosis_brain_age"
    ]

    .isin(
        CONTROL_LABELS
    )

    .to_numpy()
)


training_mask = (

    valid_prediction_mask

    &

    control_mask
)


# ============================================================
# CN training data
# ============================================================

X_brain_age_train = (

    X_brain_age_all[
        training_mask
    ]
)


brain_age_metadata_train = (

    brain_age_metadata_all

    .loc[
        training_mask
    ]

    .reset_index(
        drop=True
    )

    .copy()
)


y_brain_age_train = (

    brain_age_metadata_train[
        AGE_COL
    ]

    .to_numpy(
        dtype=float
    )
)


brain_age_groups = (

    brain_age_metadata_train[
        "participant_group"
    ]

    .to_numpy(
        dtype=str
    )
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\nTraining population"
)

print(
    "-" * 72
)


print(

    "CN training scans:",

    len(
        brain_age_metadata_train
    )
)


print(

    "CN training participants:",

    len(
        np.unique(
            brain_age_groups
        )
    )
)


print(

    "Connectome edges:",

    X_brain_age_train.shape[1]
)


print(
    "\nCN scans by dataset:"
)


print(

    brain_age_metadata_train[
        DATASET_COL
    ]

    .value_counts()
)


print(
    "\nCN age summary:"
)


print(

    brain_age_metadata_train[
        AGE_COL
    ]

    .describe()
)


if (

    len(
        brain_age_metadata_train
    )

    <
    30

):

    raise ValueError(

        "Too few CN observations remain. "
        "Inspect the diagnosis labels."
    )

# %%
# ============================================================
# CELL 4 — NESTED GROUPED CROSS-VALIDATION
# ============================================================


age_bins = make_age_bins(

    y_brain_age_train,

    requested_bins=
        AGE_BIN_NUMBER,
)


# ============================================================
# OUTER CV
# ============================================================

if StratifiedGroupKFold is not None:

    try:

        outer_cv = StratifiedGroupKFold(

            n_splits=
                OUTER_SPLITS,

            shuffle=True,

            random_state=
                RANDOM_STATE,
        )


        outer_splits = list(

            outer_cv.split(

                X_brain_age_train,

                age_bins,

                groups=
                    brain_age_groups,
            )
        )


        outer_cv_name = (

            "StratifiedGroupKFold "
            "by participant and age bin"
        )


    except ValueError:

        outer_cv = GroupKFold(

            n_splits=
                OUTER_SPLITS
        )


        outer_splits = list(

            outer_cv.split(

                X_brain_age_train,

                y_brain_age_train,

                groups=
                    brain_age_groups,
            )
        )


        outer_cv_name = (

            "GroupKFold by participant"
        )


else:

    outer_cv = GroupKFold(

        n_splits=
            OUTER_SPLITS
    )


    outer_splits = list(

        outer_cv.split(

            X_brain_age_train,

            y_brain_age_train,

            groups=
                brain_age_groups,
        )
    )


    outer_cv_name = (

        "GroupKFold by participant"
    )


print(

    "Outer CV:",

    outer_cv_name
)


# ============================================================
# STORAGE
# ============================================================

oof_predicted_age_raw = np.full(

    len(
        y_brain_age_train
    ),

    np.nan,
)


oof_predicted_age_corrected = np.full(

    len(
        y_brain_age_train
    ),

    np.nan,
)


oof_raw_bag = np.full(

    len(
        y_brain_age_train
    ),

    np.nan,
)


oof_corrected_bag = np.full(

    len(
        y_brain_age_train
    ),

    np.nan,
)


oof_fold = np.full(

    len(
        y_brain_age_train
    ),

    -1,

    dtype=int,
)


fold_summary_rows = []


# ============================================================
# OUTER FOLDS
# ============================================================

for fold_number, (

    train_index,

    validation_index,

) in enumerate(

    outer_splits,

    start=1,
):

    print(
        "\n"
        +
        "=" * 72
    )

    print(

        f"OUTER FOLD {fold_number}"
    )

    print(
        "=" * 72
    )


    X_train_fold = (

        X_brain_age_train[
            train_index
        ]
    )


    X_validation_fold = (

        X_brain_age_train[
            validation_index
        ]
    )


    y_train_fold = (

        y_brain_age_train[
            train_index
        ]
    )


    y_validation_fold = (

        y_brain_age_train[
            validation_index
        ]
    )


    groups_train_fold = (

        brain_age_groups[
            train_index
        ]
    )


    groups_validation_fold = (

        brain_age_groups[
            validation_index
        ]
    )


    # ========================================================
    # PARTICIPANT LEAKAGE CHECK
    # ========================================================

    participant_overlap = (

        set(
            groups_train_fold
        )

        &

        set(
            groups_validation_fold
        )
    )


    if participant_overlap:

        raise RuntimeError(

            f"Participant leakage "
            f"in fold {fold_number}: "

            f"{list(participant_overlap)[:10]}"
        )


    # ========================================================
    # INNER CV — RIDGE ALPHA
    # ========================================================

    inner_cv = get_grouped_cv(

        groups=
            groups_train_fold,

        requested_splits=
            INNER_SPLITS,
    )


    parameter_search = GridSearchCV(

        estimator=
            build_brain_age_pipeline(),

        param_grid=
            {

                "ridge__alpha":
                    RIDGE_ALPHA_GRID,
            },

        scoring=
            "neg_mean_absolute_error",

        cv=
            inner_cv,

        n_jobs=-1,

        refit=True,
    )


    parameter_search.fit(

        X_train_fold,

        y_train_fold,

        groups=
            groups_train_fold,
    )


    best_model = (

        parameter_search
        .best_estimator_
    )


    best_alpha = (

        parameter_search
        .best_params_[
            "ridge__alpha"
        ]
    )


    inner_cv_mae = (

        -parameter_search
        .best_score_
    )


    # ========================================================
    # TRAINING-ONLY OOF FOR BAG BIAS
    # ========================================================

    training_oof_prediction = (

        get_training_oof_predictions(

            estimator=
                best_model,

            X_train=
                X_train_fold,

            y_train=
                y_train_fold,

            groups_train=
                groups_train_fold,

            requested_splits=
                INNER_SPLITS,
        )
    )


    bias_beta, bias_alpha = (

        fit_bag_bias(

            age=
                y_train_fold,

            predicted_age=
                training_oof_prediction,
        )
    )


    # ========================================================
    # HELD-OUT PREDICTION
    # ========================================================

    validation_prediction_raw = (

        best_model.predict(

            X_validation_fold
        )
    )


    (

        validation_raw_bag,

        validation_corrected_bag,

        validation_prediction_corrected,

    ) = apply_bag_correction(

        age=
            y_validation_fold,

        predicted_age=
            validation_prediction_raw,

        beta=
            bias_beta,

        alpha=
            bias_alpha,
    )


    # ========================================================
    # STORE
    # ========================================================

    oof_predicted_age_raw[
        validation_index
    ] = validation_prediction_raw


    oof_predicted_age_corrected[
        validation_index
    ] = validation_prediction_corrected


    oof_raw_bag[
        validation_index
    ] = validation_raw_bag


    oof_corrected_bag[
        validation_index
    ] = validation_corrected_bag


    oof_fold[
        validation_index
    ] = fold_number


    fold_metrics = calculate_metrics(

        y_true=
            y_validation_fold,

        y_pred=
            validation_prediction_raw,
    )


    fold_summary_rows.append(
        {

            "Fold":
                fold_number,

            "Train scans":
                len(
                    train_index
                ),

            "Validation scans":
                len(
                    validation_index
                ),

            "Train participants":
                len(
                    np.unique(
                        groups_train_fold
                    )
                ),

            "Validation participants":
                len(
                    np.unique(
                        groups_validation_fold
                    )
                ),

            "Participant overlap":
                len(
                    participant_overlap
                ),

            "Best Ridge alpha":
                best_alpha,

            "Inner CV MAE":
                inner_cv_mae,

            "Bias beta":
                bias_beta,

            "Bias alpha":
                bias_alpha,

            **fold_metrics,
        }
    )


    print(

        f"Best alpha: "
        f"{best_alpha}"
    )


    print(

        f"MAE: "
        f"{fold_metrics['MAE']:.3f}"
    )


    print(

        f"RMSE: "
        f"{fold_metrics['RMSE']:.3f}"
    )


    print(

        f"R²: "
        f"{fold_metrics['R2']:.3f}"
    )


if not np.isfinite(

    oof_predicted_age_raw

).all():

    raise RuntimeError(

        "Some CN observations did not "
        "receive OOF predictions."
    )


print(
    "\nNested grouped cross-validation completed."
)
# %%
# ============================================================
# CELL 5 — OOF RESULTS
# ============================================================


oof_metrics_raw = calculate_metrics(

    y_true=
        y_brain_age_train,

    y_pred=
        oof_predicted_age_raw,
)


oof_metrics_corrected = calculate_metrics(

    y_true=
        y_brain_age_train,

    y_pred=
        oof_predicted_age_corrected,
)


brain_age_oof_results = (

    brain_age_metadata_train[
        [

            DATASET_COL,

            ID_COL,

            "participant_group",

            AGE_COL,

            DIAGNOSIS_COL,

            "brain_age_original_row",
        ]
    ]

    .copy()
)


brain_age_oof_results[
    "Fold"
] = oof_fold


brain_age_oof_results[
    "Brain_age_raw"
] = oof_predicted_age_raw


brain_age_oof_results[
    "BAG_raw"
] = oof_raw_bag


brain_age_oof_results[
    "Brain_age_bias_corrected"
] = oof_predicted_age_corrected


brain_age_oof_results[
    "cBAG_foldwise"
] = oof_corrected_bag


brain_age_fold_summary = pd.DataFrame(

    fold_summary_rows
)


brain_age_metric_summary = pd.DataFrame(
    [

        {

            "Prediction":
                "Raw OOF",

            **oof_metrics_raw,
        },

        {

            "Prediction":
                "Fold-wise bias corrected",

            **oof_metrics_corrected,
        },
    ]
)


print("=" * 72)

print(
    "FINAL CN OOF BRAIN-AGE PERFORMANCE"
)

print("=" * 72)


print(
    "\nUse the Raw OOF row "
    "for model performance."
)


display(

    brain_age_metric_summary

    .round(
        4
    )
)


display(

    brain_age_fold_summary

    .round(
        4
    )
)
# %%
# ============================================================
# CELL 6 — FINAL CN MODEL + ALL SUBJECT PREDICTIONS
# ============================================================


final_inner_cv = get_grouped_cv(

    groups=
        brain_age_groups,

    requested_splits=
        INNER_SPLITS,
)


final_parameter_search = GridSearchCV(

    estimator=
        build_brain_age_pipeline(),

    param_grid=
        {

            "ridge__alpha":
                RIDGE_ALPHA_GRID,
        },

    scoring=
        "neg_mean_absolute_error",

    cv=
        final_inner_cv,

    n_jobs=-1,

    refit=True,
)


final_parameter_search.fit(

    X_brain_age_train,

    y_brain_age_train,

    groups=
        brain_age_groups,
)


final_brain_age_model = (

    final_parameter_search
    .best_estimator_
)


final_best_alpha = (

    final_parameter_search
    .best_params_[
        "ridge__alpha"
    ]
)


# ============================================================
# OOF predictions for final BAG bias
# ============================================================

final_training_oof_prediction = (

    get_training_oof_predictions(

        estimator=
            final_brain_age_model,

        X_train=
            X_brain_age_train,

        y_train=
            y_brain_age_train,

        groups_train=
            brain_age_groups,

        requested_splits=
            INNER_SPLITS,
    )
)


final_bias_beta, final_bias_alpha = (

    fit_bag_bias(

        age=
            y_brain_age_train,

        predicted_age=
            final_training_oof_prediction,
    )
)


print(

    "Final best Ridge alpha:",

    final_best_alpha
)


print(

    "Final BAG bias beta:",

    final_bias_beta
)


print(

    "Final BAG bias alpha:",

    final_bias_alpha
)


# ============================================================
# PREDICT ALL ELIGIBLE SUBJECTS
# ============================================================

all_brain_age_prediction = np.full(

    len(
        brain_age_metadata_all
    ),

    np.nan,
)


all_brain_age_prediction[

    valid_prediction_mask

] = final_brain_age_model.predict(

    X_brain_age_all[
        valid_prediction_mask
    ]
)


chronological_age_all = (

    brain_age_metadata_all[
        AGE_COL
    ]

    .to_numpy(
        dtype=float
    )
)


# ============================================================
# RAW BAG
# ============================================================

all_raw_bag = (

    all_brain_age_prediction

    -

    chronological_age_all
)


# ============================================================
# CORRECTED BAG
# ============================================================

all_corrected_bag = (

    all_raw_bag

    -

    (

        final_bias_beta

        *

        chronological_age_all

        +

        final_bias_alpha
    )
)


all_corrected_predicted_age = (

    chronological_age_all

    +

    all_corrected_bag
)


# ============================================================
# FINAL TABLE
# ============================================================

brain_age_all_results = (

    brain_age_metadata_all

    .copy()
)


brain_age_all_results[
    "Brain_age_raw"
] = all_brain_age_prediction


brain_age_all_results[
    "BAG_raw"
] = all_raw_bag


brain_age_all_results[
    "Brain_age_bias_corrected"
] = all_corrected_predicted_age


brain_age_all_results[
    "cBAG"
] = all_corrected_bag


print(
    "\nFirst predictions:"
)


display(

    brain_age_all_results[
        [

            DATASET_COL,

            ID_COL,

            AGE_COL,

            DIAGNOSIS_COL,

            "Brain_age_raw",

            "BAG_raw",

            "cBAG",
        ]
    ]

    .head(
        10
    )
)


print(
    "\nMean cBAG by diagnosis:"
)


display(

    brain_age_all_results

    .groupby(

        DIAGNOSIS_COL,

        dropna=False,
    )[
        "cBAG"
    ]

    .agg(
        [

            "count",

            "mean",

            "std",

            "median",
        ]
    )

    .round(
        4
    )
)
# %%
# ============================================================
# CELL 7 — SAVE RESULTS + FINAL MODEL
# ============================================================


brain_age_oof_results.to_csv(

    BRAIN_AGE_OUTPUT_DIR

    / "CN_OOF_brain_age_predictions.csv",

    index=False,
)


brain_age_fold_summary.to_csv(

    BRAIN_AGE_OUTPUT_DIR

    / "brain_age_CV_fold_summary.csv",

    index=False,
)


brain_age_metric_summary.to_csv(

    BRAIN_AGE_OUTPUT_DIR

    / "brain_age_OOF_metric_summary.csv",

    index=False,
)


brain_age_all_results.to_csv(

    BRAIN_AGE_OUTPUT_DIR

    / "all_subject_brain_age_predictions.csv",

    index=False,
)


model_information = {

    "feature_source":
        "harmonized fMRI X_combat",

    "training_population":
        "CN only",

    "best_ridge_alpha":
        float(
            final_best_alpha
        ),

    "bias_beta":
        float(
            final_bias_beta
        ),

    "bias_alpha":
        float(
            final_bias_alpha
        ),

    "age_column":
        AGE_COL,

    "diagnosis_column":
        DIAGNOSIS_COL,

    "dataset_column":
        DATASET_COL,

    "id_column":
        ID_COL,

    "number_of_training_scans":
        int(
            len(
                brain_age_metadata_train
            )
        ),

    "number_of_training_participants":
        int(
            len(
                np.unique(
                    brain_age_groups
                )
            )
        ),

    "number_of_connectome_edges":
        int(
            X_brain_age_train.shape[1]
        ),

    "raw_oof_metrics": {

        key:
            (

                int(
                    value
                )

                if key == "N"

                else

                float(
                    value
                )
            )

        for key, value
        in oof_metrics_raw.items()
    },
}


model_package = {

    "model":
        final_brain_age_model,

    **model_information,
}


joblib.dump(

    model_package,

    BRAIN_AGE_OUTPUT_DIR

    / "final_CN_normative_brain_age_model.joblib",
)


with open(

    BRAIN_AGE_OUTPUT_DIR

    / "brain_age_run_information.json",

    "w",

    encoding="utf-8",

) as file:

    json.dump(

        model_information,

        file,

        indent=2,
    )


print(
    "Brain-age outputs saved successfully:"
)


print(
    BRAIN_AGE_OUTPUT_DIR
)


print(
    "\nSaved files:"
)


for saved_file in sorted(

    BRAIN_AGE_OUTPUT_DIR.iterdir()
):

    print(

        " -",

        saved_file.name
    )
# %%
