"""
Preprocessing logic for the ARC recovery-rate model.

IMPORTANT: this file is a faithful, 1:1 port of the cleaning / target-encoding
logic from the original notebook (cells 2 and 4). Training and the API MUST
both import from here so the transformation applied at inference time is
mathematically identical to the one used during training. Never re-implement
this logic a second time somewhere else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 42
TARGET = "recovery_rate"
ID_COL = "loan_id"

# Exact feature list the model was trained on (from the notebook, cell 3 output).
# Order does not matter for training, but the ONE_HOT_COLS / TE_COLS split below
# does — do not edit without re-running the full training pipeline.
FEATURE_COLUMNS = [
    "loan_amount", "term_months", "grade", "sub_grade", "int_rate",
    "emp_length_years", "home_ownership", "annual_income", "verification_status",
    "purpose", "dti", "fico_range_low", "fico_range_high", "delinq_2yrs",
    "pub_rec", "open_acc", "total_acc", "revol_bal", "revol_util",
    "collections_12_mths_ex_med", "collateral_flag", "collateral_value",
    "total_pymnt_before_default", "outstanding_principal",
    "days_past_due_at_default",
    # engineered missing-flags, added by clean()
    "annual_income_missing", "emp_length_years_missing",
    "revol_util_missing", "dti_missing",
]

# Columns that only exist because of post-default outcome data (leakage if
# used as a feature for a *pre-recovery* prediction).
LEAKAGE_COLS = [
    "recoveries", "collection_recovery_fee",
    "payment_plan_status", "months_in_recovery", "recovery_status",
]

TE_COLS = ["grade", "sub_grade"]          # target-encoded (high-cardinality ordinal-ish)
ONE_HOT_COLS = ["home_ownership", "verification_status", "purpose"]
TE_SMOOTHING = 20


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Exact port of notebook cell 2."""
    df = df.copy()
    df["home_ownership"] = df["home_ownership"].str.upper()
    df["dti"] = df["dti"].replace(-999, np.nan)
    df["collateral_value"] = np.where(df["collateral_flag"] == 0, 0, df["collateral_value"])

    for col in ["annual_income", "emp_length_years", "revol_util", "dti"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    return df


def to_logit(p, eps: float = 1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def from_logit(z):
    return np.clip(1 / (1 + np.exp(-z)), 0, 1)


class TargetEncoder:
    """Smoothed mean target-encoder, fit ONLY on training data.

    Stores the fitted maps so the exact same numbers are reused at inference
    time (this is the piece that is easiest to get subtly wrong when you
    "productionize" a notebook — refitting on new data at inference time is
    a silent correctness bug, not just a style issue).
    """

    def __init__(self, cols=TE_COLS, smoothing: float = TE_SMOOTHING):
        self.cols = cols
        self.smoothing = smoothing
        self.maps_: dict[str, pd.Series] = {}
        self.global_mean_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        self.global_mean_ = float(y.mean())
        for col in self.cols:
            stats = y.groupby(X[col]).agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
                stats["count"] + self.smoothing
            )
            self.maps_[col] = smoothed
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.global_mean_ is None:
            raise RuntimeError("TargetEncoder.transform called before fit()")
        out = X.copy()
        for col in self.cols:
            out[col + "_te"] = out[col].map(self.maps_[col]).fillna(self.global_mean_)
        return out.drop(columns=self.cols)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class FullPreprocessor:
    """clean() -> target-encode -> one-hot, bundled into one fit/transform
    object so it can be pickled/joblib-dumped alongside the model and reused
    verbatim by the API. Also fixes the one-hot column set learned at train
    time so inference-time data can never silently produce a different
    number of columns than the model expects.
    """

    def __init__(self):
        self.te = TargetEncoder()
        self.train_columns_: list[str] | None = None

    def fit_transform(self, raw_df: pd.DataFrame, y_raw: pd.Series) -> pd.DataFrame:
        df = clean(raw_df)
        X = df[FEATURE_COLUMNS]
        X = self.te.fit_transform(X, y_raw)
        X = pd.get_dummies(X, columns=ONE_HOT_COLS)
        self.train_columns_ = list(X.columns)
        return X

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if self.train_columns_ is None:
            raise RuntimeError("FullPreprocessor.transform called before fit_transform()")
        df = clean(raw_df)
        X = df[FEATURE_COLUMNS]
        X = self.te.transform(X)
        X = pd.get_dummies(X, columns=ONE_HOT_COLS)
        X = X.reindex(columns=self.train_columns_, fill_value=0)
        return X
