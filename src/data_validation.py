"""
Validation gate for NEW REAL rows you want to add to the training set.

No synthetic data is generated anywhere in this file (augmentation was
explicitly excluded). This only checks that genuinely new rows are internally
consistent before they're allowed to touch the training data — bad rows here
would otherwise silently corrupt the model, which is the #1 way "just add
more data" backfires.

Usage:
    from src.data_validation import validate_new_rows
    report = validate_new_rows(new_df)
    if not report.ok:
        raise ValueError(report.errors)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .preprocessing import FEATURE_COLUMNS, ID_COL, TARGET

REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET, ID_COL]

# (column, min, max) — hard business-logic bounds. None = unbounded on that side.
RANGE_RULES = [
    ("loan_amount", 0, None),
    ("term_months", 1, 600),
    ("int_rate", 0, 100),
    ("emp_length_years", 0, 60),
    ("annual_income", 0, None),
    ("dti", -999, 100),          # -999 is the documented sentinel for missing, handled by clean()
    ("fico_range_low", 300, 850),
    ("fico_range_high", 300, 850),
    ("delinq_2yrs", 0, None),
    ("pub_rec", 0, None),
    ("open_acc", 0, None),
    ("total_acc", 0, None),
    ("revol_bal", 0, None),
    ("revol_util", 0, 100),
    ("collections_12_mths_ex_med", 0, None),
    ("collateral_value", 0, None),
    ("total_pymnt_before_default", 0, None),
    ("outstanding_principal", 0, None),
    ("days_past_due_at_default", 0, None),
    (TARGET, 0, 1),
]

VALID_HOME_OWNERSHIP = {"ANY", "MORTGAGE", "OWN", "RENT"}  # after .upper()
VALID_COLLATERAL_FLAG = {0, 1}


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows_checked: int = 0


def validate_new_rows(df: pd.DataFrame, existing_ids: set[str] | None = None) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    # 1. schema
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return ValidationReport(ok=False, errors=errors, n_rows_checked=len(df))

    # 2. duplicate / missing IDs
    if df[ID_COL].isna().any():
        errors.append(f"{df[ID_COL].isna().sum()} row(s) have a null {ID_COL}")
    dup = df[ID_COL].duplicated()
    if dup.any():
        errors.append(f"{dup.sum()} duplicate {ID_COL} value(s) within the new batch")
    if existing_ids:
        overlap = set(df[ID_COL]) & existing_ids
        if overlap:
            errors.append(f"{len(overlap)} {ID_COL}(s) already exist in the training set: "
                           f"{sorted(list(overlap))[:5]}{'...' if len(overlap) > 5 else ''}")

    # 3. range / business-rule checks
    for col, lo, hi in RANGE_RULES:
        if col not in df.columns:
            continue
        series = df[col]
        # dti's sentinel -999 is legitimate for "missing" and is handled downstream,
        # so exclude it before range-checking the real range.
        check = series[series != -999] if col == "dti" else series
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= (check < lo)
        if hi is not None:
            bad |= (check > hi)
        bad = bad.reindex(df.index, fill_value=False)
        if bad.any():
            bad_ids = df.loc[bad, ID_COL].tolist()
            errors.append(f"{col}: {bad.sum()} row(s) outside [{lo}, {hi}] -> ids {bad_ids[:5]}"
                           f"{'...' if len(bad_ids) > 5 else ''}")

    # 4. fico_range_low <= fico_range_high
    if {"fico_range_low", "fico_range_high"} <= set(df.columns):
        inverted = df["fico_range_low"] > df["fico_range_high"]
        if inverted.any():
            errors.append(f"{inverted.sum()} row(s) have fico_range_low > fico_range_high")

    # 5. collateral consistency: flag == 0 must imply value == 0 (or will be forced to 0
    #    by clean() anyway — but a mismatch usually means an upstream data bug, so flag it)
    if {"collateral_flag", "collateral_value"} <= set(df.columns):
        mismatch = (df["collateral_flag"] == 0) & (df["collateral_value"].fillna(0) != 0)
        if mismatch.any():
            warnings.append(
                f"{mismatch.sum()} row(s) have collateral_flag=0 but nonzero collateral_value "
                "(will be zeroed out automatically by clean(), but double-check the source data)"
            )
        bad_flag = ~df["collateral_flag"].isin(VALID_COLLATERAL_FLAG)
        if bad_flag.any():
            errors.append(f"collateral_flag: {bad_flag.sum()} row(s) not in {VALID_COLLATERAL_FLAG}")

    # 6. categorical domains
    if "home_ownership" in df.columns:
        bad_ho = ~df["home_ownership"].str.upper().isin(VALID_HOME_OWNERSHIP)
        if bad_ho.any():
            errors.append(f"home_ownership: {bad_ho.sum()} row(s) not in {VALID_HOME_OWNERSHIP}")

    # 7. target sanity: recovery_rate should correlate with recovery_status if present
    if "recovery_status" in df.columns and TARGET in df.columns:
        closed_low = (df["recovery_status"] == "Closed") & (df[TARGET] < 0.02)
        if closed_low.any():
            warnings.append(
                f"{closed_low.sum()} 'Closed' row(s) have recovery_rate < 2% — verify this is real, "
                "not a data-entry error, before training on it."
            )

    # 8. duplicate feature rows with materially different targets (contradictory data)
    feat_no_id = [c for c in FEATURE_COLUMNS if c in df.columns]
    dup_feats = df.duplicated(subset=feat_no_id, keep=False)
    if dup_feats.any():
        grp = df[dup_feats].groupby(feat_no_id)[TARGET].agg(["min", "max"])
        contradictory = grp[(grp["max"] - grp["min"]) > 0.15]
        if len(contradictory) > 0:
            warnings.append(
                f"{len(contradictory)} group(s) of rows share identical features but disagree on "
                f"{TARGET} by >0.15 — likely duplicate/contradictory records."
            )

    ok = len(errors) == 0
    return ValidationReport(ok=ok, errors=errors, warnings=warnings, n_rows_checked=len(df))
