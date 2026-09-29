"""
Retrain the recovery-rate model on the ORIGINAL training data plus any new,
validated real rows, and export one deployable bundle.

    python -m src.train \
        --base data/arc_loan_recovery_train.csv \
        --new  data/new_rows_batch1.csv \
        --model-type lightgbm \
        --out artifacts/model_bundle.joblib

Notes:
- No synthetic data is generated here. --new must be real rows that already
  passed src.data_validation.validate_new_rows.
- --model-type defaults to "lightgbm" because that is what the notebook
  actually saved and used for submission.csv (cell 9). XGBoost was only
  compared against it out-of-fold in cell 8 and was NOT the model that
  produced the final predictions. Pass --model-type xgboost if you want to
  deploy XGBoost instead — the code path is symmetrical either way.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocessing import (
    FEATURE_COLUMNS, ID_COL, LEAKAGE_COLS, ONE_HOT_COLS, TARGET,
    FullPreprocessor, TargetEncoder, clean, from_logit, to_logit,
)
from .data_validation import validate_new_rows

RANDOM_STATE = 42
N_FOLDS = 5


def load_and_merge(base_path: str, new_path: str | None) -> pd.DataFrame:
    base_df = pd.read_csv(base_path)
    if new_path is None:
        return base_df

    new_df = pd.read_csv(new_path)
    report = validate_new_rows(new_df, existing_ids=set(base_df[ID_COL]))
    for w in report.warnings:
        print(f"[WARN] {w}")
    if not report.ok:
        print("[ERROR] New data failed validation — refusing to train on it:")
        for e in report.errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"[OK] {len(new_df)} new row(s) passed validation and will be merged in.")
    return pd.concat([base_df, new_df], ignore_index=True)


def build_fold_matrices(X_tr_raw, y_tr_raw, X_val_raw):
    te = TargetEncoder()
    X_tr = te.fit_transform(X_tr_raw, y_tr_raw)
    X_val = te.transform(X_val_raw)
    X_tr = pd.get_dummies(X_tr, columns=ONE_HOT_COLS)
    X_val = pd.get_dummies(X_val, columns=ONE_HOT_COLS)
    X_val = X_val.reindex(columns=X_tr.columns, fill_value=0)
    return X_tr, X_val


def cross_validate(X_full: pd.DataFrame, y: pd.Series, y_logit: pd.Series, model_type: str) -> dict:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_logit = np.zeros(len(X_full))
    best_iters = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_full)):
        X_tr_raw, X_val_raw = X_full.iloc[tr_idx], X_full.iloc[val_idx]
        y_tr_logit, y_val_logit = y_logit.iloc[tr_idx], y_logit.iloc[val_idx]
        y_tr_raw = y.iloc[tr_idx]

        X_tr, X_val = build_fold_matrices(X_tr_raw, y_tr_raw, X_val_raw)
        model, best_iter = _fit_one(model_type, X_tr, y_tr_logit, X_val, y_val_logit)
        oof_logit[val_idx] = model.predict(X_val)
        best_iters.append(best_iter)
        print(f"  fold {fold}: best_iteration={best_iter}")

    oof_pred = from_logit(oof_logit)
    mae = mean_absolute_error(y, oof_pred)
    rmse = mean_squared_error(y, oof_pred) ** 0.5
    r2 = r2_score(y, oof_pred)
    print(f"[CV] MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mean_best_iteration": int(np.mean(best_iters))}


def _fit_one(model_type, X_tr, y_tr_logit, X_val, y_val_logit):
    if model_type == "lightgbm":
        import lightgbm as lgb
        params = dict(n_estimators=1000, learning_rate=0.03, num_leaves=31,
                      min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                      random_state=RANDOM_STATE, verbosity=-1)
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr_logit, eval_set=[(X_val, y_val_logit)],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        return model, model.best_iteration_
    elif model_type == "xgboost":
        import xgboost as xgb
        params = dict(n_estimators=1000, learning_rate=0.03, max_depth=5, min_child_weight=20,
                      subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1,
                      early_stopping_rounds=50, eval_metric="mae")
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr_logit, eval_set=[(X_val, y_val_logit)], verbose=False)
        return model, model.best_iteration
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def fit_final_model(model_type: str, n_estimators: int, X: pd.DataFrame, y_logit: pd.Series):
    if model_type == "lightgbm":
        import lightgbm as lgb
        params = dict(n_estimators=n_estimators, learning_rate=0.03, num_leaves=31,
                      min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                      random_state=RANDOM_STATE, verbosity=-1)
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y_logit)
        return model
    elif model_type == "xgboost":
        import xgboost as xgb
        params = dict(n_estimators=n_estimators, learning_rate=0.03, max_depth=5,
                      min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                      random_state=RANDOM_STATE, n_jobs=-1)
        model = xgb.XGBRegressor(**params)
        model.fit(X, y_logit)
        return model
    raise ValueError(model_type)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to the original training csv")
    ap.add_argument("--new", default=None, help="optional path to new, real, validated rows")
    ap.add_argument("--model-type", choices=["lightgbm", "xgboost"], default="lightgbm")
    ap.add_argument("--out", default="artifacts/model_bundle.joblib")
    args = ap.parse_args()

    raw = load_and_merge(args.base, args.new)
    df = clean(raw)
    y = df[TARGET].astype(float)
    y_logit = to_logit(y)

    drop_cols = [c for c in LEAKAGE_COLS + [TARGET, ID_COL] if c in df.columns]
    X_full = df.drop(columns=drop_cols)
    # keep exactly the modeled feature columns, in a stable order
    X_full = X_full[[c for c in X_full.columns if c in FEATURE_COLUMNS]]

    print(f"[INFO] training on {len(X_full)} rows, {X_full.shape[1]} raw feature columns")
    cv_metrics = cross_validate(X_full, y, y_logit, args.model_type)

    # final fit on 100% of the data, preprocessing bundled for serving
    pre = FullPreprocessor()
    X_enc = pre.fit_transform(raw, y)  # fits TE + one-hot on ALL rows, for the deployed model
    final_model = fit_final_model(args.model_type, cv_metrics["mean_best_iteration"], X_enc, y_logit)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final_model,
        "model_type": args.model_type,
        "preprocessor": pre,
        "feature_columns": pre.train_columns_,
        "cv_metrics": cv_metrics,
        "n_training_rows": len(X_full),
    }
    joblib.dump(bundle, args.out)
    print(f"[OK] wrote bundle to {args.out}")

    metrics_path = Path(args.out).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(cv_metrics, indent=2))
    print(f"[OK] wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
