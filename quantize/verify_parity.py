"""
Before shipping ANY compiled/quantized artifact, run this to confirm it
produces the same predictions as the original Python model within tolerance.
This is the guardrail against "quantized thresholds silently change the MAE"
mentioned in the risk matrix.

    python -m quantize.verify_parity --bundle artifacts/model_bundle.joblib \
        --onnx artifacts/mobile/onnx/recovery_model_fp16.onnx \
        --holdout data/arc_loan_recovery_train.csv --tol 0.01
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from src.preprocessing import from_logit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--holdout", required=True, help="csv with the raw feature columns")
    ap.add_argument("--tol", type=float, default=0.01,
                     help="max allowed absolute difference in predicted_recovery_rate")
    ap.add_argument("--n-rows", type=int, default=200)
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)
    raw = pd.read_csv(args.holdout).sample(n=args.n_rows, random_state=0)

    X = bundle["preprocessor"].transform(raw)
    py_logit = bundle["model"].predict(X)
    py_pred = from_logit(py_logit)

    sess = ort.InferenceSession(args.onnx)
    input_name = sess.get_inputs()[0].name
    onnx_logit = sess.run(None, {input_name: X.values.astype(np.float32)})[0].ravel()
    onnx_pred = from_logit(onnx_logit)

    diff = np.abs(py_pred - onnx_pred)
    print(f"max abs diff: {diff.max():.6f}   mean abs diff: {diff.mean():.6f}")
    if diff.max() > args.tol:
        raise SystemExit(f"FAILED parity check: max diff {diff.max():.6f} > tol {args.tol}")
    print("[OK] quantized model matches Python model within tolerance — safe to ship.")


if __name__ == "__main__":
    main()
