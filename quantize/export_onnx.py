"""
Export the tree model to ONNX and quantize split thresholds / leaf values to
float16, for cross-platform mobile inference via ONNX Runtime Mobile
(Java/Kotlin on Android, Swift on iOS).

    python -m quantize.export_onnx --bundle artifacts/model_bundle.joblib \
        --out artifacts/mobile/onnx

Same caveat as export_treelite.py: this only exports the MODEL. Preprocessing
must be ported to native code separately using export_manifest.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import onnx
from onnxconverter_common import float16


def export(bundle_path: str, out_dir: str):
    bundle = joblib.load(bundle_path)
    model_type = bundle["model_type"]
    sk_model = bundle["model"]
    n_features = len(bundle["feature_columns"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if model_type == "lightgbm":
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        onnx_model = convert_lightgbm(
            sk_model, initial_types=[("input", FloatTensorType([None, n_features]))]
        )
    elif model_type == "xgboost":
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
        onnx_model = convert_xgboost(
            sk_model, initial_types=[("input", FloatTensorType([None, n_features]))]
        )
    else:
        raise ValueError(model_type)

    fp32_path = out / "recovery_model_fp32.onnx"
    onnx.save(onnx_model, str(fp32_path))

    # quantize weights/thresholds to float16 -> ~2x smaller, negligible accuracy loss for tree splits
    fp16_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
    fp16_path = out / "recovery_model_fp16.onnx"
    onnx.save(fp16_model, str(fp16_path))

    manifest_src = json.loads((Path("artifacts/mobile/treelite/export_manifest.json").read_text())
                               if Path("artifacts/mobile/treelite/export_manifest.json").exists()
                               else "{}")
    manifest = {
        **manifest_src,
        "onnx_input_name": "input",
        "onnx_feature_order": bundle["feature_columns"],
        "fp32_size_bytes": fp32_path.stat().st_size,
        "fp16_size_bytes": fp16_path.stat().st_size,
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[OK] fp32: {fp32_path.stat().st_size/1024:.1f} KB, "
          f"fp16: {fp16_path.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default="artifacts/mobile/onnx")
    args = ap.parse_args()
    export(args.bundle, args.out)
