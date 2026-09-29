"""
Compile the trained tree ensemble into a native shared library (.so for
Android/Linux, dylib for macOS/iOS build chain) for sub-millisecond,
Python-free inference on mobile/edge devices.

Only the TREE ENSEMBLE is compiled here — the preprocessing (clean +
target-encode + one-hot) is NOT part of the compiled binary. You must port
`src/preprocessing.py` logic into native code (Kotlin/Swift) using the
exported `te_maps.json` / `feature_order.json` so mobile predictions match
the Python model exactly. This is the #1 source of "mobile predictions
don't match the server" bugs — see export_manifest below for what to check.

    python -m quantize.export_treelite --bundle artifacts/model_bundle.joblib \
        --out artifacts/mobile/treelite

Supports LightGBM bundles directly. For an XGBoost bundle, treelite reads
the XGBoost booster the same way (see treelite.frontend.from_xgboost).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def export(bundle_path: str, out_dir: str):
    import treelite

    bundle = joblib.load(bundle_path)
    model_type = bundle["model_type"]
    sk_model = bundle["model"]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if model_type == "lightgbm":
        booster = sk_model.booster_
        tl_model = treelite.Model.from_lightgbm(booster)
    elif model_type == "xgboost":
        booster = sk_model.get_booster()
        tl_model = treelite.Model.from_xgboost(booster)
    else:
        raise ValueError(model_type)

    # compile to a native shared library; toolchain='gcc' works on Linux CI runners,
    # switch to 'clang' when cross-compiling for iOS/macOS.
    tl_model.export_lib(
        toolchain="gcc",
        libpath=str(out / "recovery_model.so"),
        params={"parallel_comp": 4},
        verbose=True,
    )

    # everything the native app needs to reproduce preprocessing bit-for-bit
    manifest = {
        "model_type": model_type,
        "feature_order": bundle["feature_columns"],
        "logit_transform": "sigmoid",   # apply sigmoid(raw_output) to get recovery_rate in [0,1]
        "target_encoder_maps": {
            col: series.to_dict()
            for col, series in bundle["preprocessor"].te.maps_.items()
        },
        "target_encoder_global_mean": bundle["preprocessor"].te.global_mean_,
        "one_hot_columns": ["home_ownership", "verification_status", "purpose"],
        "cv_metrics": bundle["cv_metrics"],
        "note": ("Port src/preprocessing.py::clean() + TargetEncoder.transform() into "
                 "native code using this manifest. Validate parity on 100+ real rows "
                 "before shipping (see quantize/verify_parity.py)."),
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[OK] wrote {out / 'recovery_model.so'} and export_manifest.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default="artifacts/mobile/treelite")
    args = ap.parse_args()
    export(args.bundle, args.out)
