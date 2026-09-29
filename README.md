# ARC Recovery Rate — Production Implementation Plan

## Important correction before anything else

Your notebook's **final, saved model is LightGBM**, not XGBoost. Cell 8 trains
an XGBoost model too, but only to *compare* it against LightGBM out-of-fold —
it's never refit on the full data or used to generate `submission.csv`.
Cell 9's `final_model = lgb.LGBMRegressor(...)` is what produced your
predictions. All code here defaults to LightGBM to match what you actually
validated and submitted. If you specifically want XGBoost in production,
pass `--model-type xgboost` everywhere below — every script supports both,
since your OOF comparison showed they perform similarly (correlation ~high,
comparable MAE). I flagged this instead of quietly assuming XGBoost so this
doesn't turn into a change nobody asked for.

**Augmentation was intentionally skipped**, per your instruction. Everything
below only uses real, validated data.

---

## 1. Adding new real data points (no synthetic data)

New rows must pass `src/data_validation.py::validate_new_rows()` before
they're allowed anywhere near training. It checks, against the exact rules
your notebook already encodes:

- schema completeness (every required column present)
- no duplicate/missing `loan_id`, no collision with existing training IDs
- hard numeric ranges (FICO 300–850, rates/percentages 0–100, no negative
  amounts, `fico_range_low <= fico_range_high`, etc.)
- `collateral_flag == 0 → collateral_value == 0` consistency (warns if the
  source data violates this, since `clean()` will silently zero it anyway)
- valid `home_ownership` categories
- contradictory duplicates (identical features, wildly different target)

Only rows that pass move into `--new` for `src/train.py`. Nothing here
invents data — it's a gate, not a generator.

## 2. Retraining

```bash
python -m src.train \
    --base data/arc_loan_recovery_train.csv \
    --new  data/new_rows_batch1.csv \
    --model-type lightgbm \
    --out artifacts/model_bundle.joblib
```

This runs the *same* 5-fold CV, early stopping, target-encoding-fit-inside-
each-fold discipline, and logit transform as your notebook — just on the
combined dataset — then fits the final model on 100% of the merged data.
The output is a single `joblib` bundle containing the fitted preprocessor
(target-encoding maps + one-hot column order) and the model together, so
training and serving can never drift apart.

## 3. Deployment (local AND cloud — same artifact, same container)

```
Client (web/app/core banking)
        │  JSON
        ▼
 FastAPI (api/main.py)  ── loads artifacts/model_bundle.joblib once at startup
        │
        ├─ Pydantic validation (api/schema.py)
        ├─ src/preprocessing.py transform (identical to training)
        └─ model.predict → sigmoid → predicted_recovery_rate ∈ [0,1]
```

**Run locally** (no cloud account needed):
```bash
docker compose up --build
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_loan.json
```

**Run in the cloud** — the exact same image, no code changes:
```bash
docker build -t arc-recovery-api .
docker push <your-registry>/arc-recovery-api
# then point Cloud Run / App Runner / Azure Container Apps at that image
```
This directly answers "local or server" — it's the same Docker image either
way; only where you point `docker run` / `docker compose up` differs. Swap a
new model by replacing `artifacts/model_bundle.joblib` and restarting — no
rebuild needed since it's mounted read-only in `docker-compose.yml`.

## 4. Mobile / edge (hybrid quantization)

Two complementary paths — use Treelite for the fastest/smallest native
binary, ONNX for cross-platform portability with a single artifact:

```bash
python -m quantize.export_treelite --bundle artifacts/model_bundle.joblib
python -m quantize.export_onnx     --bundle artifacts/model_bundle.joblib
python -m quantize.verify_parity   --bundle artifacts/model_bundle.joblib \
    --onnx artifacts/mobile/onnx/recovery_model_fp16.onnx \
    --holdout data/arc_loan_recovery_train.csv
```

- **Treelite** → compiled `.so`, no Python/LightGBM runtime, <1 MB, µs-level inference.
- **ONNX + float16** → one file runs on Android (ONNX Runtime Mobile) and iOS via the same SDK.
- **`verify_parity.py` is mandatory before shipping** — it's the guardrail
  against the "quantized thresholds silently change the accuracy" risk. It
  fails the build if predictions drift by more than your tolerance.

The one thing that does **not** get compiled automatically: preprocessing.
`export_manifest.json` (written by both export scripts) contains the exact
target-encoding maps and feature order so you can port `clean()` +
target-encoding into Kotlin/Swift and get bit-identical results. This is the
single most common way mobile predictions silently diverge from the server —
budget real test time for it, not just the model export.

## 5. Feasibility matrix (updated for your actual asks)

| Objective | Feasibility | Tool | Key risk |
|---|---|---|---|
| Add real data + retrain | High | `data_validation.py` + `train.py` | Bad rows silently corrupting the model — mitigated by the validation gate |
| Local deployment | Very High | Docker Compose | None significant |
| Cloud deployment | Very High | Same Docker image → Cloud Run/App Runner | Must bundle preprocessor with model (done via joblib bundle) |
| Mobile compilation | Very High | Treelite | Preprocessing must be ported natively — verify with `verify_parity.py` |
| Mobile quantization | High | ONNX float16 | Always parity-check before shipping |
| Local **and** server simultaneously | Yes | Identical container, different host | Keep `MODEL_BUNDLE_PATH` as the only environment difference |

## Recommended order of execution

1. Validate any new rows (`data_validation.py`) — fix or reject anything that fails.
2. Retrain (`train.py`) and confirm CV MAE hasn't regressed vs. your notebook's 0.123.
3. Stand up the API locally with `docker compose up`, hit `/health` and `/predict`.
4. Push the same image to your cloud container service of choice.
5. Export Treelite + ONNX, run `verify_parity.py`, then port preprocessing to native mobile code and test on real device data before release.
