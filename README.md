# Project Kingfisher: ARC Recovery Rate Engine

End-to-end machine learning service & deployment pipeline for predicting recovery rates on defaulted and distressed retail credit portfolios. Designed for Asset Reconstruction Companies (ARCs), debt buyers, and special situations desks to estimate expected liquidation proceeds with zero data leakage, strict schema gating, and high-throughput inference across cloud, containerized, and mobile edge environments.

---

## System Architecture

```
                    Raw Ingestion / Batch Data
                                │
                                ▼
               ┌─────────────────────────────────┐
               │  Data Validation Gate           │
               │  src/data_validation.py         │
               └────────────────┬────────────────┘
                                │ (Valid rows only, zero synthetic data)
                                ▼
               ┌─────────────────────────────────┐
               │  CV Target Encoding & Training  │
               │  src/train.py                   │
               └────────────────┬────────────────┘
                                │ Produces unified bundle
                                ▼
               ┌─────────────────────────────────┐
               │  artifacts/model_bundle.joblib  │
               │  (Fitted Preprocessor + Model)  │
               └───────┬─────────────────┬───────┘
                       │                 │
            Deploy API │                 │ Compile & Quantize
                       ▼                 ▼
        ┌─────────────────────────┐   ┌───────────────────────────┐
        │ FastAPI Microservice    │   │ Edge Export (quantize/)   │
        │ api/main.py             │   │ • Treelite (.so / .dylib) │
        │ • Single & Batch REST   │   │ • ONNX float16 (Android)  │
        │ • Docker / Cloud Run    │   │ • verify_parity.py check  │
        └─────────────────────────┘   └───────────────────────────┘
```

---

## Core Principles

1. **LightGBM Primary Ensemble**: The default production model is a LightGBM regressor trained on logit-transformed targets:
   $$\text{logit}(y) = \ln\left(\frac{y}{1 - y}\right)$$
   Predictions are transformed back via the sigmoid function, mathematically guaranteeing that all outputs remain bounded within $(0, 1)$. Full support for XGBoost is maintained (`--model-type xgboost`) via symmetric training and serving interfaces.
2. **Zero Synthetic Leakage**: Strict validation gates reject corrupt data or out-of-range artifacts before retraining. No synthetic or generated records enter the training pipeline.
3. **Atomic Preprocessor & Model Bundling**: Preprocessors (cross-validated target encoding mappings and one-hot encoders) are serialized together with the trained estimator in a single `joblib` bundle. Serving and training pipelines cannot drift.
4. **Environment-Agnostic Serving**: The same container image serves local development (`docker compose`) and serverless cloud runtimes (Google Cloud Run, AWS App Runner / ECS, Azure Container Apps).
5. **Verified Edge Portability**: Native C shared libraries (Treelite) and cross-platform runtimes (ONNX FP16) allow offline mobile scoring with sub-millisecond latency, verified by automated precision checks.

---

## Repository Structure

```
Project-Kingfisher/
├── api/
│   ├── main.py              # FastAPI application with /health, /predict, /predict/batch
│   └── schema.py            # Pydantic input/output schemas with range validation
├── artifacts/
│   ├── model_bundle.joblib  # Serialized preprocessor + LightGBM model bundle
│   └── model_bundle.metrics.json # Out-of-fold cross-validation metrics
├── quantize/
│   ├── export_onnx.py       # Converts model to ONNX FP16 for iOS/Android
│   ├── export_treelite.py   # Compiles ensemble into standalone native C binary (.so)
│   └── verify_parity.py     # Automated numerical drift test (Python vs. Quantized)
├── src/
│   ├── data_validation.py   # Schema integrity and domain constraint validator
│   ├── preprocessing.py     # Out-of-fold target encoding, cleaning, and logit transforms
│   └── train.py             # 5-fold CV retraining pipeline and artifact packager
├── Dockerfile               # Multi-stage production container image
├── docker-compose.yml       # Local container orchestration with live artifact mount
└── requirements.txt         # Core dependencies and edge export libraries
```

---

## 1. Data Ingestion & Quality Gate

All incoming real data must pass `validate_new_rows` in `src/data_validation.py` before inclusion in the training set:

- **Schema completeness**: Ensures all required feature and target columns are present.
- **Identifier integrity**: Enforces non-null `loan_id` values with no collisions against existing training IDs.
- **Domain ranges**: Validates hard numeric bounds (FICO score between 300–850, rates/percentages within 0–100, non-negative amounts, and `fico_range_low <= fico_range_high`).
- **Collateral consistency**: Enforces that loans with `collateral_flag == 0` possess `collateral_value == 0`.
- **Category validation**: Restricts categorical fields (e.g., `home_ownership`) to permitted categories.
- **Duplicate reconciliation**: Detects conflicting records possessing identical features but divergent recovery rates.

---

## 2. Model Training & Retraining

Retraining merges validated new batches with historical baseline datasets using 5-fold cross-validation, target-encoding fit strictly inside each fold, and logit-space optimization.

```bash
python -m src.train \
    --base data/arc_loan_recovery_train.csv \
    --new data/new_rows_batch1.csv \
    --model-type lightgbm \
    --out artifacts/model_bundle.joblib
```

To train an XGBoost bundle instead:

```bash
python -m src.train \
    --base data/arc_loan_recovery_train.csv \
    --model-type xgboost \
    --out artifacts/model_bundle.joblib
```

The output bundle bundles:
- `preprocessor`: Fitted `FullPreprocessor` (fold-isolated target encoding maps and one-hot encoding columns).
- `model`: Estimator trained on 100% of the dataset post-validation.
- `metadata`: Training row counts, timestamp, feature column list, and out-of-fold evaluation metrics (MAE, RMSE, $R^2$).

---

## 3. Serving & Deployment

### Local Development

Start the service via Uvicorn:

```bash
uvicorn api.main:app --reload --port 8000
```

Or spin up the pre-configured Docker Compose environment:

```bash
docker compose up --build
```

The artifact directory is mounted read-only (`./artifacts:/app/artifacts:ro`). Updating models does not require rebuilding the Docker image—simply swap `artifacts/model_bundle.joblib` and restart the container.

### Cloud Deployment

The container image is runtime-agnostic and ready for serverless container hosts:

```bash
# Build and tag
docker build -t arc-recovery-api .

# Push to your cloud registry
docker tag arc-recovery-api <registry-url>/arc-recovery-api:latest
docker push <registry-url>/arc-recovery-api:latest
```

Set the following environment variables on your cloud container service (e.g., Cloud Run, App Runner, ECS):
- `MODEL_BUNDLE_PATH`: Path to the bundle (`/app/artifacts/model_bundle.joblib` by default).
- `MODEL_VERSION`: Active deployment identifier (e.g., `v1`).

---

## 4. API Reference

### Health Check

```bash
curl -X GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_type": "lightgbm",
  "cv_metrics": {
    "mae": 0.1234,
    "rmse": 0.1548,
    "r2": 0.3495,
    "mean_best_iteration": 164
  }
}
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_id": "LN-2024-9081",
    "loan_amount": 18000.0,
    "term_months": 36,
    "grade": "C",
    "sub_grade": "C2",
    "int_rate": 14.5,
    "emp_length_years": 5.0,
    "home_ownership": "MORTGAGE",
    "annual_income": 65000.0,
    "verification_status": "Source Verified",
    "purpose": "debt_consolidation",
    "dti": 18.4,
    "fico_range_low": 670,
    "fico_range_high": 674,
    "delinq_2yrs": 0,
    "pub_rec": 0,
    "open_acc": 11,
    "total_acc": 22,
    "revol_bal": 14200.0,
    "revol_util": 54.2,
    "collections_12_mths_ex_med": 0,
    "collateral_flag": 0,
    "collateral_value": 0.0,
    "total_pymnt_before_default": 4200.0,
    "outstanding_principal": 13800.0,
    "days_past_due_at_default": 120
  }'
```

**Response:**
```json
{
  "loan_id": "LN-2024-9081",
  "predicted_recovery_rate": 0.341829,
  "model_type": "lightgbm",
  "model_version": "v1"
}
```

### Batch Prediction

`POST /predict/batch` accepts an array of loan records and returns a list of prediction objects in the same order.

---

## 5. Edge & Mobile Compilation

For field operations, offline debt evaluations, and native mobile clients, Kingfisher provides two complementary compilation targets:

### Option A: Treelite (C Shared Library)
Compiles tree structures into standalone native machine code (`.so` / `.dylib`). Removes Python and runtime dependencies entirely, achieving sub-millisecond execution with a binary size under 1 MB.

```bash
python -m quantize.export_treelite \
    --bundle artifacts/model_bundle.joblib \
    --out artifacts/mobile/treelite
```

### Option B: ONNX Runtime (Float16 Quantization)
Exports models into ONNX format with float16 leaf and threshold quantization for unified deployment across Android and iOS via the ONNX Runtime Mobile SDK.

```bash
python -m quantize.export_onnx \
    --bundle artifacts/model_bundle.joblib \
    --out artifacts/mobile/onnx
```

### Numerical Parity Verification

Quantization and native porting must not introduce accuracy regression. The automated verification suite asserts that edge predictions remain within a strict tolerance threshold ($\Delta \le 0.01$) against the reference model:

```bash
python -m quantize.verify_parity \
    --bundle artifacts/model_bundle.joblib \
    --onnx artifacts/mobile/onnx/recovery_model_fp16.onnx \
    --holdout data/arc_loan_recovery_train.csv \
    --tol 0.01
```

> **Client Implementation Note**: Tree export targets compile model structures only. Feature transformation (`clean()` logic and target encoding) must be mirrored in native client code (Swift / Kotlin). The export process outputs `export_manifest.json`, detailing exact feature ordering and target-encoding mapping tables to ensure client-server calculation parity.

---

## 6. Implementation Summary

| Objective | Target / Runtime | Tooling | Mitigation / Control |
|---|---|---|---|
| **Data Ingestion** | Batch / Streaming | `src/data_validation.py` | Schema validation gate rejects malformed rows and prevents data drift |
| **Retraining** | Compute node / CI | `src/train.py` | 5-fold CV with fold-isolated target encoding to prevent leakage |
| **Local Serving** | Developer workstations | Docker Compose / FastAPI | Volume-mounted bundle allows hot artifact reload without image rebuild |
| **Cloud Serving** | Cloud Run / App Runner / ECS | Production Docker image | Unified preprocessor/model bundle eliminates serving drift |
| **Mobile Runtime (Native)** | Android / iOS native C | Treelite | Native compilation; verified via `export_manifest.json` |
| **Mobile Runtime (Cross-Platform)** | Mobile ONNX Runtime | ONNX float16 | Build pipeline enforces `verify_parity.py` before release |
