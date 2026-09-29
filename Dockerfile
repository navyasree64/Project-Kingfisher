FROM python:3.12-slim

WORKDIR /app

# system lib required by LightGBM (OpenMP for parallel tree building)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# only the runtime deps needed to SERVE, not to train/export — keeps the image small
COPY requirements.txt .
RUN pip install --no-cache-dir \
    pandas==2.2.2 numpy==1.26.4 scikit-learn==1.5.1 \
    lightgbm==4.5.0 joblib==1.4.2 \
    fastapi==0.112.0 "uvicorn[standard]==0.30.5" pydantic==2.8.2

COPY src ./src
COPY api ./api
COPY artifacts ./artifacts

ENV MODEL_BUNDLE_PATH=/app/artifacts/model_bundle.joblib
ENV MODEL_VERSION=v1

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
