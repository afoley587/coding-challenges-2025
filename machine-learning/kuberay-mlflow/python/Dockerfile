FROM rayproject/ray:2.52.1-py312-cu128-aarch64 AS builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=2.1.3

RUN pip install poetry==$POETRY_VERSION

WORKDIR /serve_app

COPY pyproject.toml poetry.lock* ./

RUN pip install poetry-plugin-export
RUN poetry export -f requirements.txt \
    --output requirements.txt \
    --without-hashes \
    --without dev

FROM rayproject/ray:2.52.1-py312-cu128-aarch64 AS runtime

WORKDIR /serve_app

COPY --from=builder /serve_app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
COPY kuberay_mlflow/ ./kuberay_mlflow/

LABEL \
    org.opencontainers.image.title="ML API" \
    org.opencontainers.image.description="ML API using Ray, Kubeflow, and MLFlow" \
    org.opencontainers.image.vendor="ML DevOps Team" \
    security.scan="enabled"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import ray; print('OK')" || exit 1
