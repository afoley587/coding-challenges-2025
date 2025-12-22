FROM python:3.12-slim AS builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=2.1.3

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip install poetry==$POETRY_VERSION

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN pip install poetry-plugin-export
RUN poetry export -f requirements.txt \
    --output requirements.txt \
    --without-hashes \
    --without dev

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    ca-certificates \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN groupadd -r -g 1001 appuser && \
    useradd -r -u 1001 -g appuser -m -d /home/appuser -s /bin/bash appuser

WORKDIR /app

COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser mcp_rag/ /app/mcp_rag

ENV PYTHONPATH=/app/
USER appuser

CMD ["python", "mcp_rag/app.py"]
