# syntax=docker/dockerfile:1

# ---- Stage 1: build the frontend -------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


# ---- Stage 2: backend runtime ----------------------------------------------
FROM python:3.11-slim AS runtime

# HF_HOME keeps the model cache inside the image so the pre-download below is
# reused at runtime; the OFFLINE flags stop any runtime model fetch.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# libgomp1 is required by torch; the rest ship manylinux wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Pre-download the ML models (sentiment + embeddings) into the image so a cold
# start never pays a multi-hundred-MB fetch.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" \
    && python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-sentiment-latest', top_k=None)"

COPY app ./app
COPY ingest ./ingest
COPY services ./services
COPY scripts ./scripts
COPY data_sources ./data_sources
COPY --from=frontend /web/dist ./frontend/dist

# Run as an unprivileged user; the model cache is owned by it.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
# Render (and most PaaS) provide $PORT; fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
