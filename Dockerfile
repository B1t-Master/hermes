FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# The legacy RAG stack requires NumPy 1.x; newer PyArrow wheels require 2.x.
RUN pip install --no-cache-dir pyarrow==19.0.1
RUN pip install --no-cache-dir datasets==3.2.0 && pip check

RUN useradd --create-home --uid 10001 hermes
COPY --chown=hermes:hermes app ./app
COPY --chown=hermes:hermes services ./services
COPY --chown=hermes:hermes ingest ./ingest
# Bundled sources are seed data: the entrypoint copies them into the writable
# /app/data_sources volume (uploads and rebuild updates both work).
COPY --chown=hermes:hermes data_sources ./seed_data_sources
COPY --chown=hermes:hermes docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /app/data_sources /app/data /app/.cache/huggingface \
    && chown -R hermes:hermes /app

USER hermes
EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
