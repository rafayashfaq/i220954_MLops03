# Multi-stage production Dockerfile for FastAPI Prediction Service
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# Non-root security user
RUN groupadd -g 10001 mlops && \
    useradd -u 10001 -g mlops -s /bin/bash mlops

COPY --from=builder /root/.local /home/mlops/.local
COPY api/ /app/api/
COPY models/ /app/models/

ENV PATH=/home/mlops/.local/bin:$PATH \
    PYTHONPATH=/app \
    PORT=8000 \
    ENVIRONMENT=production

RUN chown -R mlops:mlops /app

USER mlops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
