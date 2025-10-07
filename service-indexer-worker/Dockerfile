# Dockerfile for Indexer Worker
#
# Purpose:
# - Package the Celery worker that performs background ingestion jobs
#   (reindexing, batch embedding, cleanup) for the ML platform.
#
# Notes:
# - Uses Python 3.11 slim base.
# - Runs as a non-root user.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared libraries
COPY ../libs ./libs

# Copy application code
COPY . .

# Create non-root user for safer runtime
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

# Health check (placeholder): replace with a lightweight task or metrics probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python - <<'PY' || exit 1
import sys
sys.exit(0)
PY

# Start the worker
CMD ["python", "-m", "app.main"]
