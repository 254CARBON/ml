# Dockerfile for MLflow Server
#
# Purpose:
# - Run an MLflow tracking server configured against Postgres + MinIO.
# - Ships a minimal Python 3.11 base with MLflow dependencies.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for safer runtime
RUN useradd -m -u 1000 mlflow && chown -R mlflow:mlflow /app
USER mlflow

# Expose port
EXPOSE 5000

# Health check: validate HTTP health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start MLflow server
CMD ["python", "-m", "mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
