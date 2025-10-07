# Dockerfile for Model Serving Service
#
# Purpose:
# - Package the REST service that loads MLflow models and serves predictions.
# - Based on Python 3.11 slim with minimal system packages.
#
# Security & Ops:
# - Non-root runtime user is created.
# - Health check probes the /health endpoint.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies (caching-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared libraries
COPY ../libs ./libs

# Copy application code
COPY . .

# Create non-root user for safer runtime
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

# Expose port
EXPOSE 9005

# Health check: detect readiness and liveness via HTTP
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9005/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9005"]
