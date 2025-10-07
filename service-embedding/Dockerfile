# Dockerfile for Embedding Service
#
# Purpose:
# - Build a lightweight container image for the embedding service.
# - Uses Python 3.11 slim base to reduce footprint.
#
# Notes:
# - System build tools are installed for scientific libs (numpy/torch).
# - A non-root user is created for runtime safety.
# - Health check validates the service's /health endpoint.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared libraries
COPY ../libs ./libs

# Copy application code
COPY . .

# Create non-root user for better container security
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

# Expose port
EXPOSE 9006

# Health check: fail the container if the service is not responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9006/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9006"]
