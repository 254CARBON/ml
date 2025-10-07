# Dockerfile for Search Service
#
# Purpose:
# - Containerize the hybrid search service (semantic + lexical).
# - Python 3.11 slim base with minimal system packages.
#
# Practices:
# - Install only what is needed; keep image small.
# - Run as non-root; expose port 9007.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies (use caching)
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
EXPOSE 9007

# Health check to ensure the service remains responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9007/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9007"]
