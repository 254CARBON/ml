#!/bin/bash
## Build all services for the ML Platform
# Purpose: Convenience wrapper to build local Docker images for every service
# Prereqs: Docker daemon running; sufficient disk space
# Usage: ./scripts/build_all.sh
# Notes:
# - Tags images with :latest; adjust to include versions when needed
# - Intended for local dev; CI/CD pipelines should build reproducibly

set -e

echo "Building all ML platform services..."

cd ..

# Build model serving service
echo "Building model serving service..."
cd service-model-serving
docker build -t 254carbon/model-serving:latest .
cd ..

# Build embedding service
echo "Building embedding service..."
cd service-embedding
docker build -t 254carbon/embedding-service:latest .
cd ..

# Build search service
echo "Building search service..."
cd service-search
docker build -t 254carbon/search-service:latest .
cd ..

# Build indexer worker
echo "Building indexer worker..."
cd service-indexer-worker
docker build -t 254carbon/indexer-worker:latest .
cd ..

echo "All services built successfully!"

# List built images
echo "Built images:"
docker images | grep 254carbon
