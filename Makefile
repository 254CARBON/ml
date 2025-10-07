# Makefile for local development and CI targets
#
# Common tasks:
# - `make dev`      : start the Dockerized dev stack
# - `make test`     : run unit tests + coverage
# - `make lint`     : run style/type checks
# - `make build`    : build all service images

.PHONY: help install dev build test test-contract lint clean docker-build docker-up docker-down

# Default target
help:
	@echo "254Carbon ML Platform - Available Commands:"
	@echo "  install     - Install Python dependencies"
	@echo "  dev         - Start development environment"
	@echo "  build       - Build all services"
	@echo "  test        - Run all tests"
	@echo "  test-contract - Run contract & integration tests in Docker"
	@echo "  lint        - Run linting and formatting"
	@echo "  clean       - Clean build artifacts"
	@echo "  docker-up   - Start all services with Docker Compose"
	@echo "  docker-down - Stop all services"
	@echo "  docker-build - Build all Docker images"

# Install dependencies
install:
	pip install -r requirements.txt
	pre-commit install

# Development environment
dev: docker-up
	@echo "Development environment started. Services available at:"
	@echo "  MLflow UI: http://localhost:5000"
	@echo "  Model Serving: http://localhost:9005"
	@echo "  Embedding Service: http://localhost:9006"
	@echo "  Search Service: http://localhost:9007"
	@echo "  Jaeger UI: http://localhost:16686"

# Build all services
build:
	@echo "Building all services..."
	@for service in service-*; do \
		if [ -f "$$service/Dockerfile" ]; then \
			echo "Building $$service..."; \
			docker build -t 254carbon/$$service:latest $$service/; \
		fi; \
	done

# Run tests
test:
	pytest tests/ -v --cov=libs --cov=service-* --cov-report=html --cov-report=term

test-contract:
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit test-runner
	docker-compose -f docker-compose.test.yml down --volumes

# Linting and formatting
lint:
	black --check .
	flake8 .
	mypy libs/ service-*/

# Format code
format:
	black .
	isort .

# Clean build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/

# Docker operations
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

# Service-specific commands
dev-mlflow:
	cd service-mlflow && uvicorn app.main:app --reload --port 5000

dev-model-serving:
	cd service-model-serving && uvicorn app.main:app --reload --port 9005

dev-embedding:
	cd service-embedding && uvicorn app.main:app --reload --port 9006

dev-search:
	cd service-search && uvicorn app.main:app --reload --port 9007

# Utility commands
promote-model:
	python scripts/promote_model.py $(ARGS)

reindex-all:
	python scripts/reindex_all.py $(ARGS)

sync-specs:
	python scripts/sync_specs.py

# Multi-arch builds
buildx:
	docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/254carbon/$(SERVICE):$(VERSION) ./$(SERVICE)/

# Generate SBOM (future)
sbom:
	@echo "SBOM generation not yet implemented"

# Database operations
db-migrate:
	alembic upgrade head

db-reset:
	docker-compose down -v
	docker-compose up -d postgres
	sleep 10
	make db-migrate
