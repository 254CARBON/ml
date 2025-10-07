# 254Carbon ML Platform (`254carbon-ml`)

> Central repository for Machine Learning, Embeddings, Vector Search, and Model Operationalization within the 254Carbon platform.

This repository bundles early-phase ML + semantic capabilities to minimize operational overhead while enabling rapid iteration by a single developer plus AI coding agents. It will be split later when scaling or isolation requirements emerge (e.g., dedicated high‑throughput model serving or vector index sharding).

---

## Table of Contents
1. Purpose & Scope  
2. Included Services  
3. High-Level Architecture  
4. Repository Structure  
5. Service Manifests & Contracts  
6. ML Lifecycle & Workflows  
7. Embedding & Search Pipeline  
8. Model Serving Patterns  
9. Events & Contracts  
10. Data & Storage Layout  
11. Multi-Tenancy Strategy  
12. Configuration & Environment Variables  
13. Local Development (Multi-Arch + GPU Macs)  
14. Build & Image Strategy  
15. Deployment Flow  
16. Observability (Metrics, Traces, Logs)  
17. Security & Policy Considerations  
18. Performance Targets (Draft)  
19. Testing Strategy  
20. Dependency Boundaries  
21. Roadmap & Future Splits  
22. Contribution Workflow  
23. Troubleshooting  
24. Appendix: Example Flows  
25. License / Ownership  

---

## 1. Purpose & Scope

This repo standardizes:
- ML experiment tracking (MLflow 3.x)
- Model registry & promotion lifecycle
- Reusable model serving (REST/gRPC-style via FastAPI initially)
- Embedding generation (instrument metadata, curve summaries, analytics descriptors)
- Vector search & hybrid semantic retrieval
- Indexing orchestration & reindex jobs
- Event emission for model promotions, index refreshes
- Foundational feature preparation hooks (future Feature Store integration)

Not in scope (lives elsewhere):
- Raw ingestion & normalization (`254carbon-ingestion`, `data-processing`)
- Core domain analytics (backtesting, scenario engine)
- Gateway / streaming access layer responsibilities
- Policy enforcement (future OPA service)

---

## 2. Included Services

| Service | Purpose | Type | Scaling | Status |
|---------|---------|------|---------|--------|
| mlflow-server | Experiment tracking + artifact store | Stateful | Vertical + backups | Beta |
| model-serving | Serve “Production” models via REST (predict/batch) | Stateless | Horizontal | Alpha |
| embedding-service | Generate embeddings (batch + on-demand) | CPU/GPU aware | Horizontal (GPU preferred) | Alpha |
| search-service | Hybrid semantic + lexical search aggregator | Stateless | Horizontal | Alpha |
| indexer-worker | Async pipeline to (re)index entities | Worker (queue-based) | Horizontal | Planned |
| vector-store-adapter | Storage abstraction (PgVector → OpenSearch migration path) | Library/internal svc | N/A | Alpha |

---

## 3. High-Level Architecture

```
                    ┌──────────────────┐
      Specs (contracts, events)        │ 254carbon-specs │
                    └────────┬─────────┘
                              │
                              ▼
                      ┌────────────┐
                      │ mlflow     │  <─ MinIO (artifacts)
                      │ server     │  <─ Postgres (tracking DB)
                      └─────┬──────┘
                            │ model promoted event
                            ▼
                   ┌──────────────────┐
                   │ model-serving    │──▶ REST /predict
                   │ (Production)     │
                   └──┬────────┬──────┘
      embeddings req  │        │ search queries
                      ▼        ▼
             ┌────────────┐  ┌──────────────┐
             │ embedding   │  │ search       │
             │ service     │  │ service      │
             └────┬────────┘  └──────┬───────┘
                  │ embeddings        │ query orchestration
                  ▼                   ▼
         ┌─────────────────┐   ┌──────────────┐
         │ vector store    │   │ metadata DB  │
         │ (PgVector now,  │   │ (Postgres)   │
         │  OpenSearch later)  └──────────────┘
         └─────────────────┘
```

---

## 4. Repository Structure

```
/
  service-mlflow/
    Dockerfile
    config/
    service-manifest.yaml
  service-model-serving/
    app/
      api/
      runtime/
      loaders/
      adapters/
    tests/
    service-manifest.yaml
  service-embedding/
    app/
      encoders/
      batching/
      pipelines/
    models/
    tests/
    service-manifest.yaml
  service-search/
    app/
      routes/
      hybrid/
      ranking/
      retrievers/
    tests/
    service-manifest.yaml
  libs/
    vector_store/
    common/
  scripts/
    build_all.sh
    sync_specs.py
    promote_model.py
    reindex_all.py
  specs.lock.json
  .agent/
    context.yaml
  Makefile
  CHANGELOG.md
  README.md
  docs/
    EMBEDDING_STRATEGY.md
    MODEL_LIFECYCLE.md
    SEARCH_DESIGN.md
    VECTOR_STORE_MIGRATION.md
```

---

## 5. Service Manifests & Contracts

Each service has a `service-manifest.yaml` used by meta automation:

```yaml
service_name: model-serving
domain: ml
runtime: python
api_contracts:
  - model-serving-api@0.1.0
events_in:
  - ml.model.promoted.v1
events_out:
  - ml.inference.usage.v1
dependencies:
  internal: []
  external: [mlflow, minio, postgres, redis]
maturity: alpha
sla:
  p95_latency_ms: 120
  availability: "99.0%"
owner: ml
```

Contracts pinned via `specs.lock.json` (synced from `254carbon-specs`).

---

## 6. ML Lifecycle & Workflows

### Overview

1. Data prep (external pipelines) produces features & training datasets.
2. Training scripts log parameters, metrics, artifacts, and optional lineage metadata into MLflow.
3. Runs complete → candidate models are staged in MLflow.
4. Evaluation (batch/offline) decides promotion criteria.
5. Promotion emits `ml.model.promoted.v1`; model-serving reloads on the event (or scheduled poll).
6. Optional A/B or shadow deployments shape traffic across model versions.
7. Rollback reverts the MLflow stage tag and invalidates runtime caches.

Artifacts land in MinIO at `s3://mlflow-artifacts/{experiment}/{run_id}/`.

### Workflow: Train → Promote → Serve (local happy path)

1. Bootstrap the environment (see Section 13) and ensure `ML_ENV=local` in `.env`.
2. Start the stack: `make docker-up` (or `make dev` for full set of services).
3. Run or iterate on a training script, e.g.:
   ```bash
   cd training/curve_forecaster
   python train.py --curve-type yield --model-type ensemble
   ```
4. Review the run in MLflow (http://localhost:5000) and, when satisfied, promote the artifact:
   ```bash
   python ../../scripts/promote_model.py --model curve_forecaster --stage Production
   ```
5. Confirm the promotion via the model-serving API:
   ```bash
   curl -X POST http://localhost:9005/predict \
     -H "Content-Type: application/json" \
     -d '{"inputs":[{"curve_id":"NG_BALMO"}]}'
   ```
   A `200` with model metadata indicates the promotion event propagated end-to-end.

### Workflow: Embedding Refresh & Backfill

1. Keep the stack running (`make docker-up`); verify the embedding service health at `http://localhost:9006/health`.
2. Trigger reindex events per entity type (examples: `instruments`, `curves`, `scenarios`):
   ```bash
   python scripts/reindex_all.py --entity instruments --batch-size 256 --model-version latest
   ```
3. Tail logs for progress: `docker-compose logs -f embedding-service` (or `make dev-embedding` if running outside Docker).
4. Spot-check vectors through the API workflow (see docs/API_DOCUMENTATION.md) to ensure new embeddings are persisted.

### Workflow: Search Warm-Up & Validation

1. Seed or update search metadata with the indexing endpoint (repeat per entity):
   ```bash
   curl -X POST http://localhost:9007/api/v1/index \
     -H "Content-Type: application/json" \
     -d '{"entity_type":"instrument","entity_id":"UST-10Y","text":"UST 10Y benchmark curve commentary","metadata":{"tenor":"10Y"}}'
   ```
2. Hit the search API to validate semantic + lexical fusion:
   ```bash
   curl -X POST http://localhost:9007/api/v1/search \
     -H "Content-Type: application/json" \
     -d '{"query":"front-end curve steepener","semantic":true,"limit":5}'
   ```
3. Use the response metadata to compare vector vs. lexical contributions before approving the release. `search.index.updated.v1` events can be monitored to verify downstream consumers observed the refresh.

---

## 7. Embedding & Search Pipeline

### Supported Entities
- Instruments
- Curves (metadata + statistical signature)
- Scenario summaries
- Backtest descriptions (short form)
- Freeform domain documents (future)

### Embedding Generation Modes
| Mode | Trigger | Storage |
|------|---------|---------|
| Batch | Reindex job | Vector store + metadata |
| On-demand | Missing vector at query time | Cache + vector store write-through |
| Refresh | Model upgrade or drift threshold | Background job |

### Hybrid Retrieval Process
1. Accept query: `{ text, filters?, semantic=true|false }`
2. If semantic:
   - Generate query embedding
   - Vector similarity search (top K)
   - Lexical search (Postgres full text or OpenSearch phase 2)
   - Fuse ranks (reciprocal rank fusion or weighted)
3. Apply filters (tenant_id, type, tags)
4. Return normalized response with scoring breakdown (optional debug).

---

## 8. Model Serving Patterns

### Endpoint Layout
| Path | Method | Purpose |
|------|--------|---------|
| `/predict` | POST | Synchronous inference |
| `/batch` | POST | Async batch job (returns job_id) |
| `/models` | GET | List deployed model versions |
| `/health` | GET | Liveness/readiness |
| `/experiments` | GET/POST | A/B testing experiments (feature flag) |
| `/shadow` | GET/POST | Shadow deployment controls (feature flag) |
| `/metrics` | GET | Prometheus metrics |
| `/reload` | POST (internal) | Force reload (if event missed) |

### Inference Flow
1. Request validated against OpenAPI schema
2. Input preprocessor (if bundling transformation)
3. Model inference (CPU/GPU):
   - CPU default
   - GPU label scheduling (ARM Macs w/ accelerator)
4. Postprocessing (probabilities → decision / embedding)
5. Metrics posted:
   - latency
   - model_version
   - request_size
6. Response returned

### Caching
- Embedding model warm caches (tokenizer, weights).
- If deterministic & identical payload hash: optional short-lived result reuse.

---

## 9. Events & Contracts

| Event | Producer | Consumer | Purpose |
|-------|----------|----------|---------|
| `ml.model.promoted.v1` | Out-of-band admin / CI | model-serving | Deploy new model |
| `ml.inference.usage.v1` | model-serving | Metrics / billing (future) | Track model usage |
| `ml.embedding.reindex.request.v1` | admin ops | embedding-service | Rebuild vectors |
| `search.index.updated.v1` | search-service | Observability / gateway (optional) | Notify index freshness |
| `ml.embedding.generated.v1` | embedding-service | search-service | Trace async generation (future) |

All schemas live in `254carbon-specs/events/avro/ml/...`.

---

## 10. Data & Storage Layout

| Data Type | Store | Notes |
|-----------|-------|-------|
| MLflow tracking | PostgreSQL | DB schema versioned by MLflow |
| Artifacts | MinIO | Versioned bucket |
| Vector embeddings | PgVector table (initial) | Table: `embeddings(entity_type, entity_id, vector, meta, model_version, tenant_id)` |
| Search metadata | Postgres JSONB | `search_items` table: id, type, text, tags, updated_at |
| Future large-scale | OpenSearch | Phase 2 swap-in (synced from PgVector) |
| Caches | Redis | Embedding generation throttle, index warm state |
| Model runtime cache | In memory | Loaded weight handles |

---

## 11. Multi-Tenancy Strategy

Soft isolation:
- All records include `tenant_id`
- Embedding + search queries filter by tenant scope
- A model may be globally shared; per-tenant overrides supported via metadata layering (future)
- No per-tenant model forks initially (cost & complexity deferment)

---

## 12. Configuration & Environment Variables

Common prefix: `ML_`

| Variable | Description | Example |
|----------|-------------|---------|
| ML_ENV | Environment name | local |
| ML_MLFLOW_BACKEND_DSN | Tracking DB DSN | postgres://... |
| ML_MLFLOW_ARTIFACT_URI | Artifact root | s3://mlflow-artifacts |
| ML_MINIO_ENDPOINT | MinIO host | http://minio:9000 |
| ML_VECTOR_DB_DSN | PgVector/Postgres DSN | postgres://... |
| ML_MODEL_DEFAULT_NAME | Default model name | curve_forecaster |
| ML_EMBEDDING_MODEL | HF/local model id | sentence-transformers/all-MiniLM-L6-v2 |
| ML_MAX_BATCH_SIZE | Inference batch limit | 256 |
| ML_INGEST_QUEUE | Reindex queue name | embeddings_rebuild |
| ML_TRACING_ENABLED | OTel toggle | true |
| ML_OTEL_EXPORTER | Collector URL | http://otel:4318 |
| ML_GPU_PREFERENCE | Allowed values | auto |
| ML_AUTH_ENABLED | Enable JWT authentication | false |
| ML_AB_TESTING_ENABLED | Enable A/B testing features | false |
| ML_VECTOR_BACKEND | Vector store backend (pgvector/opensearch) | pgvector |

Per-service config under `service-*/config/`.

---

## 13. Local Development (Multi-Arch + GPU Macs)

### Prerequisites
- Python 3.9+ (use `pyenv` or `uv` to pin versions if juggling projects)
- Docker Desktop with Compose V2 and BuildKit enabled
- Optional: Apple Silicon GPU access (set `accelerator` resources in Docker Desktop → Features in development)
- `make` and `pre-commit` installed locally

### Bootstrap (once per machine)
```bash
python -m venv .venv
source .venv/bin/activate
make install
cp env.example .env
```
- Tailor `.env` for local DSNs, credentials, and feature toggles.
- Install the git hooks with `pre-commit install` (done automatically by `make install`).

### Start & Stop the Stack
- Full platform: `make docker-up` (background) or `make dev` (with friendly service summary).
- Stop everything: `make docker-down` (add `-v` for a clean slate).
- Service-only loops (hot reload):

| Service | Command | Local URL |
|---------|---------|-----------|
| MLflow | `make dev-mlflow` | http://localhost:5000 |
| Model Serving | `make dev-model-serving` | http://localhost:9005 |
| Embedding Service | `make dev-embedding` | http://localhost:9006 |
| Search Service | `make dev-search` | http://localhost:9007 |

### Common Local Workflows
1. **Happy-path smoke test**
   ```bash
   python scripts/promote_model.py --model curve_forecaster --stage Production
   curl -X POST http://localhost:9005/predict \
     -H "Content-Type: application/json" \
     -d '{"inputs":[{"curve_id":"NG_BALMO"}]}'
   ```
2. **Embedding sanity check**
   ```bash
   curl -X POST http://localhost:9006/api/v1/embed \
     -H "Content-Type: application/json" \
     -d '{"inputs":["3m curve steepener commentary"]}'
   ```
3. **Run tests & lint**
   ```bash
   make test
   make lint
   ```

### Multi-Arch / GPU Notes
- Docker Desktop on Apple Silicon exposes GPUs through `--device /dev/dri` (enable via Docker settings); the embedding service falls back to CPU when GPU acceleration is unavailable.
- For Kubernetes-based Mac GPU scheduling use node labels `arch=arm64` and `accelerator=gpu`; deployments already include matching tolerations.
- Prefer pure-Python or pre-built wheel models to avoid local compilation across architectures.

---

## 14. Build & Image Strategy

Multi-arch build (example):
```
make buildx SERVICE=embedding-service VERSION=0.2.0
```

Images tagged:
- `ghcr.io/254carbon/model-serving:0.2.0`
- Additional tags: `:0.2`, `:0`, `:sha-<short>`

SBOM generation (future):
```
make sbom SERVICE=model-serving
```

---

## 15. Deployment Flow

1. Update or add model training script (external or here under `/training` if added later)
2. Run training: logs to MLflow
3. Promote via script or MLflow UI
4. Event or poll triggers model-serving reload
5. Embedding service reindex (if embedding model changed)
6. Search service warms updated hybrid indexes

Canary Deploy (future):
- Deploy new model-serving instance w/ shadow traffic
- Compare metrics (latency, error, drift)

---

## 16. Observability (Metrics, Traces, Logs)

| Service | Metrics Examples |
|---------|------------------|
| model-serving | `inference_latency_ms`, `inference_requests_total`, `model_version_loaded` |
| embedding-service | `embedding_gen_latency_ms`, `embedding_failures_total`, `batch_size_histogram` |
| search-service | `search_query_latency_ms`, `hybrid_merge_time_ms`, `vector_hits_per_query` |
| mlflow-server | Proxy metrics (requests, artifact IO) |
| indexer-worker | `reindex_jobs_total`, `reindex_duration_seconds`, `reindex_failures_total` |

Tracing:
- Each inference request includes traceparent header (if upstream present)
- Embedding generation spans include model name/size

Logging Format (JSON):
```
{"ts":"...","service":"model-serving","model":"curve_forecaster","version":"1.2.0","latency_ms":42,"status":"ok"}
```

---

## 17. Security & Policy Considerations

- JWT validation occurs upstream (gateway) but internal verification optional for direct calls.
- Optional JWT validation via `ML_AUTH_ENABLED` environment variable.
- Multi-tenant support with optional `tenant_id` parameter in search/index APIs.
- No PII stored; still treat run metadata as internal.
- Model artifact integrity (future):
  - Hash recorded in MLflow
  - Verify hash at load
- Access roles (future):
  - `ml_admin`: promotions
  - `ml_viewer`: list runs / models
  - `ml_ops`: reindex triggers

Secrets:
- Managed via env variables or K8s Secrets; NEVER commit credentials.

---

## 18. Performance Targets (Draft)

| Target | Value |
|--------|-------|
| Single inference P95 latency | < 120 ms |
| Batch embedding 512 items P95 | < 3 s (CPU), < 1.5 s (GPU) |
| Search hybrid query P95 | < 400 ms |
| Model reload time (warm) | < 10 s |
| Throughput (inference) per pod | 150 req/s (baseline) |

Measure & refine after synthetic load tests.

---

## 19. Testing Strategy

| Layer | Type | Tool |
|-------|------|------|
| Unit | Pure logic (tokenizers, vector ops) | pytest |
| Integration | MLflow interaction, vector DB ops | dockerized Postgres |
| Contract | OpenAPI schema match | openapi-diff |
| Performance | Lightweight k6/Locust harness (future) | to add |
| Reproducibility | Hash check seeds (training scripts) | custom assertions |
| Drift detection (future) | Statistical monitors | custom jobs |

Run:
```
make test
```

For deterministic contract + integration coverage (spins up Postgres/Redis/MinIO/ML services via Docker Compose and resets state before each run):
```
make test-contract
```

---

## 20. Dependency Boundaries

| Internal Allowed | External Calls |
|------------------|----------------|
| shared libs (vector_store adapter) | MLflow backend, MinIO, Postgres, Redis |
| No direct ingestion pipelines | Only via event payloads |
| No entitlement logic | Authorization resolved upstream |
| No gateway logic (routing) | Only API contract implementation |

Avoid coupling to:
- Gateway internal modules
- Normalization or analytics internals

---

## 21. Roadmap & Future Splits

| Milestone | Description | Split Trigger |
|-----------|-------------|---------------|
| M1 | Stable model-serving + embedding batch | N/A |
| M2 | Add vector store migration (PgVector → OpenSearch) | Query volume > threshold |
| M3 | Feature Store integration (Feast or custom) | Need real-time feature pipelines |
| M4 | Canary & A/B experiments | Traffic & multi-model evaluation |
| M5 | Drift monitoring & auto-retrain hooks | Data distribution shift |
| M6 | Split model-serving to its own repo | Independent scaling / release cadence |
| M7 | GPU job queue + asynchronous micro-batching | Higher throughput requirement |

---

## 22. Contribution Workflow

1. Sync contracts: `make spec-sync`
2. Add or modify code (embedding pipeline, search ranking, etc.)
3. If changing APIs:
   - Update spec in `254carbon-specs`
   - Bump version (MINOR or MAJOR)
4. Add tests
5. `make lint test`
6. Commit with conventional prefix:  
   `feat(search): hybrid rank fusion weights adjustable`
7. Open PR → CI validates:
   - Lint / tests
   - Specs lock diff
   - Manifests schema
   - (Future) SBOM / security scan

---

## 23. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Model not reloading | Missed promotion event | POST /reload or check event bus |
| Slow embeddings | CPU fallback | Check GPU node labels & logs |
| Search returns nothing | Empty vector index | Run `python scripts/reindex_all.py` |
| 500 on /predict | Missing model artifact | Validate MLflow artifact path & permissions |
| High latency spikes | Cold start / large model | Preload on startup or use warm-reload strategy |
| Vector mismatch errors | Schema drift (dimension) | Rebuild all embeddings with new model version |

---

## 24. Appendix: Example Flows

### Model Promotion Script
```
python scripts/promote_model.py \
  --model curve_forecaster \
  --run-id 1ab23def456 \
  --stage Production
```
Emits `ml.model.promoted.v1` → model-serving reloads.

### Reindex All Entities
```
python scripts/reindex_all.py --entity instruments --batch-size 500
```

### Embedding API (Internal)
```
POST /embed
{
  "items": [
    {"type":"instrument","id":"NG_HH_BALMO","text":"Natural gas Henry Hub balance of month physical forward"}
  ],
  "model":"instrument_embedder_v1"
}
```

Response:
```
{
  "model_version":"instrument_embedder_v1",
  "vectors":[[0.013, -0.442, ...]],
  "count":1
}
```

### Search API
```
POST /search
{
  "query":"henry hub gas curve",
  "semantic": true,
  "filters": {"type":["instrument","curve"],"tenant_id":"default"},
  "limit": 10
}
```

---

## 25. License / Ownership

Internal use. Ownership: ML / Platform Engineering (single dev + AI agents).  
Public exposure of APIs may occur when external developer integration begins.

---

## Quick Start

```bash
# Install dev deps
make install

# Run model-serving locally
cd service-model-serving
uvicorn app.main:app --reload --port 9005

# Promote a model (example)
python ../scripts/promote_model.py --model curve_forecaster --run-id TEST123 --stage Production

# Generate embeddings (dev)
cd ../service-embedding
python -m app.cli.embed --input samples/instruments.json
```

---

> “ML infrastructure should amplify experimentation velocity while enforcing disciplined, observable, reproducible execution.”

---
