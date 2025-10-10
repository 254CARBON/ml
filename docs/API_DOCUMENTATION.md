# API Documentation

## Overview

The 254Carbon ML Platform provides REST APIs for model serving, embedding generation, and search operations. All APIs follow OpenAPI 3.0 specifications and support JSON request/response formats.

## Authentication

### JWT Authentication
All API endpoints require JWT authentication via the `Authorization` header:

```http
Authorization: Bearer <jwt_token>
```

### API Key Authentication
Service-to-service communication uses API keys:

```http
X-API-Key: <api_key>
```

## Service Setup & Workflows

### Local Setup Summary
1. Install dependencies and hooks:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   make install
   cp env.example .env
   ```
2. Start the full stack (Docker) or individual services:
   ```bash
   make docker-up        # all services
   make dev-model-serving
   make dev-embedding
   make dev-search
   ```
3. Smoke test health endpoints:
   ```bash
   curl http://localhost:9005/health   # Model Serving
   curl http://localhost:9006/health   # Embedding Service
   curl http://localhost:9007/health   # Search Service
   ```

### Model Serving Workflow
1. Train or reuse a model and promote it to Production:
   ```bash
   python scripts/promote_model.py --model curve_forecaster --stage Production
   ```
2. Issue an online prediction:
   ```bash
   curl -X POST http://localhost:9005/api/v1/predict \
     -H "Content-Type: application/json" \
     -d '{"inputs":[{"rate_1y":0.025,"rate_5y":0.03}]}'
   ```
3. Submit a batch job and monitor progress:
   ```bash
   curl -X POST http://localhost:9005/api/v1/batch \
     -H "Content-Type: application/json" \
     -d '{"inputs":[{"rate_1y":0.025,"rate_5y":0.03}]}'
   curl http://localhost:9005/api/v1/jobs/<job_id>
   ```

### Embedding Workflow
1. Generate embeddings on-demand:
   ```bash
   curl -X POST http://localhost:9006/api/v1/embed \
     -H "Content-Type: application/json" \
     -d '{"inputs":["UST 10Y benchmark commentary"]}'
   ```
2. Trigger bulk refresh via event publication:
   ```bash
   python scripts/reindex_all.py --entity instruments --batch-size 256
   ```
3. Monitor metrics and models:
   ```bash
   curl http://localhost:9006/metrics
   curl http://localhost:9006/api/v1/models
   ```

### Search Workflow
1. Index or update searchable content:
   ```bash
   curl -X POST http://localhost:9007/api/v1/index \
     -H "Content-Type: application/json" \
     -d '{"entity_type":"instrument","entity_id":"UST-10Y","text":"UST 10Y benchmark curve commentary"}'
   ```
2. Run a hybrid search query:
   ```bash
   curl -X POST http://localhost:9007/api/v1/search \
     -H "Content-Type: application/json" \
     -d '{"query":"front-end curve steepener","semantic":true,"limit":5}'
   ```
3. Remove outdated entities when needed:
   ```bash
   curl -X DELETE http://localhost:9007/api/v1/index/instrument/UST-10Y
   ```

## Model Serving API

### Base URL
`http://localhost:9005/api/v1` (development)
`https://api.254carbon.com/ml/model-serving/v1` (production)

### Endpoints

#### Predict (Synchronous)
Make real-time predictions using production models.

```http
POST /predict
Content-Type: application/json
Authorization: Bearer <token>

{
  "inputs": [
    {
      "rate_0.25y": 0.02,
      "rate_1y": 0.025,
      "rate_5y": 0.03,
      "rate_10y": 0.035,
      "vix": 20.0,
      "fed_funds": 0.02
    }
  ],
  "model_name": "curve_forecaster",
  "model_version": "1.2.0"
}
```

**Response:**
```json
{
  "predictions": [
    {
      "forecast_horizons": {
        "h1": 0.0205,
        "h2": 0.0210,
        "h3": 0.0215,
        "h4": 0.0220,
        "h5": 0.0225
      },
      "metadata": {
        "model_name": "curve_forecaster",
        "model_type": "ensemble",
        "prediction_timestamp": "2024-01-01T12:00:00Z"
      }
    }
  ],
  "model_name": "curve_forecaster",
  "model_version": "1.2.0",
  "latency_ms": 45.2
}
```

#### Batch Predict (Asynchronous)
Submit batch prediction jobs for processing.

```http
POST /batch
Content-Type: application/json

{
  "inputs": [
    {"rate_1y": 0.025, "rate_5y": 0.03},
    {"rate_1y": 0.026, "rate_5y": 0.031}
  ],
  "model_name": "curve_forecaster"
}
```

**Response:**
```json
{
  "job_id": "batch_1704110400_1234",
  "status": "queued",
  "message": "Batch prediction job queued successfully"
}
```

#### List Models
Get information about available models.

```http
GET /models
```

**Response:**
```json
[
  {
    "name": "curve_forecaster",
    "version": "1.2.0",
    "stage": "Production",
    "created_at": "2024-01-01T00:00:00Z",
    "description": "Ensemble curve forecasting model"
  }
]
```

#### Get Batch Job Status
Check the status of a batch prediction job.

```http
GET /jobs/{job_id}
```

**Response:**
```json
{
  "job_id": "batch_1704110400_1234",
  "status": "completed",
  "started_at": 1704110400,
  "completed_at": 1704110450,
  "duration_seconds": 50,
  "model_name": "curve_forecaster",
  "model_version": "1.2.0",
  "input_count": 2,
  "predictions": [...]
}
```

## Embedding Service API

### Base URL
`http://localhost:9006/api/v1` (development)

### Endpoints

#### Generate Embeddings
Generate vector embeddings for text inputs.

```http
POST /embed
Content-Type: application/json

{
  "items": [
    {
      "type": "instrument",
      "id": "NG_HH_BALMO",
      "text": "Natural gas Henry Hub balance of month physical forward"
    },
    {
      "type": "curve",
      "id": "USD_YIELD_CURVE",
      "text": "US Dollar yield curve government bonds"
    }
  ],
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

**Response:**
```json
{
  "model_version": "sentence-transformers/all-MiniLM-L6-v2",
  "vectors": [
    [0.013, -0.442, 0.123, ...],
    [-0.234, 0.567, -0.089, ...]
  ],
  "count": 2,
  "latency_ms": 150.5
}
```

#### Batch Embeddings
Submit batch embedding jobs.

```http
POST /batch
Content-Type: application/json

{
  "items": [
    {"type": "instrument", "id": "INST_1", "text": "..."},
    {"type": "instrument", "id": "INST_2", "text": "..."}
  ],
  "batch_size": 100
}
```

#### List Models
Get available embedding models.

```http
GET /models
```

**Response:**
```json
[
  {
    "name": "sentence-transformers/all-MiniLM-L6-v2",
    "version": "1.0.0",
    "dimension": 384,
    "max_length": 256
  }
]
```

## Search Service API

### Base URL
`http://localhost:9007/api/v1` (development)

### Endpoints

#### Search
Perform hybrid semantic and lexical search.

```http
POST /search
Content-Type: application/json

{
  "query": "henry hub gas curve",
  "semantic": true,
  "filters": {
    "type": ["instrument", "curve"],
    "tenant_id": "default",
    "tags": ["energy"]
  },
  "limit": 10,
  "similarity_threshold": 0.5
}
```

**Response:**
```json
{
  "results": [
    {
      "entity_type": "instrument",
      "entity_id": "NG_HH_BALMO",
      "score": 0.95,
      "metadata": {
        "text": "Natural gas Henry Hub balance of month",
        "tags": ["energy", "gas", "futures"],
        "fusion_details": {
          "semantic_score": 0.92,
          "lexical_score": 0.88,
          "rrf_score": 0.95,
          "fusion_algorithm": "rrf"
        }
      },
      "text": "Natural gas Henry Hub balance of month physical forward"
    }
  ],
  "total": 1,
  "query": "henry hub gas curve",
  "semantic": true,
  "latency_ms": 245.8
}
```

#### Index Entity
Add or update an entity in the search index.

```http
POST /index
Content-Type: application/json

{
  "entity_type": "instrument",
  "entity_id": "NG_HH_BALMO",
  "text": "Natural gas Henry Hub balance of month physical forward",
  "metadata": {
    "sector": "energy",
    "commodity": "natural_gas",
    "exchange": "NYMEX"
  },
  "tags": ["energy", "gas", "futures"]
}
```

#### Remove Entity
Remove an entity from the search index.

```http
DELETE /index/{entity_type}/{entity_id}
```

#### Index Statistics
Get search index statistics.

```http
GET /index/stats
```

**Response:**
```json
{
  "total_entities": 10000,
  "by_entity_type": {
    "instrument": 5000,
    "curve": 3000,
    "scenario": 2000
  },
  "by_tenant": {
    "default": 8000,
    "tenant_1": 2000
  }
}
```

## Error Handling

### Error Response Format
All APIs return errors in a consistent format:

```json
{
  "error": "ValidationError",
  "detail": "Invalid input format",
  "code": "INVALID_INPUT",
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_123456"
}
```

### HTTP Status Codes
- `200 OK`: Successful request
- `400 Bad Request`: Invalid input or parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Common Error Codes
- `INVALID_INPUT`: Request validation failed
- `MODEL_NOT_FOUND`: Specified model not available
- `EMBEDDING_FAILED`: Embedding generation failed
- `SEARCH_FAILED`: Search operation failed
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `AUTHENTICATION_FAILED`: Invalid credentials
- `INTERNAL_ERROR`: Unexpected server error

## Rate Limiting

### Default Limits
- **Per User**: 60 requests/minute, 1000 requests/hour
- **Per Service**: 1000 requests/minute, 10000 requests/hour
- **Batch Operations**: 10 concurrent jobs per user

### Rate Limit Headers
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1704110460
```

## Pagination

### Query Parameters
- `limit`: Maximum number of results (default: 10, max: 100)
- `offset`: Number of results to skip (default: 0)
- `cursor`: Cursor-based pagination token

### Response Format
```json
{
  "results": [...],
  "pagination": {
    "limit": 10,
    "offset": 0,
    "total": 150,
    "has_more": true,
    "next_cursor": "eyJ0aW1lc3RhbXAiOjE3MDQxMTA0MDB9"
  }
}
```

## Webhooks

### Model Promotion Webhook
Receive notifications when models are promoted.

```http
POST /webhooks/model-promoted
Content-Type: application/json

{
  "event_type": "ml.model.promoted.v1",
  "model_name": "curve_forecaster",
  "version": "1.2.0",
  "stage": "Production",
  "timestamp": 1704110400000,
  "run_id": "abc123def456"
}
```

### Search Index Update Webhook
Receive notifications when search indexes are updated.

```http
POST /webhooks/index-updated
Content-Type: application/json

{
  "event_type": "search.index.updated.v1",
  "entity_type": "instruments",
  "count": 500,
  "timestamp": 1704110400000
}
```

## SDK and Client Libraries

### Python SDK
```python
from ml_platform_sdk import MLPlatformClient

# Initialize client
client = MLPlatformClient(
    base_url="https://api.254carbon.com/ml",
    api_key="your_api_key"
)

# Make predictions
predictions = client.model_serving.predict(
    inputs=[{"rate_1y": 0.025, "rate_5y": 0.03}],
    model_name="curve_forecaster"
)

# Generate embeddings
embeddings = client.embedding.embed(
    items=[{"text": "Natural gas futures", "type": "instrument"}]
)

# Search entities
results = client.search.search(
    query="energy futures",
    semantic=True,
    limit=10
)
```

### cURL Examples
```bash
# Model prediction
curl -X POST "http://localhost:9005/api/v1/predict" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{"rate_1y": 0.025, "rate_5y": 0.03}]
  }'

# Generate embeddings
curl -X POST "http://localhost:9006/api/v1/embed" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [{"text": "Natural gas futures", "type": "instrument"}]
  }'

# Search
curl -X POST "http://localhost:9007/api/v1/search" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "energy futures",
    "semantic": true,
    "limit": 10
  }'
```

## Testing the APIs

### Health Checks
```bash
# Check all services
curl http://localhost:9005/health  # Model Serving
curl http://localhost:9006/health  # Embedding Service
curl http://localhost:9007/health  # Search Service
curl http://localhost:5000/health  # MLflow Server
```

### Integration Test
```bash
# Complete workflow test
pytest tests/integration/test_ml_pipeline.py::TestMLPipeline::test_end_to_end_workflow -v
```

### Load Testing
```bash
# Run load tests
locust -f tests/load/locustfile.py --host=http://localhost:9005 \
       --users=50 --spawn-rate=5 --run-time=300s
```

## Monitoring and Observability

### Metrics Endpoints
All services expose Prometheus metrics:

```http
GET /metrics
```

### Distributed Tracing
All requests include tracing headers for end-to-end observability:

```http
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
```

### Logging
All services use structured JSON logging:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "service": "model-serving",
  "operation": "predict",
  "model_name": "curve_forecaster",
  "latency_ms": 45.2,
  "status": "success"
}
```

## Best Practices

### Request Optimization
- Batch requests when possible to reduce overhead
- Use appropriate timeout values for long-running operations
- Implement retry logic with exponential backoff
- Cache frequently requested data

### Error Handling
- Always check HTTP status codes
- Parse error responses for detailed information
- Implement proper retry logic for transient errors
- Log errors with sufficient context for debugging

### Security
- Never log sensitive data (API keys, tokens, PII)
- Validate all inputs on the client side
- Use HTTPS in production
- Rotate API keys regularly

### Performance
- Monitor API latency and throughput
- Use connection pooling for high-volume applications
- Implement client-side caching where appropriate
- Consider async/await patterns for concurrent requests
