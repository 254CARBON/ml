# Embedding Strategy

## Overview

The 254Carbon ML Platform uses embeddings to enable semantic search and similarity matching across various entity types including instruments, curves, scenarios, and backtest descriptions.

## Supported Entity Types

- **Instruments**: Financial instruments with metadata and descriptions
- **Curves**: Yield curves, volatility curves, and other financial curves
- **Scenarios**: Market scenarios and stress test scenarios
- **Backtests**: Backtest descriptions and results
- **Documents**: Freeform domain documents (future)

## Embedding Models

### Default Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **Max Length**: 256 tokens
- **Language**: English
- **Performance**: Balanced speed and quality

### Model Selection Criteria
- **Speed**: Fast inference for real-time search
- **Quality**: Good semantic understanding
- **Size**: Reasonable memory footprint
- **Compatibility**: Works well with financial domain text

## Embedding Generation Modes

### 1. Batch Processing
- **Trigger**: Reindex jobs, bulk operations
- **Storage**: Vector store + metadata database
- **Performance**: Optimized for throughput
- **Use Case**: Initial indexing, model updates

### 2. On-Demand Generation
- **Trigger**: Missing vector at query time
- **Storage**: Cache + vector store write-through
- **Performance**: Optimized for latency
- **Use Case**: Real-time search, new entities

### 3. Refresh Operations
- **Trigger**: Model upgrade or drift threshold
- **Storage**: Background job with progress tracking
- **Performance**: Non-blocking, incremental
- **Use Case**: Model updates, data quality improvements

## Vector Storage

### Current Implementation: PgVector
- **Database**: PostgreSQL with pgvector extension
- **Index**: IVFFlat with cosine similarity
- **Partitioning**: By entity type and tenant
- **Backup**: Automated with database backups

### Future Migration: OpenSearch
- **Reason**: Better scalability for large vector collections
- **Migration Path**: Gradual migration with dual-write
- **Compatibility**: Same embedding dimensions and similarity metrics

## Search Integration

### Hybrid Search
- **Semantic**: Vector similarity search
- **Lexical**: PostgreSQL full-text search
- **Fusion**: Reciprocal Rank Fusion (RRF)
- **Filtering**: Multi-tenant and entity type filters

### Performance Targets
- **Single Query**: < 400ms P95 latency
- **Batch Processing**: < 3s for 512 items (CPU), < 1.5s (GPU)
- **Throughput**: 100 queries/second per instance

## Multi-Tenancy

### Data Isolation
- **Tenant ID**: All vectors include tenant_id
- **Query Filtering**: Automatic tenant scope in queries
- **Access Control**: Tenant-level permissions

### Model Sharing
- **Global Models**: Shared across tenants
- **Tenant Overrides**: Per-tenant model versions (future)
- **Cost Optimization**: Avoid per-tenant model forks

## Monitoring and Observability

### Metrics
- **Generation**: Latency, throughput, error rates
- **Storage**: Vector count, storage usage
- **Search**: Query latency, result quality
- **Model**: Version, performance, drift

### Alerts
- **High Latency**: > 500ms average
- **Error Rate**: > 5% failure rate
- **Storage**: > 80% capacity
- **Model Drift**: Significant performance degradation

## Best Practices

### Text Preprocessing
- **Normalization**: Standardize financial terms
- **Cleaning**: Remove noise, standardize format
- **Chunking**: Handle long documents appropriately

### Model Management
- **Versioning**: Track model versions and performance
- **A/B Testing**: Compare model performance
- **Rollback**: Quick revert to previous model

### Performance Optimization
- **Caching**: Cache frequently accessed vectors
- **Batching**: Process multiple items together
- **GPU Acceleration**: Use GPU when available
- **Connection Pooling**: Efficient database connections

## Future Enhancements

### Model Improvements
- **Domain-Specific**: Financial domain fine-tuning
- **Multi-Language**: Support for multiple languages
- **Multimodal**: Support for charts and images

### Infrastructure
- **Distributed**: Multi-node vector search
- **Real-time**: Streaming embedding updates
- **Advanced Indexing**: HNSW, IVF-PQ indexes

### Analytics
- **Usage Analytics**: Track embedding usage patterns
- **Quality Metrics**: Measure embedding quality
- **Cost Optimization**: Optimize resource usage
