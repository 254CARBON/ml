# Search Design

## Overview

The 254Carbon ML Platform implements a hybrid search system that combines semantic vector search with traditional lexical search to provide comprehensive search capabilities across financial instruments, curves, scenarios, and other domain entities.

## Architecture

### Components
- **Search Service**: Main search API and orchestration
- **Embedding Service**: Vector embedding generation
- **Vector Store**: PgVector for similarity search
- **Metadata Database**: PostgreSQL for lexical search
- **Cache Layer**: Redis for performance optimization

### Data Flow
```
Query → Search Service → [Semantic Search] → Vector Store
                      → [Lexical Search] → Metadata DB
                      → [Result Fusion] → Ranked Results
```

## Search Types

### 1. Semantic Search
- **Technology**: Vector similarity search
- **Use Case**: Conceptual similarity, fuzzy matching
- **Performance**: < 200ms P95 latency
- **Accuracy**: High for semantic understanding

### 2. Lexical Search
- **Technology**: PostgreSQL full-text search
- **Use Case**: Exact matches, keyword search
- **Performance**: < 100ms P95 latency
- **Accuracy**: High for exact matches

### 3. Hybrid Search
- **Technology**: Reciprocal Rank Fusion (RRF)
- **Use Case**: Best of both approaches
- **Performance**: < 400ms P95 latency
- **Accuracy**: Optimal for most queries

## Search Pipeline

### 1. Query Processing
- **Input Validation**: Validate query parameters
- **Query Expansion**: Expand query terms
- **Filtering**: Apply tenant and entity type filters
- **Caching**: Check query result cache

### 2. Semantic Search
- **Embedding Generation**: Generate query embedding
- **Vector Search**: Search similar vectors
- **Similarity Scoring**: Calculate similarity scores
- **Result Ranking**: Rank by similarity

### 3. Lexical Search
- **Text Processing**: Process query text
- **Full-Text Search**: PostgreSQL full-text search
- **Relevance Scoring**: Calculate relevance scores
- **Result Ranking**: Rank by relevance

### 4. Result Fusion
- **RRF Algorithm**: Combine search results
- **Score Normalization**: Normalize scores
- **Deduplication**: Remove duplicate results
- **Final Ranking**: Produce final ranked list

### 5. Result Formatting
- **Metadata Enrichment**: Add metadata to results
- **Score Explanation**: Provide score breakdown
- **Pagination**: Handle result pagination
- **Response Formatting**: Format final response

## Search Features

### Filtering
- **Entity Type**: Filter by instrument, curve, scenario
- **Tenant**: Multi-tenant data isolation
- **Tags**: Custom tag-based filtering
- **Date Range**: Temporal filtering
- **Custom Fields**: Domain-specific filters

### Ranking
- **Similarity Score**: Vector similarity score
- **Relevance Score**: Text relevance score
- **Combined Score**: RRF combined score
- **Boost Factors**: Custom boost factors
- **Decay Functions**: Time-based decay

### Pagination
- **Offset-Based**: Traditional offset pagination
- **Cursor-Based**: Cursor-based pagination
- **Limit Control**: Configurable result limits
- **Total Count**: Optional total count

## Performance Optimization

### Caching Strategy
- **Query Cache**: Cache frequent queries
- **Embedding Cache**: Cache query embeddings
- **Result Cache**: Cache search results
- **Metadata Cache**: Cache entity metadata

### Database Optimization
- **Indexes**: Optimized database indexes
- **Connection Pooling**: Efficient connection management
- **Query Optimization**: Optimized SQL queries
- **Partitioning**: Data partitioning strategies

### Vector Search Optimization
- **Index Type**: IVFFlat for cosine similarity
- **Index Parameters**: Optimized index parameters
- **Batch Processing**: Batch vector operations
- **GPU Acceleration**: GPU support for large datasets

## Search APIs

### Search Endpoint
```http
POST /api/v1/search
Content-Type: application/json

{
  "query": "henry hub gas curve",
  "semantic": true,
  "filters": {
    "type": ["instrument", "curve"],
    "tenant_id": "default"
  },
  "limit": 10,
  "similarity_threshold": 0.0
}
```

### Response Format
```json
{
  "results": [
    {
      "entity_type": "instrument",
      "entity_id": "NG_HH_BALMO",
      "score": 0.95,
      "metadata": {
        "text": "Natural gas Henry Hub balance of month",
        "tags": ["energy", "gas", "futures"]
      }
    }
  ],
  "total": 1,
  "query": "henry hub gas curve",
  "semantic": true,
  "latency_ms": 150
}
```

### Index Management
```http
POST /api/v1/index
Content-Type: application/json

{
  "entity_type": "instrument",
  "entity_id": "NG_HH_BALMO",
  "text": "Natural gas Henry Hub balance of month",
  "metadata": {
    "sector": "energy",
    "currency": "USD"
  },
  "tags": ["energy", "gas", "futures"]
}
```

## Multi-Tenancy

### Data Isolation
- **Tenant ID**: All queries include tenant_id
- **Row-Level Security**: Database-level isolation
- **Query Filtering**: Automatic tenant filtering
- **Access Control**: Tenant-based permissions

### Performance Considerations
- **Tenant Indexes**: Optimized indexes per tenant
- **Cache Partitioning**: Tenant-specific caches
- **Resource Allocation**: Fair resource allocation
- **Monitoring**: Tenant-specific metrics

## Monitoring and Observability

### Key Metrics
- **Query Latency**: P50, P95, P99 response times
- **Query Throughput**: Queries per second
- **Cache Hit Rate**: Cache effectiveness
- **Error Rate**: Failed queries percentage
- **Result Quality**: Relevance and accuracy

### Alerting Rules
- **High Latency**: P95 > 500ms
- **High Error Rate**: > 2% failures
- **Low Cache Hit Rate**: < 80% hit rate
- **Index Issues**: Vector index problems

### Dashboards
- **Performance**: Real-time performance metrics
- **Usage**: Query patterns and usage
- **Quality**: Search result quality
- **Infrastructure**: System health metrics

## Best Practices

### Query Design
- **Clear Intent**: Well-defined search intent
- **Appropriate Filters**: Use relevant filters
- **Limit Results**: Set reasonable limits
- **Error Handling**: Graceful error handling

### Index Management
- **Regular Updates**: Keep indexes current
- **Bulk Operations**: Use bulk indexing
- **Index Optimization**: Regular index maintenance
- **Backup Strategy**: Index backup and recovery

### Performance Tuning
- **Query Optimization**: Optimize search queries
- **Index Tuning**: Tune vector indexes
- **Cache Tuning**: Optimize cache settings
- **Resource Allocation**: Proper resource allocation

## Future Enhancements

### Advanced Features
- **Auto-complete**: Query auto-completion
- **Spell Correction**: Automatic spell correction
- **Query Suggestions**: Search suggestions
- **Personalization**: User-specific ranking

### Infrastructure
- **Distributed Search**: Multi-node search
- **Real-time Updates**: Live index updates
- **Advanced Indexing**: HNSW, IVF-PQ indexes
- **GPU Acceleration**: GPU-accelerated search

### Analytics
- **Search Analytics**: Comprehensive search analytics
- **User Behavior**: User search patterns
- **Result Quality**: Search result quality metrics
- **Performance Optimization**: Automated optimization
