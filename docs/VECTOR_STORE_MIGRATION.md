# Vector Store Migration Strategy

## Overview

The 254Carbon ML Platform currently uses PgVector for vector storage and similarity search. This document outlines the strategy for migrating to OpenSearch while maintaining service availability and data consistency.

## OpenSearch Bootstrap

### Quick Start

```bash
# Bootstrap OpenSearch with default settings
python scripts/opensearch_bootstrap.py

# Bootstrap with custom settings
python scripts/opensearch_bootstrap.py \
  --hosts "https://opensearch.example.com:9200" \
  --username "admin" \
  --password "password" \
  --verify-certs \
  --embeddings-index "ml_embeddings_prod" \
  --vector-dimension 768
```

### Environment Variables

Set these environment variables to use OpenSearch:

```bash
export ML_VECTOR_BACKEND=opensearch
export ML_OPENSEARCH_HOSTS="https://opensearch.example.com:9200"
export ML_OPENSEARCH_USERNAME="admin"
export ML_OPENSEARCH_PASSWORD="password"
export ML_OPENSEARCH_VERIFY_CERTS=true
export ML_OPENSEARCH_INDEX="ml_embeddings"
export ML_VECTOR_DIMENSION=384
```

### Index Templates

The bootstrap script creates index templates for consistent index creation:

- `ml_embeddings_template`: Template for embedding indices
- Supports dynamic index creation with tenant-specific naming
- Configures kNN vector search with HNSW algorithm

## Current State: PgVector

### Advantages
- **Simplicity**: Single database for metadata and vectors
- **ACID Compliance**: Strong consistency guarantees
- **SQL Integration**: Native SQL query capabilities
- **Backup**: Integrated backup and recovery
- **Cost**: Lower operational complexity

### Limitations
- **Scalability**: Limited horizontal scaling
- **Performance**: Slower for large vector collections
- **Memory Usage**: High memory requirements
- **Index Options**: Limited vector index types
- **Search Features**: Basic similarity search only

## Target State: OpenSearch

### Advantages
- **Scalability**: Horizontal scaling capabilities
- **Performance**: Optimized for large-scale vector search
- **Memory Efficiency**: Better memory utilization
- **Advanced Indexing**: HNSW, IVF-PQ index support
- **Rich Search**: Advanced search and filtering
- **Real-time**: Real-time index updates

### Considerations
- **Complexity**: Additional infrastructure component
- **Consistency**: Eventual consistency model
- **Backup**: Separate backup strategy needed
- **Cost**: Additional operational overhead
- **Learning Curve**: Team expertise required

## Migration Strategy

### Phase 1: Preparation (Weeks 1-2)
- **Infrastructure Setup**: Deploy OpenSearch cluster
- **Index Design**: Design vector index schema
- **Data Mapping**: Map PgVector schema to OpenSearch
- **Testing Environment**: Set up test environment
- **Performance Testing**: Benchmark performance

### Phase 2: Dual-Write (Weeks 3-4)
- **Dual-Write Implementation**: Write to both stores
- **Data Synchronization**: Keep both stores in sync
- **Validation**: Validate data consistency
- **Performance Monitoring**: Monitor performance impact
- **Rollback Plan**: Prepare rollback procedures

### Phase 3: Read Migration (Weeks 5-6)
- **Read Routing**: Route reads to OpenSearch
- **A/B Testing**: Compare search quality
- **Performance Validation**: Validate performance targets
- **Monitoring**: Enhanced monitoring and alerting
- **Gradual Rollout**: Gradual traffic migration

### Phase 4: Cleanup (Weeks 7-8)
- **Write Migration**: Migrate writes to OpenSearch
- **Data Validation**: Final data consistency check
- **PgVector Deprecation**: Deprecate PgVector usage
- **Documentation**: Update documentation
- **Training**: Team training on OpenSearch

## Technical Implementation

### Vector Store Adapter
```python
class VectorStoreAdapter:
    def __init__(self, config):
        self.pgvector = PgVectorStore(config.pgvector)
        self.opensearch = OpenSearchStore(config.opensearch)
        self.migration_mode = config.migration_mode
    
    async def store_embedding(self, **kwargs):
        if self.migration_mode == "dual_write":
            await self.pgvector.store_embedding(**kwargs)
            await self.opensearch.store_embedding(**kwargs)
        elif self.migration_mode == "opensearch":
            await self.opensearch.store_embedding(**kwargs)
        else:
            await self.pgvector.store_embedding(**kwargs)
```

### Data Synchronization
```python
class DataSynchronizer:
    async def sync_pgvector_to_opensearch(self):
        """Sync data from PgVector to OpenSearch."""
        async for batch in self.pgvector.get_all_embeddings():
            await self.opensearch.batch_store_embeddings(batch)
    
    async def validate_consistency(self):
        """Validate data consistency between stores."""
        pgvector_count = await self.pgvector.get_count()
        opensearch_count = await self.opensearch.get_count()
        assert pgvector_count == opensearch_count
```

### Index Schema
```json
{
  "mappings": {
    "properties": {
      "entity_type": {"type": "keyword"},
      "entity_id": {"type": "keyword"},
      "vector": {
        "type": "knn_vector",
        "dimension": 384,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "nmslib"
        }
      },
      "metadata": {"type": "object"},
      "model_version": {"type": "keyword"},
      "tenant_id": {"type": "keyword"},
      "created_at": {"type": "date"},
      "updated_at": {"type": "date"}
    }
  }
}
```

## Performance Considerations

### Index Configuration
- **HNSW Parameters**: Optimize HNSW index parameters
- **Shard Count**: Appropriate shard count for data size
- **Replica Count**: Balance availability and performance
- **Refresh Interval**: Optimize refresh interval
- **Memory Allocation**: Proper memory allocation

### Query Optimization
- **Batch Queries**: Batch multiple queries
- **Filter Optimization**: Optimize filter queries
- **Caching**: Implement query result caching
- **Connection Pooling**: Efficient connection management

### Monitoring
- **Performance Metrics**: Track query performance
- **Resource Usage**: Monitor CPU, memory, disk usage
- **Index Health**: Monitor index health and status
- **Error Rates**: Track error rates and failures

## Risk Mitigation

### Data Loss Prevention
- **Backup Strategy**: Regular backups of both stores
- **Data Validation**: Continuous data consistency checks
- **Rollback Plan**: Quick rollback to PgVector
- **Monitoring**: Real-time monitoring and alerting

### Performance Degradation
- **Performance Testing**: Comprehensive performance testing
- **Gradual Migration**: Gradual traffic migration
- **A/B Testing**: Compare performance between stores
- **Optimization**: Continuous performance optimization

### Service Disruption
- **Zero-Downtime**: Zero-downtime migration strategy
- **Health Checks**: Comprehensive health checks
- **Circuit Breakers**: Circuit breakers for failures
- **Fallback**: Fallback to PgVector on failures

## Migration Checklist

### Pre-Migration
- [ ] OpenSearch cluster deployed and configured
- [ ] Index schema designed and tested
- [ ] Data synchronization scripts ready
- [ ] Performance benchmarks established
- [ ] Rollback procedures documented
- [ ] Team training completed

### During Migration
- [ ] Dual-write mode enabled
- [ ] Data synchronization running
- [ ] Performance monitoring active
- [ ] Error rates within acceptable limits
- [ ] Data consistency validated
- [ ] Team on standby for issues

### Post-Migration
- [ ] All traffic migrated to OpenSearch
- [ ] PgVector writes disabled
- [ ] Performance targets met
- [ ] Data consistency validated
- [ ] Documentation updated
- [ ] Team training completed

## Success Criteria

### Performance Targets
- **Query Latency**: < 200ms P95 (vs 400ms PgVector)
- **Throughput**: > 1000 queries/second (vs 100 QPS)
- **Memory Usage**: < 50% of current usage
- **Index Size**: < 80% of current size

### Quality Metrics
- **Search Quality**: No degradation in search quality
- **Data Consistency**: 100% data consistency
- **Availability**: > 99.9% availability
- **Error Rate**: < 0.1% error rate

### Operational Metrics
- **Migration Time**: < 8 weeks total
- **Downtime**: Zero downtime
- **Data Loss**: Zero data loss
- **Rollback Time**: < 1 hour rollback time

## Future Considerations

### Advanced Features
- **Multi-Modal**: Support for different vector types
- **Real-Time**: Real-time index updates
- **Federated Search**: Multi-cluster search
- **Advanced Analytics**: Search analytics and insights

### Infrastructure Evolution
- **Cloud Migration**: Cloud-native OpenSearch
- **Auto-Scaling**: Automatic scaling based on demand
- **Multi-Region**: Multi-region deployment
- **Disaster Recovery**: Comprehensive disaster recovery

### Cost Optimization
- **Resource Optimization**: Optimize resource usage
- **Storage Optimization**: Optimize storage costs
- **Query Optimization**: Optimize query costs
- **Monitoring**: Cost monitoring and optimization
