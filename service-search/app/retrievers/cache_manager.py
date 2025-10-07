"""Cache manager for search results and query embeddings."""

import asyncio
import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
import redis.asyncio as redis
import structlog

logger = structlog.get_logger("search_cache")


class SearchCacheManager:
    """Manages caching for search operations."""
    
    def __init__(
        self,
        redis_url: str,
        query_cache_ttl: int = 300,      # 5 minutes
        embedding_cache_ttl: int = 3600,  # 1 hour
        result_cache_ttl: int = 600       # 10 minutes
    ):
        self.redis_client = redis.from_url(redis_url)
        self.query_cache_ttl = query_cache_ttl
        self.embedding_cache_ttl = embedding_cache_ttl
        self.result_cache_ttl = result_cache_ttl
        
        # Cache key prefixes
        self.query_prefix = "search:query:"
        self.embedding_prefix = "search:embedding:"
        self.result_prefix = "search:result:"
        self.stats_prefix = "search:stats:"
        self.entity_result_prefix = f"{self.result_prefix}entity:"
    
    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, dict):
            # Sort keys for consistent hashing
            sorted_data = json.dumps(data, sort_keys=True)
        else:
            sorted_data = str(data)
        
        hash_obj = hashlib.md5(sorted_data.encode())
        return f"{prefix}{hash_obj.hexdigest()}"
    
    def _generate_entity_cache_key(self, entity_type: str, entity_id: str) -> str:
        """Generate cache key for entity specific lookups."""
        raw_key = f"{entity_type}:{entity_id}"
        hash_obj = hashlib.md5(raw_key.encode())
        return f"{self.entity_result_prefix}{hash_obj.hexdigest()}"
    
    async def get_cached_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get cached query embedding."""
        try:
            cache_key = self._generate_cache_key(self.embedding_prefix, query)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                embedding_data = json.loads(cached_data)
                logger.debug("Query embedding cache hit", query=query[:50])
                return embedding_data["embedding"]
            
            logger.debug("Query embedding cache miss", query=query[:50])
            return None
            
        except Exception as e:
            logger.warning("Failed to get cached query embedding", error=str(e))
            return None
    
    async def cache_query_embedding(self, query: str, embedding: List[float]):
        """Cache query embedding."""
        try:
            cache_key = self._generate_cache_key(self.embedding_prefix, query)
            cache_data = {
                "query": query,
                "embedding": embedding,
                "cached_at": time.time()
            }
            
            await self.redis_client.setex(
                cache_key,
                self.embedding_cache_ttl,
                json.dumps(cache_data)
            )
            
            logger.debug("Query embedding cached", query=query[:50])
            
        except Exception as e:
            logger.warning("Failed to cache query embedding", error=str(e))
    
    async def get_cached_search_results(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        semantic: bool = True,
        limit: int = 10
    ) -> Optional[List[Tuple[str, str, float, Dict[str, Any]]]]:
        """Get cached search results."""
        try:
            cache_data = {
                "query": query,
                "filters": filters,
                "semantic": semantic,
                "limit": limit
            }
            cache_key = self._generate_cache_key(self.result_prefix, cache_data)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                result_data = json.loads(cached_data)
                logger.debug("Search results cache hit", query=query[:50])
                
                # Convert back to tuple format
                results = []
                for item in result_data["results"]:
                    results.append((
                        item["entity_type"],
                        item["entity_id"],
                        item["score"],
                        item["metadata"]
                    ))
                
                return results
            
            logger.debug("Search results cache miss", query=query[:50])
            return None
            
        except Exception as e:
            logger.warning("Failed to get cached search results", error=str(e))
            return None
    
    async def cache_search_results(
        self,
        query: str,
        results: List[Tuple[str, str, float, Dict[str, Any]]],
        filters: Optional[Dict[str, Any]] = None,
        semantic: bool = True,
        limit: int = 10
    ):
        """Cache search results."""
        try:
            cache_data = {
                "query": query,
                "filters": filters,
                "semantic": semantic,
                "limit": limit
            }
            cache_key = self._generate_cache_key(self.result_prefix, cache_data)
            
            # Convert results to serializable format
            serializable_results = []
            for entity_type, entity_id, score, metadata in results:
                serializable_results.append({
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "score": score,
                    "metadata": metadata
                })
            
            # Link result cache to entities for targeted invalidation
            if results:
                async with self.redis_client.pipeline(transaction=False) as pipe:
                    for entity_type, entity_id, *_ in results:
                        entity_cache_key = self._generate_entity_cache_key(entity_type, entity_id)
                        pipe.sadd(entity_cache_key, cache_key)
                        pipe.expire(entity_cache_key, self.result_cache_ttl)
                    await pipe.execute()
            
            result_data = {
                "query": query,
                "results": serializable_results,
                "cached_at": time.time(),
                "filters": filters,
                "semantic": semantic,
                "limit": limit
            }
            
            await self.redis_client.setex(
                cache_key,
                self.result_cache_ttl,
                json.dumps(result_data, default=str)
            )
            
            logger.debug("Search results cached", query=query[:50], count=len(results))
            
        except Exception as e:
            logger.warning("Failed to cache search results", error=str(e))
    
    async def invalidate_entity_cache(self, entity_type: str, entity_id: str):
        """Invalidate cache entries related to a specific entity."""
        try:
            entity_cache_key = self._generate_entity_cache_key(entity_type, entity_id)
            related_cache_keys = await self.redis_client.smembers(entity_cache_key)
            
            deleted_count = 0
            if related_cache_keys:
                decoded_keys = [
                    key.decode() if isinstance(key, bytes) else key
                    for key in related_cache_keys
                ]
                
                deleted_count = await self.redis_client.delete(*decoded_keys)
            
            await self.redis_client.delete(entity_cache_key)
            
            if deleted_count:
                logger.info(
                    "Cache invalidated for entity",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    deleted_keys=deleted_count
                )
            else:
                logger.debug(
                    "No cached search results found for entity",
                    entity_type=entity_type,
                    entity_id=entity_id
                )
            
        except Exception as e:
            logger.warning("Cache invalidation failed", error=str(e))
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            # Get cache key counts
            query_keys = await self.redis_client.keys(f"{self.query_prefix}*")
            embedding_keys = await self.redis_client.keys(f"{self.embedding_prefix}*")
            result_keys = await self.redis_client.keys(f"{self.result_prefix}*")
            
            # Get memory usage
            memory_info = await self.redis_client.info("memory")
            
            stats = {
                "cache_counts": {
                    "query_embeddings": len(embedding_keys),
                    "search_results": len(result_keys),
                    "total_keys": len(query_keys) + len(embedding_keys) + len(result_keys)
                },
                "memory_usage": {
                    "used_memory": memory_info.get("used_memory", 0),
                    "used_memory_human": memory_info.get("used_memory_human", "0B"),
                    "maxmemory": memory_info.get("maxmemory", 0)
                },
                "ttl_settings": {
                    "query_cache_ttl": self.query_cache_ttl,
                    "embedding_cache_ttl": self.embedding_cache_ttl,
                    "result_cache_ttl": self.result_cache_ttl
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {}
    
    async def clear_cache(self, cache_type: str = "all"):
        """Clear cache entries."""
        try:
            if cache_type == "all":
                patterns = [f"{self.query_prefix}*", f"{self.embedding_prefix}*", f"{self.result_prefix}*"]
            elif cache_type == "embeddings":
                patterns = [f"{self.embedding_prefix}*"]
            elif cache_type == "results":
                patterns = [f"{self.result_prefix}*"]
            elif cache_type == "queries":
                patterns = [f"{self.query_prefix}*"]
            else:
                raise ValueError(f"Unknown cache type: {cache_type}")
            
            total_deleted = 0
            for pattern in patterns:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    deleted = await self.redis_client.delete(*keys)
                    total_deleted += deleted
            
            logger.info("Cache cleared", cache_type=cache_type, keys_deleted=total_deleted)
            
        except Exception as e:
            logger.error("Failed to clear cache", cache_type=cache_type, error=str(e))
    
    async def warm_cache(self, popular_queries: List[str]):
        """Warm up cache with popular queries."""
        try:
            logger.info("Warming up cache", query_count=len(popular_queries))
            
            # This would typically:
            # 1. Generate embeddings for popular queries
            # 2. Execute searches and cache results
            # 3. Pre-compute common filter combinations
            
            for query in popular_queries:
                # Generate and cache query embedding
                # This would call the embedding service
                logger.debug("Warming cache for query", query=query[:50])
            
            logger.info("Cache warm-up completed")
            
        except Exception as e:
            logger.error("Cache warm-up failed", error=str(e))
    
    async def close(self):
        """Close Redis connection."""
        try:
            await self.redis_client.close()
            logger.info("Search cache manager closed")
        except Exception as e:
            logger.warning("Failed to close cache manager", error=str(e))


def create_search_cache_manager(
    redis_url: str,
    query_cache_ttl: int = 300,
    embedding_cache_ttl: int = 3600,
    result_cache_ttl: int = 600
) -> SearchCacheManager:
    """Create search cache manager."""
    return SearchCacheManager(
        redis_url=redis_url,
        query_cache_ttl=query_cache_ttl,
        embedding_cache_ttl=embedding_cache_ttl,
        result_cache_ttl=result_cache_ttl
    )
