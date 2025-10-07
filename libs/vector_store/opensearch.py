"""OpenSearch vector store implementation."""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import structlog
from opensearchpy import OpenSearch, exceptions
from opensearchpy.helpers import bulk

from .base import VectorStore

logger = structlog.get_logger("vector_store.opensearch")


class OpenSearchVectorStore(VectorStore):
    """OpenSearch-based vector store implementation."""

    def __init__(
        self,
        hosts: List[str],
        index_name: str = "ml_embeddings",
        vector_dimension: int = 384,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_certs: bool = False,
        ssl_assert_hostname: bool = False,
        ssl_show_warn: bool = False,
    ):
        """Initialize OpenSearch vector store.

        Args:
            hosts: List of OpenSearch host URLs
            index_name: Name of the index to store vectors
            vector_dimension: Dimension of the vectors
            username: OpenSearch username
            password: OpenSearch password
            verify_certs: Whether to verify SSL certificates
            ssl_assert_hostname: Whether to assert hostname
            ssl_show_warn: Whether to show SSL warnings
        """
        self.hosts = hosts
        self.index_name = index_name
        self.vector_dimension = vector_dimension
        self.username = username
        self.password = password
        self.verify_certs = verify_certs
        self.ssl_assert_hostname = ssl_assert_hostname
        self.ssl_show_warn = ssl_show_warn
        
        # Initialize OpenSearch client
        self.client = OpenSearch(
            hosts=hosts,
            http_auth=(username, password) if username and password else None,
            verify_certs=verify_certs,
            ssl_assert_hostname=ssl_assert_hostname,
            ssl_show_warn=ssl_show_warn,
            use_ssl=True if hosts[0].startswith('https') else False,
        )
        
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the vector store by creating the index if it doesn't exist."""
        try:
            # Check if index exists
            if not self.client.indices.exists(index=self.index_name):
                # Create index with vector mapping
                mapping = {
                    "mappings": {
                        "properties": {
                            "entity_type": {"type": "keyword"},
                            "entity_id": {"type": "keyword"},
                            "vector": {
                                "type": "knn_vector",
                                "dimension": self.vector_dimension,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "nmslib",
                                    "parameters": {
                                        "ef_construction": 128,
                                        "m": 24
                                    }
                                }
                            },
                            "meta": {"type": "object"},
                            "model_version": {"type": "keyword"},
                            "tenant_id": {"type": "keyword"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"}
                        }
                    },
                    "settings": {
                        "index": {
                            "knn": True,
                            "knn.algo_param.ef_search": 100,
                            "number_of_shards": 1,
                            "number_of_replicas": 0
                        }
                    }
                }
                
                self.client.indices.create(index=self.index_name, body=mapping)
                logger.info("OpenSearch index created", index_name=self.index_name)
            
            self._initialized = True
            logger.info("OpenSearch vector store initialized", index_name=self.index_name)
            
        except Exception as e:
            logger.error("Failed to initialize OpenSearch vector store", error=str(e))
            raise

    async def store_embedding(
        self,
        entity_type: str,
        entity_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        model_version: str = "default",
        tenant_id: str = "default"
    ) -> None:
        """Store a single embedding."""
        try:
            if not self._initialized:
                await self.initialize()
            
            doc = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "vector": vector.tolist(),
                "meta": metadata or {},
                "model_version": model_version,
                "tenant_id": tenant_id,
                "created_at": int(time.time() * 1000),
                "updated_at": int(time.time() * 1000)
            }
            
            # Use entity_type + entity_id + tenant_id as document ID
            doc_id = f"{entity_type}:{entity_id}:{tenant_id}"
            
            self.client.index(
                index=self.index_name,
                id=doc_id,
                body=doc
            )
            
            logger.info(
                "Embedding stored in OpenSearch",
                entity_type=entity_type,
                entity_id=entity_id,
                tenant_id=tenant_id
            )
            
        except Exception as e:
            logger.error(
                "Failed to store embedding in OpenSearch",
                entity_type=entity_type,
                entity_id=entity_id,
                error=str(e)
            )
            raise

    async def batch_store_embeddings(
        self,
        embeddings: List[Dict[str, Any]],
        tenant_id: str = "default"
    ) -> int:
        """Store multiple embeddings in batch."""
        if not embeddings:
            return 0
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Prepare batch data
            actions = []
            for emb in embeddings:
                doc = {
                    "entity_type": emb['entity_type'],
                    "entity_id": emb['entity_id'],
                    "vector": emb['vector'].tolist(),
                    "meta": emb.get('metadata', {}),
                    "model_version": emb.get('model_version', 'default'),
                    "tenant_id": tenant_id,
                    "created_at": int(time.time() * 1000),
                    "updated_at": int(time.time() * 1000)
                }
                
                # Use entity_type + entity_id + tenant_id as document ID
                doc_id = f"{emb['entity_type']}:{emb['entity_id']}:{tenant_id}"
                
                actions.append({
                    "_index": self.index_name,
                    "_id": doc_id,
                    "_source": doc
                })
            
            # Bulk insert
            success_count, failed_items = bulk(self.client, actions)
            
            if failed_items:
                logger.warning(
                    "Some embeddings failed to store in OpenSearch",
                    failed_count=len(failed_items),
                    total_count=len(embeddings)
                )
            
            logger.info(
                "Batch stored embeddings in OpenSearch",
                count=success_count,
                tenant_id=tenant_id
            )
            
            return success_count
            
        except Exception as e:
            logger.error(
                "Batch store embeddings failed in OpenSearch",
                count=len(embeddings),
                error=str(e)
            )
            raise

    async def search_similar(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        similarity_threshold: float = 0.0,
        entity_type: Optional[str] = None,
        tenant_id: str = "default"
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Search for similar vectors using kNN."""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Build query
            query = {
                "size": limit,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "vector": {
                                        "vector": query_vector.tolist(),
                                        "k": limit
                                    }
                                }
                            }
                        ],
                        "filter": [
                            {"term": {"tenant_id": tenant_id}}
                        ]
                    }
                }
            }
            
            # Add entity type filter if specified
            if entity_type:
                query["query"]["bool"]["filter"].append(
                    {"term": {"entity_type": entity_type}}
                )
            
            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=query
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                score = hit['_score']
                
                if score >= similarity_threshold:
                    results.append((
                        source['entity_type'],
                        source['entity_id'],
                        float(score),
                        source.get('meta', {})
                    ))
            
            logger.info(
                "OpenSearch similarity search completed",
                results_count=len(results),
                tenant_id=tenant_id
            )
            
            return results
            
        except Exception as e:
            logger.error("OpenSearch similarity search failed", error=str(e))
            return []

    async def delete_embedding(
        self,
        entity_type: str,
        entity_id: str,
        tenant_id: str = "default"
    ) -> bool:
        """Delete an embedding."""
        try:
            if not self._initialized:
                await self.initialize()
            
            doc_id = f"{entity_type}:{entity_id}:{tenant_id}"
            
            response = self.client.delete(
                index=self.index_name,
                id=doc_id
            )
            
            if response['result'] == 'deleted':
                logger.info(
                    "Embedding deleted from OpenSearch",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    tenant_id=tenant_id
                )
                return True
            else:
                logger.warning(
                    "Embedding not found in OpenSearch",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    tenant_id=tenant_id
                )
                return False
                
        except exceptions.NotFoundError:
            logger.warning(
                "Embedding not found in OpenSearch",
                entity_type=entity_type,
                entity_id=entity_id,
                tenant_id=tenant_id
            )
            return False
        except Exception as e:
            logger.error(
                "Failed to delete embedding from OpenSearch",
                entity_type=entity_type,
                entity_id=entity_id,
                error=str(e)
            )
            raise

    async def get_embedding(
        self,
        entity_type: str,
        entity_id: str,
        tenant_id: str = "default"
    ) -> Optional[np.ndarray]:
        """Get a specific embedding."""
        try:
            if not self._initialized:
                await self.initialize()
            
            doc_id = f"{entity_type}:{entity_id}:{tenant_id}"
            
            response = self.client.get(
                index=self.index_name,
                id=doc_id
            )
            
            if response['found']:
                vector = np.array(response['_source']['vector'])
                logger.info(
                    "Embedding retrieved from OpenSearch",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    tenant_id=tenant_id
                )
                return vector
            else:
                return None
                
        except exceptions.NotFoundError:
            return None
        except Exception as e:
            logger.error(
                "Failed to get embedding from OpenSearch",
                entity_type=entity_type,
                entity_id=entity_id,
                error=str(e)
            )
            raise

    async def list_embeddings(
        self,
        entity_type: Optional[str] = None,
        tenant_id: str = "default",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List embeddings with optional filtering."""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Build query
            query = {
                "size": limit,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"tenant_id": tenant_id}}
                        ]
                    }
                }
            }
            
            # Add entity type filter if specified
            if entity_type:
                query["query"]["bool"]["filter"].append(
                    {"term": {"entity_type": entity_type}}
                )
            
            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=query
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                results.append({
                    "entity_type": source['entity_type'],
                    "entity_id": source['entity_id'],
                    "vector": np.array(source['vector']),
                    "metadata": source.get('meta', {}),
                    "model_version": source.get('model_version', 'default'),
                    "tenant_id": source['tenant_id'],
                    "created_at": source.get('created_at'),
                    "updated_at": source.get('updated_at')
                })
            
            logger.info(
                "OpenSearch embeddings listed",
                results_count=len(results),
                entity_type=entity_type,
                tenant_id=tenant_id
            )
            
            return results
            
        except Exception as e:
            logger.error("OpenSearch list embeddings failed", error=str(e))
            return []

    async def close(self) -> None:
        """Close the OpenSearch client connection."""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
            logger.info("OpenSearch client connection closed")
        except Exception as e:
            logger.error("Failed to close OpenSearch client", error=str(e))
