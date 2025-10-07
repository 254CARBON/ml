#!/usr/bin/env python3
"""OpenSearch bootstrap script for ML platform.

This script sets up OpenSearch indices and mappings for the ML platform.
It creates the necessary indices for embeddings and search functionality.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List
import structlog
from opensearchpy import OpenSearch, exceptions

logger = structlog.get_logger("opensearch_bootstrap")


def create_opensearch_client(
    hosts: List[str],
    username: str = None,
    password: str = None,
    verify_certs: bool = False
) -> OpenSearch:
    """Create OpenSearch client."""
    return OpenSearch(
        hosts=hosts,
        http_auth=(username, password) if username and password else None,
        verify_certs=verify_certs,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        use_ssl=True if hosts[0].startswith('https') else False,
    )


def create_embeddings_index(client: OpenSearch, index_name: str, vector_dimension: int) -> None:
    """Create the embeddings index with kNN vector mapping."""
    mapping = {
        "mappings": {
            "properties": {
                "entity_type": {"type": "keyword"},
                "entity_id": {"type": "keyword"},
                "vector": {
                    "type": "knn_vector",
                    "dimension": vector_dimension,
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
                "number_of_replicas": 0,
                "refresh_interval": "1s"
            }
        }
    }
    
    try:
        if client.indices.exists(index=index_name):
            logger.info("Index already exists", index_name=index_name)
            return
        
        client.indices.create(index=index_name, body=mapping)
        logger.info("Created embeddings index", index_name=index_name, vector_dimension=vector_dimension)
        
    except Exception as e:
        logger.error("Failed to create embeddings index", index_name=index_name, error=str(e))
        raise


def create_search_index(client: OpenSearch, index_name: str) -> None:
    """Create the search index for hybrid search."""
    mapping = {
        "mappings": {
            "properties": {
                "entity_type": {"type": "keyword"},
                "entity_id": {"type": "keyword"},
                "text": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "meta": {"type": "object"},
                "tags": {"type": "keyword"},
                "tenant_id": {"type": "keyword"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"}
            }
        },
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "1s"
            }
        }
    }
    
    try:
        if client.indices.exists(index=index_name):
            logger.info("Index already exists", index_name=index_name)
            return
        
        client.indices.create(index=index_name, body=mapping)
        logger.info("Created search index", index_name=index_name)
        
    except Exception as e:
        logger.error("Failed to create search index", index_name=index_name, error=str(e))
        raise


def create_template(client: OpenSearch, template_name: str, vector_dimension: int) -> None:
    """Create index template for consistent index creation."""
    template = {
        "index_patterns": [f"{template_name}-*"],
        "template": {
            "mappings": {
                "properties": {
                    "entity_type": {"type": "keyword"},
                    "entity_id": {"type": "keyword"},
                    "vector": {
                        "type": "knn_vector",
                        "dimension": vector_dimension,
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
    }
    
    try:
        client.indices.put_index_template(name=template_name, body=template)
        logger.info("Created index template", template_name=template_name)
        
    except Exception as e:
        logger.error("Failed to create index template", template_name=template_name, error=str(e))
        raise


def wait_for_cluster(client: OpenSearch, timeout: int = 60) -> None:
    """Wait for OpenSearch cluster to be ready."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            health = client.cluster.health()
            if health['status'] in ['green', 'yellow']:
                logger.info("OpenSearch cluster is ready", status=health['status'])
                return
            else:
                logger.info("Waiting for OpenSearch cluster", status=health['status'])
                time.sleep(5)
        except Exception as e:
            logger.warning("Failed to check cluster health", error=str(e))
            time.sleep(5)
    
    raise TimeoutError("OpenSearch cluster did not become ready within timeout")


def main():
    """Main bootstrap function."""
    parser = argparse.ArgumentParser(description="Bootstrap OpenSearch for ML platform")
    parser.add_argument("--hosts", default="http://localhost:9200", help="OpenSearch hosts (comma-separated)")
    parser.add_argument("--username", help="OpenSearch username")
    parser.add_argument("--password", help="OpenSearch password")
    parser.add_argument("--verify-certs", action="store_true", help="Verify SSL certificates")
    parser.add_argument("--embeddings-index", default="ml_embeddings", help="Embeddings index name")
    parser.add_argument("--search-index", default="ml_search", help="Search index name")
    parser.add_argument("--vector-dimension", type=int, default=384, help="Vector dimension")
    parser.add_argument("--template-name", default="ml_embeddings_template", help="Index template name")
    parser.add_argument("--wait-timeout", type=int, default=60, help="Cluster wait timeout in seconds")
    parser.add_argument("--skip-template", action="store_true", help="Skip creating index template")
    
    args = parser.parse_args()
    
    # Parse hosts
    hosts = [host.strip() for host in args.hosts.split(",")]
    
    # Create client
    client = create_opensearch_client(
        hosts=hosts,
        username=args.username,
        password=args.password,
        verify_certs=args.verify_certs
    )
    
    try:
        # Wait for cluster to be ready
        wait_for_cluster(client, args.wait_timeout)
        
        # Create index template
        if not args.skip_template:
            create_template(client, args.template_name, args.vector_dimension)
        
        # Create embeddings index
        create_embeddings_index(client, args.embeddings_index, args.vector_dimension)
        
        # Create search index
        create_search_index(client, args.search_index)
        
        logger.info("OpenSearch bootstrap completed successfully")
        
    except Exception as e:
        logger.error("OpenSearch bootstrap failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
