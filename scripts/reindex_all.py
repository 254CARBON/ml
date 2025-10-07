#!/usr/bin/env python3
"""Script to reindex all entities for embedding generation."""

import argparse
import asyncio
import sys
from typing import List, Optional
import structlog

from libs.common.config import BaseConfig
from libs.common.logging import configure_logging
from libs.common.events import create_event_publisher

logger = structlog.get_logger("reindex_all")


async def reindex_entities(
    entity_types: List[str],
    batch_size: int = 500,
    model_version: str = "default",
    config: Optional[BaseConfig] = None
) -> bool:
    """Reindex entities for embedding generation."""
    try:
        if not config:
            config = BaseConfig()
        
        # Initialize event publisher
        event_publisher = create_event_publisher(config.ml_redis_url)
        
        # Trigger reindexing for each entity type
        for entity_type in entity_types:
            logger.info(
                "Triggering reindex",
                entity_type=entity_type,
                batch_size=batch_size,
                model_version=model_version
            )
            
            # Publish reindex request event
            event_publisher.publish_embedding_reindex_request(
                entity_type=entity_type,
                batch_size=batch_size,
                model_version=model_version
            )
        
        logger.info(
            "Reindex requests published",
            entity_types=entity_types,
            batch_size=batch_size,
            model_version=model_version
        )
        
        return True
        
    except Exception as e:
        logger.error(
            "Reindexing failed",
            entity_types=entity_types,
            error=str(e)
        )
        return False


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description="Reindex all entities for embedding generation")
    parser.add_argument("--entity", required=True, help="Entity type to reindex")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for processing")
    parser.add_argument("--model-version", default="default", help="Model version to use")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging("reindex_all", "INFO", "json")
    
    # Load configuration
    config = BaseConfig()
    
    # Reindex entities
    success = asyncio.run(reindex_entities(
        entity_types=[args.entity],
        batch_size=args.batch_size,
        model_version=args.model_version,
        config=config
    ))
    
    if success:
        print(f"Reindexing triggered for {args.entity} successfully")
        sys.exit(0)
    else:
        print(f"Failed to trigger reindexing for {args.entity}")
        sys.exit(1)


if __name__ == "__main__":
    main()
