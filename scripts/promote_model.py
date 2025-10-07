#!/usr/bin/env python3
"""Script to promote ML models to production."""

import argparse
import asyncio
import sys
from typing import Optional
import mlflow
import structlog

from libs.common.config import BaseConfig
from libs.common.logging import configure_logging
from libs.common.events import create_event_publisher

logger = structlog.get_logger("promote_model")


async def promote_model(
    model_name: str,
    run_id: Optional[str] = None,
    stage: str = "Production",
    config: Optional[BaseConfig] = None
) -> bool:
    """Promote a model to the specified stage."""
    try:
        if not config:
            config = BaseConfig()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.ml_mlflow_tracking_uri)
        
        # Get the model
        if run_id:
            model_uri = f"runs:/{run_id}/model"
        else:
            # Get the latest version
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
            model_uri = f"models:/{model_name}/{latest_version.version}"
        
        # Promote the model
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=run_id or "latest",
            stage=stage
        )
        
        # Publish promotion event
        event_publisher = create_event_publisher(config.ml_redis_url)
        event_publisher.publish_model_promoted(
            model_name=model_name,
            version=run_id or "latest",
            stage=stage,
            run_id=run_id or "latest"
        )
        
        logger.info(
            "Model promoted successfully",
            model_name=model_name,
            run_id=run_id,
            stage=stage
        )
        
        return True
        
    except Exception as e:
        logger.error(
            "Model promotion failed",
            model_name=model_name,
            run_id=run_id,
            stage=stage,
            error=str(e)
        )
        return False


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description="Promote ML model to production")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--run-id", help="MLflow run ID")
    parser.add_argument("--stage", default="Production", help="Target stage")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging("promote_model", "INFO", "json")
    
    # Load configuration
    config = BaseConfig()
    
    # Promote the model
    success = asyncio.run(promote_model(
        model_name=args.model,
        run_id=args.run_id,
        stage=args.stage,
        config=config
    ))
    
    if success:
        print(f"Model {args.model} promoted to {args.stage} successfully")
        sys.exit(0)
    else:
        print(f"Failed to promote model {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
