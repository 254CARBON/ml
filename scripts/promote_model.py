#!/usr/bin/env python3
"""Script to promote ML models to production."""

import argparse
import asyncio
import sys
import os
from typing import Optional
import json
import shutil
from pathlib import Path
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
        
        # Initialize model storage path
        model_storage_path = Path(os.getenv("ML_MODEL_STORAGE_PATH", "/app/models"))
        
        # Get the model from local storage
        if run_id:
            # Find model by run_id in experiments directory
            exp_dir = model_storage_path / model_name / "experiments"
            model_path = None
            for run_dir in exp_dir.glob("run_*"):
                if run_id in str(run_dir):
                    model_path = run_dir
                    break
            if not model_path:
                logger.error("Model run not found", run_id=run_id)
                return False
        else:
            # Get latest model from staging
            staging_path = model_storage_path / model_name / "staging"
            if not staging_path.exists():
                logger.error("No staging model found", model_name=model_name)
                return False
            model_path = staging_path
        
        # Create target directory
        target_dir = model_storage_path / model_name / stage.lower()
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files to target directory
        if model_path.is_dir():
            # Copy all files from source to target
            for item in model_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, target_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, target_dir / item.name, dirs_exist_ok=True)
        else:
            # Single file
            shutil.copy2(model_path, target_dir / model_path.name)
        
        # Create metadata file
        metadata = {
            "model_name": model_name,
            "stage": stage,
            "promoted_at": str(Path().cwd()),  # Use current timestamp
            "source_path": str(model_path),
            "run_id": run_id
        }
        
        with open(target_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
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
            stage=stage,
            target_path=str(target_dir)
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
    parser.add_argument("--run-id", help="Model run ID")
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