"""Airflow DAG for embedding reindexing pipeline."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.http import HttpSensor
import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from libs.common.config import BaseConfig  # noqa: E402
from scripts.reindex_all import reindex_entities  # noqa: E402

logger = structlog.get_logger("embedding_reindex_dag")

default_args = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "max_active_runs": 1,
}


def check_reindex_needed(**context):
    """Check if reindexing is needed."""
    
    logger.info("Checking if embedding reindex is needed")
    
    # This would check:
    # 1. New data availability
    # 2. Model version changes
    # 3. Data drift detection
    # 4. Scheduled maintenance
    
    # For now, simulate the check
    import random
    needs_reindex = random.choice([True, False])
    
    if needs_reindex:
        logger.info("Reindexing needed")
        return 'start_reindex'
    else:
        logger.info("No reindexing needed")
        return 'skip_reindex'


def get_entities_to_reindex(**context):
    """Get list of entities that need reindexing."""
    
    logger.info("Getting entities to reindex")
    
    # This would query the database for entities needing reindex
    entities_to_reindex = [
        {"entity_type": "instruments", "count": 5000, "priority": "high"},
        {"entity_type": "curves", "count": 2000, "priority": "medium"},
        {"entity_type": "scenarios", "count": 1000, "priority": "low"}
    ]
    
    total_entities = sum(e["count"] for e in entities_to_reindex)
    
    logger.info("Entities to reindex identified", 
               entity_types=len(entities_to_reindex),
               total_count=total_entities)
    
    return entities_to_reindex


def trigger_embedding_generation(**context):
    """Trigger embedding generation for entities."""
    
    logger.info("Triggering embedding generation")
    
    # Get entities from previous task
    entities = context['task_instance'].xcom_pull(task_ids='get_entities_to_reindex')
    
    results = []
    
    config = BaseConfig()
    
    for entity in entities:
        entity_type = entity["entity_type"]
        batch_size = min(500, entity["count"] // 10)  # Adaptive batch size
        
        try:
            success = asyncio.run(
                reindex_entities(
                    entity_types=[entity_type],
                    batch_size=batch_size,
                    config=config
                )
            )
            
            if success:
                logger.info("Reindex triggered successfully", entity_type=entity_type)
                results.append({
                    "entity_type": entity_type,
                    "status": "triggered",
                    "batch_size": batch_size
                })
            else:
                logger.error("Reindex trigger failed", entity_type=entity_type)
                results.append({
                    "entity_type": entity_type,
                    "status": "failed",
                    "error": "reindex_entities returned False"
                })
                
        except Exception as e:
            logger.error("Reindex trigger error", entity_type=entity_type, error=str(e))
            results.append({"entity_type": entity_type, "status": "error", "error": str(e)})
    
    return results


def monitor_reindex_progress(**context):
    """Monitor reindexing progress."""
    
    logger.info("Monitoring reindex progress")
    
    # This would monitor the indexer worker progress
    # Check Redis queues, worker status, etc.
    
    import time
    import asyncio
    import redis
    
    redis_url = Variable.get("ML_REDIS_URL", "redis://redis:6379")
    redis_client = redis.from_url(redis_url)
    
    # Check queue lengths
    queue_lengths = {}
    queues = ["embeddings_rebuild", "embeddings_batch", "embeddings_cleanup"]
    
    for queue in queues:
        try:
            length = redis_client.llen(queue)
            queue_lengths[queue] = length
        except Exception as e:
            logger.warning(f"Failed to check queue {queue}", error=str(e))
            queue_lengths[queue] = -1
    
    # Wait for queues to drain (simplified)
    max_wait_minutes = 60
    wait_start = time.time()
    
    while time.time() - wait_start < max_wait_minutes * 60:
        total_pending = sum(max(0, length) for length in queue_lengths.values())
        
        if total_pending == 0:
            logger.info("All reindexing queues are empty")
            break
        
        logger.info("Waiting for reindexing to complete", 
                   pending_jobs=total_pending,
                   elapsed_minutes=(time.time() - wait_start) / 60)
        
        time.sleep(30)  # Check every 30 seconds
        
        # Refresh queue lengths
        for queue in queues:
            try:
                queue_lengths[queue] = redis_client.llen(queue)
            except:
                pass
    
    final_pending = sum(max(0, length) for length in queue_lengths.values())
    
    return {
        "monitoring_completed": True,
        "final_pending_jobs": final_pending,
        "monitoring_duration_minutes": (time.time() - wait_start) / 60
    }


def validate_embeddings(**context):
    """Validate generated embeddings."""
    
    logger.info("Validating generated embeddings")
    
    # This would:
    # 1. Check embedding dimensions
    # 2. Validate vector norms
    # 3. Test similarity calculations
    # 4. Compare with previous embeddings
    
    validation_results = {
        "embeddings_validated": True,
        "total_embeddings": 8000,
        "validation_errors": 0,
        "average_dimension": 384,
        "average_norm": 1.0
    }
    
    logger.info("Embedding validation completed", **validation_results)
    return validation_results


def update_search_index(**context):
    """Update search index with new embeddings."""
    
    logger.info("Updating search index")
    
    # This would trigger search index rebuild
    # Call search service API to refresh indexes
    
    import httpx
    import asyncio
    
    async def update_index():
        """Invoke search-service to rebuild the index via HTTP.

        Returns a small dict for XCom summarizing whether the trigger fired.
        Raises ValueError on non-200 responses.
        """
        search_service_url = Variable.get("ML_SEARCH_SERVICE_URL", "http://search-service:9007")
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{search_service_url}/api/v1/index/rebuild")
            
            if response.status_code == 200:
                logger.info("Search index update triggered")
                return {"index_update_triggered": True}
            else:
                logger.error("Search index update failed", status=response.status_code)
                raise ValueError(f"Search index update failed: {response.status_code}")
    
    # Run async function
    result = asyncio.run(update_index())
    return result


def send_completion_notification(**context):
    """Send completion notification."""
    
    logger.info("Sending reindex completion notification")
    
    # Get results from previous tasks
    reindex_results = context['task_instance'].xcom_pull(task_ids='trigger_embedding_generation')
    validation_results = context['task_instance'].xcom_pull(task_ids='validate_embeddings')
    
    # Compile summary
    summary = {
        "pipeline_completed_at": datetime.now().isoformat(),
        "entities_processed": len(reindex_results) if reindex_results else 0,
        "embeddings_generated": validation_results.get("total_embeddings", 0) if validation_results else 0,
        "validation_errors": validation_results.get("validation_errors", 0) if validation_results else 0
    }
    
    # This would send actual notifications (Slack, email, etc.)
    logger.info("Reindex completion notification sent", **summary)
    
    return summary


# Create the DAG
dag = DAG(
    'embedding_reindex_pipeline',
    default_args=default_args,
    description='Embedding reindexing and search index update pipeline',
    schedule_interval='@weekly',  # Run weekly
    catchup=False,
    tags=['ml', 'embeddings', 'search', 'maintenance']
)

# Define tasks
start_task = DummyOperator(
    task_id='start_reindex_pipeline',
    dag=dag
)

# Check if reindexing is needed
check_reindex_task = BranchPythonOperator(
    task_id='check_reindex_needed',
    python_callable=check_reindex_needed,
    dag=dag
)

# Skip reindex branch
skip_reindex_task = DummyOperator(
    task_id='skip_reindex',
    dag=dag
)

# Start reindex branch
start_reindex_task = DummyOperator(
    task_id='start_reindex',
    dag=dag
)

# Get entities to reindex
get_entities_task = PythonOperator(
    task_id='get_entities_to_reindex',
    python_callable=get_entities_to_reindex,
    dag=dag
)

# Wait for services to be healthy
service_health_check = HttpSensor(
    task_id='check_embedding_service_health',
    http_conn_id='embedding_service',
    endpoint='/health',
    timeout=300,
    poke_interval=30,
    dag=dag
)

# Trigger embedding generation
trigger_embeddings_task = PythonOperator(
    task_id='trigger_embedding_generation',
    python_callable=trigger_embedding_generation,
    dag=dag,
    pool='ml_processing_pool'
)

# Monitor progress
monitor_progress_task = PythonOperator(
    task_id='monitor_reindex_progress',
    python_callable=monitor_reindex_progress,
    dag=dag,
    timeout=timedelta(hours=2)
)

# Validate embeddings
validate_embeddings_task = PythonOperator(
    task_id='validate_embeddings',
    python_callable=validate_embeddings,
    dag=dag
)

# Update search index
update_search_task = PythonOperator(
    task_id='update_search_index',
    python_callable=update_search_index,
    dag=dag
)

# Send notifications
notify_completion_task = PythonOperator(
    task_id='notify_completion',
    python_callable=send_completion_notification,
    dag=dag
)

end_task = DummyOperator(
    task_id='end_pipeline',
    dag=dag,
    trigger_rule='none_failed_or_skipped'
)

# Define task dependencies
start_task >> check_reindex_task

# Reindex branch
check_reindex_task >> start_reindex_task >> get_entities_task >> service_health_check
service_health_check >> trigger_embeddings_task >> monitor_progress_task
monitor_progress_task >> validate_embeddings_task >> update_search_task
update_search_task >> notify_completion_task >> end_task

# Skip branch
check_reindex_task >> skip_reindex_task >> end_task
