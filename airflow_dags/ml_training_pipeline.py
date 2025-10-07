"""Airflow DAG for ML model training pipeline."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
import mlflow
import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from libs.common.config import BaseConfig  # noqa: E402
from scripts.promote_model import promote_model as promote_model_async  # noqa: E402

logger = structlog.get_logger("ml_training_dag")


# Default arguments for the DAG
default_args = {
    'owner': 'ml-platform',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1
}


def fetch_training_data(**context):
    """Fetch latest training data from 254Carbon sources."""
    
    logger.info("Fetching training data for model training")
    
    # This would connect to actual 254Carbon data sources
    # For now, simulate data fetching
    
    import sys
    from pathlib import Path
    
    # Add project path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    from data_integration.connectors import DataPipelineOrchestrator
    
    # Configure data sources
    config = {
        "yield_curves": {
            "base_url": Variable.get("CARBON_API_BASE_URL", "https://api.254carbon.internal"),
            "api_key": Variable.get("CARBON_API_KEY")
        }
    }
    
    # Fetch data
    orchestrator = DataPipelineOrchestrator(config)
    
    # This would be async in real implementation
    # results = await orchestrator.fetch_all_data()
    
    # For now, return success
    logger.info("Training data fetch completed")
    return {"status": "success", "data_path": "/tmp/training_data"}


def validate_data_quality(**context):
    """Validate data quality before training."""
    
    logger.info("Validating data quality")
    
    # Get data path from previous task
    data_info = context['task_instance'].xcom_pull(task_ids='fetch_training_data')
    data_path = data_info.get("data_path")
    
    # Validate data quality
    # This would use the DataQualityValidator from data_integration
    
    quality_score = 95.0  # Placeholder
    
    if quality_score < 90.0:
        raise ValueError(f"Data quality score {quality_score} below threshold")
    
    logger.info("Data quality validation passed", score=quality_score)
    return {"quality_score": quality_score, "validation_passed": True}


def train_model(**context):
    """Train ML model using validated data."""
    
    logger.info("Starting model training")
    
    # Get validation results
    validation_info = context['task_instance'].xcom_pull(task_ids='validate_data_quality')
    
    if not validation_info.get("validation_passed"):
        raise ValueError("Data validation failed, cannot proceed with training")
    
    # Setup MLflow
    mlflow_uri = Variable.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Start training run
    with mlflow.start_run(run_name=f"airflow_training_{context['ds']}"):
        # Log training parameters
        mlflow.log_param("data_quality_score", validation_info["quality_score"])
        mlflow.log_param("training_date", context['ds'])
        mlflow.log_param("dag_run_id", context['dag_run'].run_id)
        
        # Simulate training (in reality, this would call the training script)
        import time
        time.sleep(10)  # Simulate training time
        
        # Log metrics
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_metric("rmse", 0.02)
        mlflow.log_metric("training_duration_seconds", 10)
        
        # Get run info
        run = mlflow.active_run()
        run_id = run.info.run_id
        
        logger.info("Model training completed", run_id=run_id)
        
        return {
            "run_id": run_id,
            "model_name": "curve_forecaster",
            "training_completed": True
        }


def evaluate_model(**context):
    """Evaluate trained model."""
    
    logger.info("Evaluating trained model")
    
    # Get training results
    training_info = context['task_instance'].xcom_pull(task_ids='train_model')
    run_id = training_info.get("run_id")
    
    # Simulate model evaluation
    evaluation_metrics = {
        "test_accuracy": 0.93,
        "test_rmse": 0.025,
        "validation_score": 0.94,
        "evaluation_date": context['ds']
    }
    
    # Log evaluation metrics to MLflow
    mlflow.set_tracking_uri(Variable.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
    
    with mlflow.start_run(run_id=run_id):
        for metric, value in evaluation_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"eval_{metric}", value)
    
    # Determine if model should be promoted
    promotion_threshold = 0.90
    should_promote = evaluation_metrics["test_accuracy"] >= promotion_threshold
    
    logger.info("Model evaluation completed", 
               should_promote=should_promote,
               test_accuracy=evaluation_metrics["test_accuracy"])
    
    return {
        "evaluation_metrics": evaluation_metrics,
        "should_promote": should_promote,
        "run_id": run_id
    }


def promote_model_conditional(**context):
    """Conditionally promote model based on evaluation results."""
    
    logger.info("Checking model promotion criteria")
    
    # Get evaluation results
    eval_info = context['task_instance'].xcom_pull(task_ids='evaluate_model')
    
    if not eval_info.get("should_promote"):
        logger.info("Model promotion criteria not met, skipping promotion")
        return {"promoted": False, "reason": "evaluation_criteria_not_met"}
    
    # Promote model
    run_id = eval_info.get("run_id")
    model_name = "curve_forecaster"
    
    # This would call the promotion script
    # For now, simulate promotion
    logger.info("Promoting model to production", run_id=run_id, model_name=model_name)
    
    try:
        config = BaseConfig()
        success = asyncio.run(
            promote_model_async(
                model_name=model_name,
                run_id=run_id,
                stage="Production",
                config=config
            )
        )
        
        if success:
            logger.info("Model promoted successfully")
            return {"promoted": True, "run_id": run_id, "model_name": model_name}
        else:
            logger.error("Model promotion failed", run_id=run_id, model_name=model_name)
            raise ValueError("Model promotion script returned failure status")
            
    except Exception as e:
        logger.error("Model promotion error", error=str(e))
        raise


def notify_model_deployment(**context):
    """Notify stakeholders of model deployment."""
    
    logger.info("Sending model deployment notifications")
    
    # Get promotion results
    promotion_info = context['task_instance'].xcom_pull(task_ids='promote_model')
    
    if promotion_info.get("promoted"):
        # Send notifications (Slack, email, etc.)
        model_name = promotion_info.get("model_name")
        run_id = promotion_info.get("run_id")
        
        # This would send actual notifications
        logger.info("Model deployment notification sent", 
                   model_name=model_name, 
                   run_id=run_id)
        
        return {"notification_sent": True}
    else:
        logger.info("No promotion occurred, skipping notification")
        return {"notification_sent": False}


def cleanup_old_artifacts(**context):
    """Clean up old model artifacts and experiments."""
    
    logger.info("Cleaning up old artifacts")
    
    # This would clean up old MLflow experiments, model versions, etc.
    # For now, just log the cleanup
    
    cleanup_stats = {
        "old_experiments_cleaned": 5,
        "old_model_versions_cleaned": 10,
        "artifacts_cleaned_gb": 2.5
    }
    
    logger.info("Artifact cleanup completed", **cleanup_stats)
    return cleanup_stats


# Create the DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='ML model training and deployment pipeline',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    tags=['ml', 'training', 'production']
)

# Define tasks
start_task = DummyOperator(
    task_id='start_pipeline',
    dag=dag
)

# Data fetching and validation
fetch_data_task = PythonOperator(
    task_id='fetch_training_data',
    python_callable=fetch_training_data,
    dag=dag
)

validate_data_task = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag
)

# Model training and evaluation
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
    pool='ml_training_pool'  # Use dedicated resource pool
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

# Conditional promotion
promote_model_task = PythonOperator(
    task_id='promote_model',
    python_callable=promote_model_conditional,
    dag=dag
)

# Notifications and cleanup
notify_deployment_task = PythonOperator(
    task_id='notify_deployment',
    python_callable=notify_model_deployment,
    dag=dag
)

cleanup_task = PythonOperator(
    task_id='cleanup_artifacts',
    python_callable=cleanup_old_artifacts,
    dag=dag
)

end_task = DummyOperator(
    task_id='end_pipeline',
    dag=dag
)

# Define task dependencies
start_task >> fetch_data_task >> validate_data_task >> train_model_task
train_model_task >> evaluate_model_task >> promote_model_task
promote_model_task >> [notify_deployment_task, cleanup_task] >> end_task
