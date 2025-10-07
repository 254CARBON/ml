"""Training script for curve forecasting model."""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
import structlog

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from training.curve_forecaster.data_generator import CurveDataGenerator
from training.curve_forecaster.model import CurveForecaster, EnsembleCurveForecaster
from libs.common.config import BaseConfig
from libs.common.logging import configure_logging

logger = structlog.get_logger("train")


def setup_mlflow(config: BaseConfig):
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri(config.ml_mlflow_tracking_uri)
    
    # Create experiment if it doesn't exist
    experiment_name = "curve_forecasting"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
    except Exception as e:
        logger.warning("Could not create MLflow experiment", error=str(e))
    
    mlflow.set_experiment(experiment_name)


def prepare_data(data_path: str = None, curve_type: str = "yield") -> pd.DataFrame:
    """Prepare training data."""
    if data_path and os.path.exists(data_path):
        logger.info("Loading data from file", path=data_path)
        df = pd.read_parquet(data_path)
    else:
        logger.info("Generating synthetic data", curve_type=curve_type)
        generator = CurveDataGenerator()
        
        if curve_type == "yield":
            df = generator.generate_yield_curve_data(n_samples=2000)
        elif curve_type == "commodity":
            df = generator.generate_commodity_curve_data("NG", n_samples=2000)
        elif curve_type == "fx":
            df = generator.generate_fx_curve_data("EURUSD", n_samples=2000)
        else:
            raise ValueError(f"Unknown curve type: {curve_type}")
    
    logger.info("Data prepared", shape=df.shape, curve_type=curve_type)
    return df


def train_single_model(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    lookback_days: int = 30,
    forecast_horizon: int = 5,
    **model_kwargs
) -> CurveForecaster:
    """Train a single curve forecasting model."""
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=False)
    
    logger.info("Data split", 
                train_size=len(train_df), 
                val_size=len(val_df), 
                test_size=len(test_df))
    
    # Create and train model
    model = CurveForecaster(
        model_type=model_type,
        lookback_days=lookback_days,
        forecast_horizon=forecast_horizon,
        **model_kwargs
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"curve_forecaster_{model_type}"):
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("lookback_days", lookback_days)
        mlflow.log_param("forecast_horizon", forecast_horizon)
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("val_samples", len(val_df))
        mlflow.log_param("test_samples", len(test_df))
        
        for key, value in model_kwargs.items():
            mlflow.log_param(key, value)
        
        # Train model
        logger.info("Training model", model_type=model_type)
        model.fit(train_df)
        
        # Evaluate on validation set
        val_metrics = model.evaluate(val_df)
        for metric, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric}", value)
        
        # Evaluate on test set
        test_metrics = model.evaluate(test_df)
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)
        
        # Log feature importance
        feature_importance = model.get_feature_importance()
        if feature_importance:
            # Log top 20 features
            top_features = dict(list(feature_importance.items())[:20])
            for feature, importance in top_features.items():
                mlflow.log_metric(f"feature_importance_{feature}", importance)
        
        # Save model
        model_path = f"models/curve_forecaster_{model_type}"
        os.makedirs(model_path, exist_ok=True)
        model_file = f"{model_path}/model.joblib"
        model.save_model(model_file)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="curve_forecaster"
        )
        
        # Log model signature
        sample_input = model._create_features(train_df.head(100))
        sample_input = sample_input.drop(columns=["target_idx"])
        sample_input = sample_input[model.feature_names]
        
        signature = mlflow.models.infer_signature(
            sample_input.values,
            model.predict(train_df.head(100))
        )
        
        mlflow.models.log_model(
            model,
            "model_with_signature",
            signature=signature
        )
        
        logger.info("Model training completed",
                   val_rmse=val_metrics["rmse"],
                   test_rmse=test_metrics["rmse"],
                   val_r2=val_metrics["r2"],
                   test_r2=test_metrics["r2"])
    
    return model


def train_ensemble_model(df: pd.DataFrame) -> EnsembleCurveForecaster:
    """Train an ensemble of curve forecasting models."""
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=False)
    
    # Define ensemble models
    model_configs = [
        {
            "model_type": "random_forest",
            "lookback_days": 30,
            "forecast_horizon": 5,
            "n_estimators": 100,
            "max_depth": 10
        },
        {
            "model_type": "gradient_boosting",
            "lookback_days": 30,
            "forecast_horizon": 5,
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        },
        {
            "model_type": "ridge",
            "lookback_days": 30,
            "forecast_horizon": 5,
            "alpha": 1.0
        },
        {
            "model_type": "random_forest",
            "lookback_days": 45,
            "forecast_horizon": 5,
            "n_estimators": 150,
            "max_depth": 12
        }
    ]
    
    with mlflow.start_run(run_name="curve_forecaster_ensemble"):
        # Log ensemble parameters
        mlflow.log_param("model_type", "ensemble")
        mlflow.log_param("n_models", len(model_configs))
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("val_samples", len(val_df))
        mlflow.log_param("test_samples", len(test_df))
        
        # Train ensemble
        logger.info("Training ensemble model")
        ensemble = EnsembleCurveForecaster(model_configs)
        ensemble.fit(train_df, val_df)
        
        # Evaluate ensemble
        val_metrics = ensemble.evaluate(val_df)
        for metric, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric}", value)
        
        test_metrics = ensemble.evaluate(test_df)
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)
        
        # Log ensemble weights
        for i, weight in enumerate(ensemble.weights):
            mlflow.log_metric(f"model_{i}_weight", weight)
        
        # Save ensemble
        ensemble_path = "models/curve_forecaster_ensemble"
        os.makedirs(ensemble_path, exist_ok=True)
        
        # Save individual models
        for i, model in enumerate(ensemble.models):
            model_file = f"{ensemble_path}/model_{i}.joblib"
            model.save_model(model_file)
        
        # Log ensemble to MLflow
        mlflow.sklearn.log_model(
            sk_model=ensemble,
            artifact_path="ensemble_model",
            registered_model_name="curve_forecaster_ensemble"
        )
        
        logger.info("Ensemble training completed",
                   val_rmse=val_metrics["rmse"],
                   test_rmse=test_metrics["rmse"],
                   val_r2=val_metrics["r2"],
                   test_r2=test_metrics["r2"])
    
    return ensemble


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train curve forecasting model")
    parser.add_argument("--data-path", help="Path to training data")
    parser.add_argument("--curve-type", default="yield", choices=["yield", "commodity", "fx"])
    parser.add_argument("--model-type", default="random_forest", 
                       choices=["random_forest", "gradient_boosting", "ridge", "ensemble"])
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--forecast-horizon", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Setup logging
    configure_logging("train", "INFO", "json")
    
    # Load configuration
    config = BaseConfig()
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Prepare data
    df = prepare_data(args.data_path, args.curve_type)
    
    # Train model
    if args.model_type == "ensemble":
        model = train_ensemble_model(df)
    else:
        model_kwargs = {}
        if args.model_type in ["random_forest", "gradient_boosting"]:
            model_kwargs["n_estimators"] = args.n_estimators
            model_kwargs["max_depth"] = args.max_depth
        if args.model_type == "gradient_boosting":
            model_kwargs["learning_rate"] = args.learning_rate
        if args.model_type == "ridge":
            model_kwargs["alpha"] = args.alpha
        
        model = train_single_model(
            df,
            model_type=args.model_type,
            lookback_days=args.lookback_days,
            forecast_horizon=args.forecast_horizon,
            **model_kwargs
        )
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
