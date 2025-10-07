"""Model evaluation and validation script."""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import structlog

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from training.curve_forecaster.model import CurveForecaster
from training.curve_forecaster.data_generator import CurveDataGenerator
from libs.common.config import BaseConfig
from libs.common.logging import configure_logging

logger = structlog.get_logger("evaluate")


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model: CurveForecaster):
        """Wrap a trained ``CurveForecaster`` for evaluation utilities."""
        self.model = model
    
    def evaluate_forecasting_accuracy(self, test_df: pd.DataFrame) -> dict:
        """Evaluate forecasting accuracy across different horizons."""
        logger.info("Evaluating forecasting accuracy")
        
        # Create features and targets
        feature_df = self.model._create_features(test_df)
        target_df = self.model._create_targets(test_df, feature_df)
        
        # Make predictions
        predictions = self.model.predict(test_df)
        
        # Calculate metrics by horizon
        horizon_metrics = {}
        
        for horizon in range(1, self.model.forecast_horizon + 1):
            horizon_cols = [col for col in target_df.columns if col.endswith(f"_h{horizon}")]
            
            if horizon_cols:
                y_true = target_df[horizon_cols].values
                y_pred = predictions[:, [target_df.columns.get_loc(col) for col in horizon_cols]]
                
                horizon_metrics[f"h{horizon}"] = {
                    "mse": mean_squared_error(y_true, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "mae": mean_absolute_error(y_true, y_pred),
                    "r2": r2_score(y_true, y_pred)
                }
        
        return horizon_metrics
    
    def evaluate_curve_shape_preservation(self, test_df: pd.DataFrame) -> dict:
        """Evaluate how well the model preserves curve shapes."""
        logger.info("Evaluating curve shape preservation")
        
        # Get predictions
        predictions = self.model.predict(test_df)
        
        # Create targets
        feature_df = self.model._create_features(test_df)
        target_df = self.model._create_targets(test_df, feature_df)
        
        shape_metrics = {}
        
        # Analyze curve characteristics for each horizon
        for horizon in range(1, self.model.forecast_horizon + 1):
            horizon_cols = [col for col in target_df.columns if col.endswith(f"_h{horizon}")]
            rate_cols = [col for col in horizon_cols if col.startswith("rate_")]
            
            if len(rate_cols) >= 3:
                # Extract actual and predicted curves
                actual_curves = target_df[rate_cols].values
                pred_indices = [target_df.columns.get_loc(col) for col in rate_cols]
                pred_curves = predictions[:, pred_indices]
                
                # Calculate curve characteristics
                actual_levels = np.mean(actual_curves, axis=1)
                pred_levels = np.mean(pred_curves, axis=1)
                
                actual_slopes = actual_curves[:, -1] - actual_curves[:, 0]
                pred_slopes = pred_curves[:, -1] - pred_curves[:, 0]
                
                if actual_curves.shape[1] >= 3:
                    mid_idx = actual_curves.shape[1] // 2
                    actual_curvature = actual_curves[:, mid_idx] - (actual_curves[:, 0] + actual_curves[:, -1]) / 2
                    pred_curvature = pred_curves[:, mid_idx] - (pred_curves[:, 0] + pred_curves[:, -1]) / 2
                else:
                    actual_curvature = pred_curvature = np.zeros(len(actual_curves))
                
                shape_metrics[f"h{horizon}"] = {
                    "level_correlation": np.corrcoef(actual_levels, pred_levels)[0, 1],
                    "slope_correlation": np.corrcoef(actual_slopes, pred_slopes)[0, 1],
                    "curvature_correlation": np.corrcoef(actual_curvature, pred_curvature)[0, 1],
                    "level_rmse": np.sqrt(mean_squared_error(actual_levels, pred_levels)),
                    "slope_rmse": np.sqrt(mean_squared_error(actual_slopes, pred_slopes)),
                    "curvature_rmse": np.sqrt(mean_squared_error(actual_curvature, pred_curvature))
                }
        
        return shape_metrics
    
    def evaluate_directional_accuracy(self, test_df: pd.DataFrame) -> dict:
        """Evaluate directional accuracy of predictions."""
        logger.info("Evaluating directional accuracy")
        
        # Create features and targets
        feature_df = self.model._create_features(test_df)
        target_df = self.model._create_targets(test_df, feature_df)
        
        # Get current values for comparison
        current_values = {}
        for _, row in feature_df.iterrows():
            target_idx = int(row["target_idx"])
            current_row = test_df.iloc[target_idx - 1]  # Previous day
            
            for col in test_df.columns:
                if col.startswith(("rate_", "forward_", "spot_")) and col != "date":
                    if col not in current_values:
                        current_values[col] = []
                    current_values[col].append(current_row[col])
        
        # Make predictions
        predictions = self.model.predict(test_df)
        
        directional_metrics = {}
        
        for horizon in range(1, self.model.forecast_horizon + 1):
            horizon_cols = [col for col in target_df.columns if col.endswith(f"_h{horizon}")]
            
            correct_directions = []
            
            for col in horizon_cols:
                base_col = col.split("_h")[0]
                if base_col in current_values:
                    current_vals = np.array(current_values[base_col])
                    actual_vals = target_df[col].values
                    pred_vals = predictions[:, target_df.columns.get_loc(col)]
                    
                    # Calculate direction accuracy
                    actual_direction = np.sign(actual_vals - current_vals)
                    pred_direction = np.sign(pred_vals - current_vals)
                    
                    correct = (actual_direction == pred_direction).mean()
                    correct_directions.append(correct)
            
            if correct_directions:
                directional_metrics[f"h{horizon}"] = {
                    "directional_accuracy": np.mean(correct_directions),
                    "directional_accuracy_std": np.std(correct_directions)
                }
        
        return directional_metrics
    
    def create_evaluation_plots(self, test_df: pd.DataFrame, output_dir: str = "evaluation_plots"):
        """Create evaluation plots."""
        logger.info("Creating evaluation plots")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get predictions and targets
        feature_df = self.model._create_features(test_df)
        target_df = self.model._create_targets(test_df, feature_df)
        predictions = self.model.predict(test_df)
        
        # Plot 1: Prediction vs Actual scatter plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, horizon in enumerate(range(1, min(5, self.model.forecast_horizon + 1))):
            horizon_cols = [col for col in target_df.columns if col.endswith(f"_h{horizon}")]
            
            if horizon_cols and i < len(axes):
                y_true = target_df[horizon_cols].values.flatten()
                y_pred = predictions[:, [target_df.columns.get_loc(col) for col in horizon_cols]].flatten()
                
                axes[i].scatter(y_true, y_pred, alpha=0.5)
                axes[i].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
                axes[i].set_xlabel("Actual")
                axes[i].set_ylabel("Predicted")
                axes[i].set_title(f"Horizon {horizon} Days")
                
                # Add R² score
                r2 = r2_score(y_true, y_pred)
                axes[i].text(0.05, 0.95, f"R² = {r2:.3f}", transform=axes[i].transAxes, 
                           bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prediction_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Feature importance
        feature_importance = self.model.get_feature_importance()
        if feature_importance:
            top_features = dict(list(feature_importance.items())[:20])
            
            plt.figure(figsize=(12, 8))
            features = list(top_features.keys())
            importances = list(top_features.values())
            
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel("Feature Importance")
            plt.title("Top 20 Feature Importances")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 3: Residuals analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall residuals
        y_true_all = target_df.values.flatten()
        y_pred_all = predictions.flatten()
        residuals = y_true_all - y_pred_all
        
        axes[0].scatter(y_pred_all, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel("Predicted Values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predicted")
        
        axes[1].hist(residuals, bins=50, alpha=0.7)
        axes[1].set_xlabel("Residuals")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Residuals Distribution")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/residuals_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Evaluation plots saved", output_dir=output_dir)
    
    def generate_evaluation_report(self, test_df: pd.DataFrame) -> dict:
        """Generate comprehensive evaluation report."""
        logger.info("Generating evaluation report")
        
        report = {
            "model_info": {
                "model_type": self.model.model_type,
                "lookback_days": self.model.lookback_days,
                "forecast_horizon": self.model.forecast_horizon,
                "n_features": len(self.model.feature_names) if self.model.feature_names else 0,
                "n_targets": len(self.model.target_names) if self.model.target_names else 0
            }
        }
        
        # Basic accuracy metrics
        report["accuracy_metrics"] = self.evaluate_forecasting_accuracy(test_df)
        
        # Curve shape preservation
        report["shape_metrics"] = self.evaluate_curve_shape_preservation(test_df)
        
        # Directional accuracy
        report["directional_metrics"] = self.evaluate_directional_accuracy(test_df)
        
        # Feature importance
        report["feature_importance"] = self.model.get_feature_importance()
        
        return report


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate curve forecasting model")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--test-data-path", help="Path to test data")
    parser.add_argument("--curve-type", default="yield", choices=["yield", "commodity", "fx"])
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory for results")
    parser.add_argument("--create-plots", action="store_true", help="Create evaluation plots")
    
    args = parser.parse_args()
    
    # Setup logging
    configure_logging("evaluate", "INFO", "json")
    
    # Load model
    logger.info("Loading model", path=args.model_path)
    model = CurveForecaster.load_model(args.model_path)
    
    # Prepare test data
    if args.test_data_path and os.path.exists(args.test_data_path):
        test_df = pd.read_parquet(args.test_data_path)
    else:
        logger.info("Generating test data", curve_type=args.curve_type)
        generator = CurveDataGenerator(seed=123)  # Different seed for test data
        
        if args.curve_type == "yield":
            test_df = generator.generate_yield_curve_data(n_samples=500)
        elif args.curve_type == "commodity":
            test_df = generator.generate_commodity_curve_data("NG", n_samples=500)
        elif args.curve_type == "fx":
            test_df = generator.generate_fx_curve_data("EURUSD", n_samples=500)
    
    # Create evaluator
    evaluator = ModelEvaluator(model)
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(test_df)
    
    # Save report
    os.makedirs(args.output_dir, exist_ok=True)
    import json
    with open(f"{args.output_dir}/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create plots if requested
    if args.create_plots:
        evaluator.create_evaluation_plots(test_df, f"{args.output_dir}/plots")
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    for horizon, metrics in report["accuracy_metrics"].items():
        print(f"{horizon}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    logger.info("Evaluation completed", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
