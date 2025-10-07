"""Curve forecasting model implementation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import structlog

logger = structlog.get_logger("curve_forecaster")


class CurveForecaster(BaseEstimator, RegressorMixin):
    """Multi-output curve forecasting model."""
    
    def __init__(
        self,
        model_type: str = "random_forest",
        lookback_days: int = 30,
        forecast_horizon: int = 5,
        **model_kwargs
    ):
        """Initialize a curve forecaster.

        Parameters
        - model_type: One of ``random_forest``, ``gradient_boosting``, ``ridge``
        - lookback_days: Number of prior days to build features from
        - forecast_horizon: Number of future steps to forecast for each target
        - model_kwargs: Extra regressor-specific hyperparameters
        """
        self.model_type = model_type
        self.lookback_days = lookback_days
        self.forecast_horizon = forecast_horizon
        self.model_kwargs = model_kwargs
        
        # Initialize components
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.target_names = None
        self.is_fitted = False
    
    def _create_model(self):
        """Create the underlying model."""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.model_kwargs.get("n_estimators", 100),
                max_depth=self.model_kwargs.get("max_depth", 10),
                min_samples_split=self.model_kwargs.get("min_samples_split", 5),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.model_kwargs.get("n_estimators", 100),
                max_depth=self.model_kwargs.get("max_depth", 6),
                learning_rate=self.model_kwargs.get("learning_rate", 0.1),
                random_state=42
            )
        elif self.model_type == "ridge":
            return Ridge(
                alpha=self.model_kwargs.get("alpha", 1.0),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw curve data."""
        features = []
        
        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)
        
        for i in range(self.lookback_days, len(df)):
            feature_row = {}
            
            # Historical curve data (lookback window)
            for lookback in range(self.lookback_days):
                idx = i - lookback - 1
                row = df.iloc[idx]
                
                # Add curve points
                for col in df.columns:
                    if col.startswith(("rate_", "forward_", "spot_")) and col != "date":
                        feature_row[f"{col}_lag_{lookback}"] = row[col]
                
                # Add market features
                for col in ["vix", "fed_funds", "unemployment", "inflation", 
                           "inventory", "temperature", "production", "volatility_1m"]:
                    if col in row:
                        feature_row[f"{col}_lag_{lookback}"] = row[col]
            
            # Technical indicators
            current_idx = i - 1
            window_data = df.iloc[current_idx - min(10, current_idx):current_idx + 1]
            
            # Add moving averages and volatilities for key rates
            for col in df.columns:
                if col.startswith(("rate_", "forward_", "spot_")) and col != "date":
                    if len(window_data) > 1:
                        values = window_data[col].values
                        feature_row[f"{col}_ma_5"] = np.mean(values[-5:]) if len(values) >= 5 else values[-1]
                        feature_row[f"{col}_ma_10"] = np.mean(values[-10:]) if len(values) >= 10 else values[-1]
                        feature_row[f"{col}_vol_5"] = np.std(values[-5:]) if len(values) >= 5 else 0
                        feature_row[f"{col}_vol_10"] = np.std(values[-10:]) if len(values) >= 10 else 0
            
            # Curve shape features
            rate_cols = [col for col in df.columns if col.startswith("rate_") and col != "date"]
            if len(rate_cols) >= 3:
                current_row = df.iloc[current_idx]
                rates = [current_row[col] for col in sorted(rate_cols)]
                
                # Curve level, slope, curvature
                feature_row["curve_level"] = np.mean(rates)
                feature_row["curve_slope"] = rates[-1] - rates[0]  # Long - Short
                if len(rates) >= 3:
                    feature_row["curve_curvature"] = rates[len(rates)//2] - (rates[0] + rates[-1]) / 2
            
            # Time features
            date = df.iloc[i]["date"]
            feature_row["day_of_week"] = date.dayofweek
            feature_row["day_of_month"] = date.day
            feature_row["month"] = date.month
            feature_row["quarter"] = date.quarter
            feature_row["day_of_year"] = date.dayofyear
            
            feature_row["target_idx"] = i
            features.append(feature_row)
        
        feature_df = pd.DataFrame(features)
        logger.info("Created features", shape=feature_df.shape, lookback_days=self.lookback_days)
        return feature_df
    
    def _create_targets(self, df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for forecasting."""
        targets = []
        
        for _, row in feature_df.iterrows():
            target_idx = int(row["target_idx"])
            target_row = {}
            
            # Forecast horizon targets
            for horizon in range(1, self.forecast_horizon + 1):
                future_idx = target_idx + horizon
                if future_idx < len(df):
                    future_row = df.iloc[future_idx]
                    
                    # Add future curve points as targets
                    for col in df.columns:
                        if col.startswith(("rate_", "forward_", "spot_")) and col != "date":
                            target_row[f"{col}_h{horizon}"] = future_row[col]
                else:
                    # If we don't have future data, use last available
                    last_row = df.iloc[-1]
                    for col in df.columns:
                        if col.startswith(("rate_", "forward_", "spot_")) and col != "date":
                            target_row[f"{col}_h{horizon}"] = last_row[col]
            
            targets.append(target_row)
        
        target_df = pd.DataFrame(targets)
        logger.info("Created targets", shape=target_df.shape, forecast_horizon=self.forecast_horizon)
        return target_df
    
    def fit(self, df: pd.DataFrame) -> "CurveForecaster":
        """Fit the curve forecasting model."""
        logger.info("Fitting curve forecaster", samples=len(df))
        
        # Create features and targets
        feature_df = self._create_features(df)
        target_df = self._create_targets(df, feature_df)
        
        # Remove target_idx from features
        X = feature_df.drop(columns=["target_idx"])
        y = target_df
        
        # Store feature and target names
        self.feature_names = list(X.columns)
        self.target_names = list(y.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        train_metrics = self._calculate_metrics(y, y_pred)
        
        logger.info("Model fitted successfully", 
                   features=len(self.feature_names),
                   targets=len(self.target_names),
                   **train_metrics)
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create features
        feature_df = self._create_features(df)
        X = feature_df.drop(columns=["target_idx"])
        
        # Ensure feature consistency
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0
        
        X = X[self.feature_names]  # Reorder columns
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def _calculate_metrics(self, y_true: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Overall metrics
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2"] = r2_score(y_true, y_pred)
        
        # Per-target metrics (average across horizons)
        target_groups = {}
        for col in y_true.columns:
            base_name = col.split("_h")[0]
            if base_name not in target_groups:
                target_groups[base_name] = []
            target_groups[base_name].append(col)
        
        for base_name, cols in target_groups.items():
            y_true_group = y_true[cols]
            y_pred_group = y_pred[:, [y_true.columns.get_loc(col) for col in cols]]
            
            metrics[f"{base_name}_mse"] = mean_squared_error(y_true_group, y_pred_group)
            metrics[f"{base_name}_mae"] = mean_absolute_error(y_true_group, y_pred_group)
        
        return metrics
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model on test data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Create features and targets
        feature_df = self._create_features(df)
        target_df = self._create_targets(df, feature_df)
        
        # Make predictions
        predictions = self.predict(df)
        
        # Calculate metrics
        metrics = self._calculate_metrics(target_df, predictions)
        
        logger.info("Model evaluation completed", **metrics)
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted or not hasattr(self.model, "feature_importances_"):
            return {}
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, path: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "model_type": self.model_type,
            "lookback_days": self.lookback_days,
            "forecast_horizon": self.forecast_horizon,
            "model_kwargs": self.model_kwargs
        }
        
        joblib.dump(model_data, path)
        logger.info("Model saved", path=path)
    
    @classmethod
    def load_model(cls, path: str) -> "CurveForecaster":
        """Load a trained model."""
        model_data = joblib.load(path)
        
        # Create instance
        instance = cls(
            model_type=model_data["model_type"],
            lookback_days=model_data["lookback_days"],
            forecast_horizon=model_data["forecast_horizon"],
            **model_data["model_kwargs"]
        )
        
        # Restore state
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.feature_names = model_data["feature_names"]
        instance.target_names = model_data["target_names"]
        instance.is_fitted = True
        
        logger.info("Model loaded", path=path)
        return instance


class EnsembleCurveForecaster:
    """Ensemble of curve forecasting models."""
    
    def __init__(self, models: List[Dict[str, Any]]):
        """Initialize an ensemble of curve forecasters.

        Parameters
        - models: List of config dicts passed to ``CurveForecaster(**config)``
        """
        self.model_configs = models
        self.models = []
        self.weights = None
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None):
        """Fit ensemble of models."""
        logger.info("Fitting ensemble", n_models=len(self.model_configs))
        
        self.models = []
        validation_scores = []
        
        for i, config in enumerate(self.model_configs):
            logger.info(f"Fitting model {i+1}/{len(self.model_configs)}", config=config)
            
            model = CurveForecaster(**config)
            model.fit(df)
            self.models.append(model)
            
            # Calculate validation score for weighting
            if validation_df is not None:
                metrics = model.evaluate(validation_df)
                validation_scores.append(1.0 / (1.0 + metrics["rmse"]))  # Higher score = lower RMSE
            else:
                validation_scores.append(1.0)  # Equal weights
        
        # Calculate ensemble weights (inverse of validation error)
        total_score = sum(validation_scores)
        self.weights = [score / total_score for score in validation_scores]
        
        self.is_fitted = True
        logger.info("Ensemble fitted", weights=self.weights)
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(df)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate ensemble on test data."""
        predictions = self.predict(df)
        
        # Create targets for evaluation
        feature_df = self.models[0]._create_features(df)
        target_df = self.models[0]._create_targets(df, feature_df)
        
        metrics = self.models[0]._calculate_metrics(target_df, predictions)
        logger.info("Ensemble evaluation completed", **metrics)
        return metrics
