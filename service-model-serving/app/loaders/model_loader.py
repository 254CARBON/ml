"""Model loaders for different ML frameworks."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import sys
import joblib
import pickle
import torch
import numpy as np
import pandas as pd
import structlog

from libs.common.tracing import get_ml_tracer

logger = structlog.get_logger("model_loader")

# Add training modules to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    from training.curve_forecaster.model import CurveForecaster, EnsembleCurveForecaster
except ImportError:
    CurveForecaster = None
    EnsembleCurveForecaster = None


class ModelLoader:
    """Handles loading models from different frameworks and sources."""
    
    def __init__(self):
        """Initialize tracing and supported loader map."""
        self.tracer = get_ml_tracer("model-loader")
        self.supported_frameworks = {
            "sklearn": self._load_sklearn_model,
            "pytorch": self._load_pytorch_model,
            "tensorflow": self._load_tensorflow_model,
            "custom": self._load_custom_model,
            "joblib": self._load_joblib_model,
            "pickle": self._load_pickle_model
        }
    
    async def load_model(
        self,
        model_path: str,
        model_name: str,
        model_version: str,
        framework: Optional[str] = None
    ) -> Any:
        """Load a model from local path."""
        
        with self.tracer.trace_model_inference(
            model_name=model_name,
            model_version=model_version,
            input_count=0,
            operation="load_model"
        ) as span:
            try:
                # Determine framework if not specified
                if framework is None:
                    framework = await self._detect_framework(model_path)
                
                span.set_attribute("model.framework", framework)
                span.set_attribute("model.path", model_path)
                
                # Load using appropriate loader
                if framework in self.supported_frameworks:
                    loader_func = self.supported_frameworks[framework]
                    model = await loader_func(model_path, model_name, model_version)
                else:
                    # Fallback to generic loading
                    model = await self._load_generic_model(model_path, model_name, model_version)
                
                # Warm up the model
                await self._warmup_model(model, model_name, framework)
                
                logger.info(
                    "Model loaded successfully",
                    model_name=model_name,
                    model_version=model_version,
                    framework=framework,
                    model_path=model_path
                )
                
                return model
                
            except Exception as e:
                logger.error(
                    "Failed to load model",
                    model_name=model_name,
                    model_version=model_version,
                    framework=framework,
                    error=str(e)
                )
                span.set_attribute("error", str(e))
                raise
    
    async def _detect_framework(self, model_path: str) -> str:
        """Detect the ML framework used by the model."""
        try:
            model_path_obj = Path(model_path)
            
            # Check for framework-specific files
            if (model_path_obj / "model.joblib").exists():
                return "joblib"
            elif (model_path_obj / "model.pkl").exists():
                return "pickle"
            elif (model_path_obj / "model.pth").exists():
                return "pytorch"
            elif (model_path_obj / "model.py").exists():
                return "custom"
            elif (model_path_obj / "sklearn_model.pkl").exists():
                return "sklearn"
            elif (model_path_obj / "tensorflow_model").exists():
                return "tensorflow"
            
            # Default to sklearn if detection fails
            return "sklearn"
            
        except Exception as e:
            logger.warning("Framework detection failed", model_path=model_path, error=str(e))
            return "sklearn"
    
    async def _load_sklearn_model(self, model_path: str, model_name: str, model_version: str) -> Any:
        """Load scikit-learn model."""
        try:
            model_path_obj = Path(model_path)
            sklearn_path = model_path_obj / "sklearn_model.pkl"
            
            if sklearn_path.exists():
                with open(str(sklearn_path), "rb") as f:
                    model = pickle.load(f)
            else:
                # Try to load as joblib
                model = joblib.load(str(model_path_obj / "model.joblib"))
            
            logger.info("Loaded sklearn model", model_name=model_name)
            return model
        except Exception as e:
            logger.error("Failed to load sklearn model", error=str(e))
            raise
    
    async def _load_pytorch_model(self, model_path: str, model_name: str, model_version: str) -> Any:
        """Load PyTorch model."""
        try:
            model_path_obj = Path(model_path)
            model = torch.load(str(model_path_obj / "model.pth"), map_location="cpu")
            model.eval()  # Set to evaluation mode
            logger.info("Loaded PyTorch model", model_name=model_name)
            return model
        except Exception as e:
            logger.error("Failed to load PyTorch model", error=str(e))
            raise
    
    async def _load_tensorflow_model(self, model_path: str, model_name: str, model_version: str) -> Any:
        """Load TensorFlow model."""
        try:
            # Note: TensorFlow loading would require tensorflow dependency
            # For now, return a placeholder
            logger.warning("TensorFlow model loading not implemented", model_name=model_name)
            return None
        except Exception as e:
            logger.error("Failed to load TensorFlow model", error=str(e))
            raise
    
    async def _load_custom_model(self, model_path: str, model_name: str, model_version: str) -> Any:
        """Load custom model (like CurveForecaster)."""
        try:
            # Try to load as CurveForecaster first
            if CurveForecaster is not None:
                try:
                    model = CurveForecaster.load_model(model_path)
                    logger.info("Loaded CurveForecaster model", model_name=model_name)
                    return model
                except Exception as e:
                    logger.warning("Failed to load as CurveForecaster", error=str(e))
            
            # Fallback to generic loading
            model = await self._load_generic_model(model_path, model_name, model_version)
            logger.info("Loaded custom model via generic loader", model_name=model_name)
            return model
            
        except Exception as e:
            logger.error("Failed to load custom model", error=str(e))
            raise
    
    async def _load_joblib_model(self, model_path: str, model_name: str, model_version: str) -> Any:
        """Load joblib-serialized model."""
        try:
            model_path_obj = Path(model_path)
            model_path_file = model_path_obj / "model.joblib"
            model = joblib.load(str(model_path_file))
            
            logger.info("Loaded joblib model", model_name=model_name)
            return model
        except Exception as e:
            logger.error("Failed to load joblib model", error=str(e))
            raise
    
    async def _load_pickle_model(self, model_path: str, model_name: str, model_version: str) -> Any:
        """Load pickle-serialized model."""
        try:
            model_path_obj = Path(model_path)
            model_path_file = model_path_obj / "model.pkl"
            with open(str(model_path_file), "rb") as f:
                model = pickle.load(f)
            
            logger.info("Loaded pickle model", model_name=model_name)
            return model
        except Exception as e:
            logger.error("Failed to load pickle model", error=str(e))
            raise
    
    async def _load_generic_model(self, model_path: str, model_name: str, model_version: str) -> Any:
        """Load model using generic approach."""
        try:
            model_path_obj = Path(model_path)
            
            # Try different file extensions
            for ext in [".joblib", ".pkl", ".pth"]:
                model_file = model_path_obj / f"model{ext}"
                if model_file.exists():
                    if ext == ".joblib":
                        return joblib.load(str(model_file))
                    elif ext == ".pkl":
                        with open(str(model_file), "rb") as f:
                            return pickle.load(f)
                    elif ext == ".pth":
                        return torch.load(str(model_file), map_location="cpu")
            
            raise FileNotFoundError(f"No supported model file found in {model_path}")
            
        except Exception as e:
            logger.error("Failed to load generic model", error=str(e))
            raise
    
    async def _warmup_model(self, model: Any, model_name: str, framework: str):
        """Warm up the model with dummy predictions."""
        try:
            # Create dummy input based on model type
            if hasattr(model, '__class__') and 'CurveForecaster' in str(model.__class__):
                # CurveForecaster expects DataFrame
                dummy_data = {
                    "date": [pd.Timestamp.now()],
                    "rate_0.25y": [0.02],
                    "rate_1y": [0.025],
                    "rate_5y": [0.03],
                    "rate_10y": [0.035],
                    "vix": [20.0],
                    "fed_funds": [0.02]
                }
                dummy_df = pd.DataFrame(dummy_data)
                
                # Extend DataFrame to meet minimum requirements
                for i in range(getattr(model, 'lookback_days', 10)):
                    dummy_df = pd.concat([dummy_df, dummy_df.iloc[-1:]], ignore_index=True)
                
                _ = model.predict(dummy_df)
                
            elif hasattr(model, 'predict'):
                # Standard sklearn-like interface
                dummy_input = np.array([[0.02, 0.025, 0.03, 0.035, 20.0, 0.02]])
                _ = model.predict(dummy_input)
                
            elif isinstance(model, torch.nn.Module):
                # PyTorch model
                dummy_tensor = torch.randn(1, 6)  # Batch size 1, 6 features
                with torch.no_grad():
                    _ = model(dummy_tensor)
            
            logger.info("Model warmed up successfully", model_name=model_name, framework=framework)
            
        except Exception as e:
            logger.warning("Model warmup failed", model_name=model_name, error=str(e))
    
    async def preload_models(self, model_configs: List[Dict[str, Any]]):
        """Preload models on startup."""
        logger.info("Preloading models", count=len(model_configs))
        
        preload_tasks = []
        for config in model_configs:
            task = asyncio.create_task(
                self._preload_single_model(config)
            )
            preload_tasks.append(task)
        
        # Wait for all preloads to complete
        results = await asyncio.gather(*preload_tasks, return_exceptions=True)
        
        successful_loads = sum(1 for r in results if not isinstance(r, Exception))
        logger.info("Model preloading completed", 
                   total=len(model_configs), 
                   successful=successful_loads)
    
    async def _preload_single_model(self, config: Dict[str, Any]):
        """Preload a single model."""
        try:
            model_name = config["name"]
            model_version = config.get("version", "latest")
            model_path = config.get("path", f"/app/models/{model_name}/{model_version}")
            
            model = await self.load_model(model_path, model_name, model_version)
            if model:
                logger.info("Preloaded model", model_name=model_name, model_version=model_version)
            
        except Exception as e:
            logger.error("Failed to preload model", config=config, error=str(e))


class ModelInputProcessor:
    """Processes and validates model inputs."""
    
    def __init__(self):
        """Register input processing strategies by model name."""
        self.processors: Dict[str, Callable] = {
            "curve_forecaster": self._process_curve_inputs,
            "default": self._process_default_inputs
        }
    
    def process_inputs(
        self,
        inputs: List[Dict[str, Any]],
        model_name: str,
        model_type: str
    ) -> Any:
        """Process inputs for the specified model."""
        
        processor_key = model_name if model_name in self.processors else "default"
        processor = self.processors[processor_key]
        
        return processor(inputs, model_name, model_type)
    
    def _process_curve_inputs(
        self,
        inputs: List[Dict[str, Any]],
        model_name: str,
        model_type: str
    ) -> pd.DataFrame:
        """Process inputs for curve forecasting models."""
        
        # Convert inputs to DataFrame format expected by CurveForecaster
        df_data = []
        
        for inp in inputs:
            row = {"date": pd.Timestamp.now()}
            
            # Add rate data
            for key, value in inp.items():
                if key.startswith(("rate_", "forward_", "spot_")):
                    row[key] = float(value)
                elif key in ["vix", "fed_funds", "unemployment", "inflation"]:
                    row[key] = float(value)
                elif key in ["inventory", "temperature", "production"]:
                    row[key] = float(value)
                else:
                    # Store other fields as metadata
                    row[key] = value
            
            df_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        # Extend DataFrame to meet minimum lookback requirements
        # This is a simplified approach - in production, you'd have historical data
        min_rows = 50  # Minimum rows needed for feature generation
        while len(df) < min_rows:
            # Duplicate the last row with slight variations
            last_row = df.iloc[-1].copy()
            
            # Add small random variations to rates
            for col in df.columns:
                if col.startswith(("rate_", "forward_", "spot_")):
                    last_row[col] += np.random.normal(0, 0.0001)
                elif col in ["vix", "fed_funds", "unemployment", "inflation"]:
                    last_row[col] += np.random.normal(0, 0.01)
            
            # Update date
            last_row["date"] = last_row["date"] - pd.Timedelta(days=1)
            
            df = pd.concat([pd.DataFrame([last_row]), df], ignore_index=True)
        
        return df
    
    def _process_default_inputs(
        self,
        inputs: List[Dict[str, Any]],
        model_name: str,
        model_type: str
    ) -> np.ndarray:
        """Process inputs for standard sklearn-like models."""
        
        if isinstance(inputs[0], dict):
            # Convert dict inputs to array
            input_array = np.array([list(inp.values()) for inp in inputs])
        else:
            input_array = np.array(inputs)
        
        return input_array


class ModelOutputProcessor:
    """Processes and formats model outputs."""
    
    def __init__(self):
        """Register output processing strategies by model name."""
        self.processors: Dict[str, Callable] = {
            "curve_forecaster": self._process_curve_outputs,
            "default": self._process_default_outputs
        }
    
    def process_outputs(
        self,
        outputs: Any,
        model_name: str,
        model_type: str,
        input_count: int
    ) -> List[Any]:
        """Process outputs from the specified model."""
        
        processor_key = model_name if model_name in self.processors else "default"
        processor = self.processors[processor_key]
        
        return processor(outputs, model_name, model_type, input_count)
    
    def _process_curve_outputs(
        self,
        outputs: Any,
        model_name: str,
        model_type: str,
        input_count: int
    ) -> List[Dict[str, Any]]:
        """Process outputs from curve forecasting models."""
        
        if isinstance(outputs, np.ndarray):
            # Convert numpy array to structured output
            processed_outputs = []
            
            for i in range(len(outputs)):
                prediction = outputs[i]
                
                # Structure the prediction
                structured_pred = {
                    "forecast_horizons": {},
                    "metadata": {
                        "model_name": model_name,
                        "model_type": model_type,
                        "prediction_timestamp": pd.Timestamp.now().isoformat()
                    }
                }
                
                # Assuming prediction contains forecasts for different horizons
                # This would be customized based on actual model output structure
                if len(prediction) >= 5:
                    for h in range(1, 6):  # 5 forecast horizons
                        if h - 1 < len(prediction):
                            structured_pred["forecast_horizons"][f"h{h}"] = float(prediction[h - 1])
                
                processed_outputs.append(structured_pred)
            
            return processed_outputs
        
        else:
            # Fallback to default processing
            return self._process_default_outputs(outputs, model_name, model_type, input_count)
    
    def _process_default_outputs(
        self,
        outputs: Any,
        model_name: str,
        model_type: str,
        input_count: int
    ) -> List[Any]:
        """Process outputs from standard models."""
        
        if isinstance(outputs, np.ndarray):
            return outputs.tolist()
        elif isinstance(outputs, list):
            return outputs
        else:
            # Single prediction
            return [outputs] * input_count


def create_model_loader() -> ModelLoader:
    """Create a model loader instance."""
    return ModelLoader()


def create_input_processor() -> ModelInputProcessor:
    """Create a model input processor."""
    return ModelInputProcessor()


def create_output_processor() -> ModelOutputProcessor:
    """Create a model output processor."""
    return ModelOutputProcessor()