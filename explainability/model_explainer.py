"""Model explainability using SHAP and LIME."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import structlog

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import shap
    import lime
    from lime import lime_tabular
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False
    shap = None
    lime = None

from libs.common.logging import configure_logging

logger = structlog.get_logger("model_explainer")


class ModelExplainer:
    """Base class for model explainability."""
    
    def __init__(self, model: Any, feature_names: List[str]):
        """Base initializer.

        Parameters
        - model: Trained model exposing ``predict`` or equivalent
        - feature_names: Ordered list of input feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
        if not EXPLAINABILITY_AVAILABLE:
            logger.warning("SHAP and LIME not available - install with: pip install shap lime")
    
    def explain_prediction(
        self,
        input_data: np.ndarray,
        explanation_type: str = "shap"
    ) -> Dict[str, Any]:
        """Explain a single prediction."""
        raise NotImplementedError
    
    def explain_batch(
        self,
        input_data: np.ndarray,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """Explain a batch of predictions."""
        raise NotImplementedError


class SHAPExplainer(ModelExplainer):
    """SHAP-based model explainer."""
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
        explainer_type: str = "auto"
    ):
        """Initialize a SHAP explainer.

        Parameters
        - model: Trained model to explain
        - feature_names: Ordered list of input feature names
        - background_data: Optional background sample for Kernel/Linear explainers
        - explainer_type: One of ``auto``, ``tree``, ``linear``, or ``kernel``
        """
        super().__init__(model, feature_names)
        
        if not EXPLAINABILITY_AVAILABLE:
            return
        
        self.background_data = background_data
        self.explainer_type = explainer_type
        
        # Initialize SHAP explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type."""
        
        if not EXPLAINABILITY_AVAILABLE:
            return
        
        try:
            if self.explainer_type == "tree" or hasattr(self.model, "estimators_"):
                # Tree-based models (Random Forest, Gradient Boosting)
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized SHAP TreeExplainer")
                
            elif self.explainer_type == "linear" or hasattr(self.model, "coef_"):
                # Linear models
                self.explainer = shap.LinearExplainer(self.model, self.background_data)
                logger.info("Initialized SHAP LinearExplainer")
                
            elif self.explainer_type == "kernel" or self.explainer_type == "auto":
                # Model-agnostic kernel explainer
                if self.background_data is None:
                    logger.warning("Background data recommended for KernelExplainer")
                    self.background_data = np.random.randn(100, len(self.feature_names))
                
                self.explainer = shap.KernelExplainer(self.model.predict, self.background_data)
                logger.info("Initialized SHAP KernelExplainer")
                
            else:
                logger.error("Unknown explainer type", explainer_type=self.explainer_type)
                
        except Exception as e:
            logger.error("Failed to initialize SHAP explainer", error=str(e))
    
    def explain_prediction(
        self,
        input_data: np.ndarray,
        explanation_type: str = "shap"
    ) -> Dict[str, Any]:
        """Explain a single prediction using SHAP."""
        
        if not EXPLAINABILITY_AVAILABLE or self.explainer is None:
            return {"error": "SHAP explainer not available"}
        
        try:
            # Ensure input is 2D
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(input_data)
            
            # Handle multi-output models
            if isinstance(shap_values, list):
                # Multi-output model - take first output for simplicity
                shap_values = shap_values[0]
            
            # Get base value
            if hasattr(self.explainer, "expected_value"):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, np.ndarray):
                    expected_value = expected_value[0]
            else:
                expected_value = 0.0
            
            # Get model prediction
            prediction = self.model.predict(input_data)
            if isinstance(prediction, np.ndarray):
                prediction = prediction[0] if prediction.ndim > 0 else prediction
            
            # Create explanation
            explanation = {
                "prediction": float(prediction),
                "expected_value": float(expected_value),
                "shap_values": shap_values[0].tolist() if shap_values.ndim > 1 else shap_values.tolist(),
                "feature_names": self.feature_names,
                "feature_values": input_data[0].tolist(),
                "feature_importance": dict(zip(
                    self.feature_names,
                    np.abs(shap_values[0] if shap_values.ndim > 1 else shap_values).tolist()
                )),
                "explanation_type": "shap",
                "timestamp": time.time()
            }
            
            # Add top contributing features
            feature_contributions = list(zip(
                self.feature_names,
                shap_values[0] if shap_values.ndim > 1 else shap_values,
                input_data[0]
            ))
            
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            explanation["top_features"] = [
                {
                    "feature": name,
                    "value": float(value),
                    "contribution": float(contrib),
                    "contribution_percentage": float(abs(contrib) / np.sum(np.abs(shap_values)) * 100)
                }
                for name, contrib, value in feature_contributions[:10]
            ]
            
            logger.info("SHAP explanation generated",
                       prediction=prediction,
                       top_feature=feature_contributions[0][0])
            
            return explanation
            
        except Exception as e:
            logger.error("SHAP explanation failed", error=str(e))
            return {"error": str(e)}
    
    def explain_batch(
        self,
        input_data: np.ndarray,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """Explain a batch of predictions using SHAP."""
        
        if not EXPLAINABILITY_AVAILABLE or self.explainer is None:
            return {"error": "SHAP explainer not available"}
        
        try:
            # Limit sample size for performance
            if len(input_data) > max_samples:
                indices = np.random.choice(len(input_data), max_samples, replace=False)
                sample_data = input_data[indices]
            else:
                sample_data = input_data
            
            # Get SHAP values for batch
            shap_values = self.explainer.shap_values(sample_data)
            
            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Calculate feature importance across batch
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            feature_importance = dict(zip(self.feature_names, mean_abs_shap))
            
            # Sort features by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            batch_explanation = {
                "sample_size": len(sample_data),
                "feature_importance": feature_importance,
                "top_features": [
                    {"feature": name, "importance": float(importance)}
                    for name, importance in sorted_features[:10]
                ],
                "shap_summary": {
                    "mean_absolute_shap": mean_abs_shap.tolist(),
                    "max_shap_values": np.max(np.abs(shap_values), axis=0).tolist(),
                    "min_shap_values": np.min(shap_values, axis=0).tolist()
                },
                "explanation_type": "shap_batch",
                "timestamp": time.time()
            }
            
            logger.info("SHAP batch explanation generated",
                       sample_size=len(sample_data),
                       top_feature=sorted_features[0][0])
            
            return batch_explanation
            
        except Exception as e:
            logger.error("SHAP batch explanation failed", error=str(e))
            return {"error": str(e)}
    
    def create_summary_plot(
        self,
        input_data: np.ndarray,
        output_path: str = "shap_summary.png",
        max_samples: int = 100
    ):
        """Create SHAP summary plot."""
        
        if not EXPLAINABILITY_AVAILABLE or self.explainer is None:
            logger.error("SHAP not available for plotting")
            return
        
        try:
            # Limit samples for performance
            if len(input_data) > max_samples:
                indices = np.random.choice(len(input_data), max_samples, replace=False)
                sample_data = input_data[indices]
            else:
                sample_data = input_data
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(sample_data)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                sample_data,
                feature_names=self.feature_names,
                show=False
            )
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("SHAP summary plot created", output_path=output_path)
            
        except Exception as e:
            logger.error("SHAP summary plot creation failed", error=str(e))


class LIMEExplainer(ModelExplainer):
    """LIME-based model explainer."""
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        training_data: Optional[np.ndarray] = None,
        mode: str = "regression"
    ):
        """Initialize a LIME explainer.

        Parameters
        - model: Trained model to explain
        - feature_names: Ordered list of input feature names
        - training_data: Background data for tabular explainer
        - mode: ``regression`` or ``classification``
        """
        super().__init__(model, feature_names)
        
        if not EXPLAINABILITY_AVAILABLE:
            return
        
        self.training_data = training_data
        self.mode = mode
        
        # Initialize LIME explainer
        if training_data is not None:
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=feature_names,
                mode=mode,
                discretize_continuous=True
            )
            logger.info("LIME explainer initialized", mode=mode)
        else:
            logger.warning("Training data required for LIME explainer")
    
    def explain_prediction(
        self,
        input_data: np.ndarray,
        num_features: int = 10,
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """Explain a single prediction using LIME."""
        
        if not EXPLAINABILITY_AVAILABLE or self.explainer is None:
            return {"error": "LIME explainer not available"}
        
        try:
            # Ensure input is 1D for LIME
            if input_data.ndim > 1:
                input_data = input_data[0]
            
            # Get LIME explanation
            explanation = self.explainer.explain_instance(
                data_row=input_data,
                predict_fn=self.model.predict,
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Extract explanation data
            feature_importance = explanation.as_map()[1] if self.mode == "classification" else explanation.as_map()[0]
            
            # Get model prediction
            prediction = self.model.predict(input_data.reshape(1, -1))
            if isinstance(prediction, np.ndarray):
                prediction = prediction[0]
            
            # Create structured explanation
            lime_explanation = {
                "prediction": float(prediction),
                "feature_importance": {
                    self.feature_names[feat_idx]: float(importance)
                    for feat_idx, importance in feature_importance
                },
                "feature_values": {
                    name: float(value)
                    for name, value in zip(self.feature_names, input_data)
                },
                "explanation_type": "lime",
                "num_features_used": len(feature_importance),
                "num_samples": num_samples,
                "timestamp": time.time()
            }
            
            # Add top contributing features
            sorted_importance = sorted(
                feature_importance,
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            lime_explanation["top_features"] = [
                {
                    "feature": self.feature_names[feat_idx],
                    "value": float(input_data[feat_idx]),
                    "importance": float(importance)
                }
                for feat_idx, importance in sorted_importance[:5]
            ]
            
            logger.info("LIME explanation generated",
                       prediction=prediction,
                       top_feature=self.feature_names[sorted_importance[0][0]])
            
            return lime_explanation
            
        except Exception as e:
            logger.error("LIME explanation failed", error=str(e))
            return {"error": str(e)}


class FinancialModelExplainer:
    """Financial domain-specific model explainer."""
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        training_data: Optional[np.ndarray] = None
    ):
        """Create a convenience wrapper combining SHAP and LIME.

        Parameters
        - model: Trained estimator for predictions
        - feature_names: Input feature names
        - training_data: Optional sample data for background distributions
        """
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        if EXPLAINABILITY_AVAILABLE:
            self._initialize_explainers()
        
        # Financial domain knowledge
        self.financial_feature_groups = self._create_financial_feature_groups()
    
    def _initialize_explainers(self):
        """Initialize both SHAP and LIME explainers."""
        
        try:
            # Initialize SHAP explainer
            self.shap_explainer = SHAPExplainer(
                model=self.model,
                feature_names=self.feature_names,
                background_data=self.training_data
            )
            
            # Initialize LIME explainer
            if self.training_data is not None:
                self.lime_explainer = LIMEExplainer(
                    model=self.model,
                    feature_names=self.feature_names,
                    training_data=self.training_data
                )
            
            logger.info("Financial model explainers initialized")
            
        except Exception as e:
            logger.error("Failed to initialize explainers", error=str(e))
    
    def _create_financial_feature_groups(self) -> Dict[str, List[str]]:
        """Create financial domain feature groups."""
        
        groups = {
            "yield_curve": [],
            "market_indicators": [],
            "volatility": [],
            "technical_indicators": [],
            "macro_economic": [],
            "time_features": []
        }
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            
            if "rate_" in feature_lower or "yield" in feature_lower:
                groups["yield_curve"].append(feature)
            elif any(indicator in feature_lower for indicator in ["vix", "dxy", "oil", "gold"]):
                groups["market_indicators"].append(feature)
            elif "vol" in feature_lower or "volatility" in feature_lower:
                groups["volatility"].append(feature)
            elif any(tech in feature_lower for tech in ["ma_", "sma", "ema", "rsi"]):
                groups["technical_indicators"].append(feature)
            elif any(macro in feature_lower for macro in ["fed", "unemployment", "inflation", "gdp"]):
                groups["macro_economic"].append(feature)
            elif any(time_feat in feature_lower for time_feat in ["day", "month", "quarter", "year"]):
                groups["time_features"].append(feature)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}
        
        logger.info("Financial feature groups created", groups=list(groups.keys()))
        return groups
    
    def explain_prediction_comprehensive(
        self,
        input_data: np.ndarray,
        include_shap: bool = True,
        include_lime: bool = True,
        include_financial_analysis: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive explanation combining multiple methods."""
        
        comprehensive_explanation = {
            "input_data": input_data[0].tolist() if input_data.ndim > 1 else input_data.tolist(),
            "feature_names": self.feature_names,
            "timestamp": time.time(),
            "explanations": {}
        }
        
        # Get model prediction
        try:
            prediction = self.model.predict(input_data.reshape(1, -1) if input_data.ndim == 1 else input_data)
            comprehensive_explanation["prediction"] = float(prediction[0] if isinstance(prediction, np.ndarray) else prediction)
        except Exception as e:
            logger.error("Failed to get model prediction", error=str(e))
            comprehensive_explanation["prediction"] = None
        
        # SHAP explanation
        if include_shap and self.shap_explainer:
            shap_explanation = self.shap_explainer.explain_prediction(input_data)
            comprehensive_explanation["explanations"]["shap"] = shap_explanation
        
        # LIME explanation
        if include_lime and self.lime_explainer:
            lime_explanation = self.lime_explainer.explain_prediction(input_data)
            comprehensive_explanation["explanations"]["lime"] = lime_explanation
        
        # Financial domain analysis
        if include_financial_analysis:
            financial_analysis = self._analyze_financial_factors(input_data)
            comprehensive_explanation["explanations"]["financial_analysis"] = financial_analysis
        
        # Consensus analysis
        comprehensive_explanation["consensus"] = self._create_consensus_explanation(
            comprehensive_explanation["explanations"]
        )
        
        logger.info("Comprehensive explanation generated")
        return comprehensive_explanation
    
    def _analyze_financial_factors(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Analyze financial factors contributing to prediction."""
        
        if input_data.ndim > 1:
            input_data = input_data[0]
        
        input_dict = dict(zip(self.feature_names, input_data))
        
        analysis = {
            "curve_shape_analysis": {},
            "market_regime_analysis": {},
            "risk_factor_analysis": {},
            "temporal_analysis": {}
        }
        
        # Yield curve shape analysis
        yield_curve_features = self.financial_feature_groups.get("yield_curve", [])
        if len(yield_curve_features) >= 3:
            rates = [input_dict[feature] for feature in yield_curve_features if feature in input_dict]
            
            if len(rates) >= 3:
                # Calculate curve characteristics
                curve_level = np.mean(rates)
                curve_slope = rates[-1] - rates[0]  # Long - Short
                curve_curvature = rates[len(rates)//2] - (rates[0] + rates[-1]) / 2
                
                analysis["curve_shape_analysis"] = {
                    "level": float(curve_level),
                    "slope": float(curve_slope),
                    "curvature": float(curve_curvature),
                    "shape_interpretation": self._interpret_curve_shape(curve_level, curve_slope, curve_curvature)
                }
        
        # Market regime analysis
        market_features = self.financial_feature_groups.get("market_indicators", [])
        if market_features:
            market_values = {feature: input_dict[feature] for feature in market_features if feature in input_dict}
            
            # Analyze market conditions
            vix_level = market_values.get("vix", 20)  # Default VIX
            market_regime = "high_volatility" if vix_level > 30 else "normal" if vix_level > 15 else "low_volatility"
            
            analysis["market_regime_analysis"] = {
                "regime": market_regime,
                "vix_level": float(vix_level),
                "market_stress_indicator": float(max(0, (vix_level - 15) / 15)),
                "regime_interpretation": self._interpret_market_regime(market_regime, vix_level)
            }
        
        # Risk factor analysis
        risk_factors = []
        for feature, value in input_dict.items():
            if "risk" in feature.lower() or "spread" in feature.lower():
                risk_factors.append({"factor": feature, "value": float(value)})
        
        analysis["risk_factor_analysis"] = {
            "risk_factors": risk_factors,
            "overall_risk_level": self._assess_overall_risk(input_dict)
        }
        
        return analysis
    
    def _interpret_curve_shape(self, level: float, slope: float, curvature: float) -> str:
        """Interpret yield curve shape."""
        
        if slope > 0.01:
            slope_desc = "steep upward slope"
        elif slope < -0.01:
            slope_desc = "inverted (downward slope)"
        else:
            slope_desc = "flat"
        
        if abs(curvature) > 0.005:
            curve_desc = "pronounced curvature" if curvature > 0 else "negative curvature"
        else:
            curve_desc = "linear shape"
        
        return f"Yield curve shows {slope_desc} with {curve_desc} at {level:.2%} average level"
    
    def _interpret_market_regime(self, regime: str, vix_level: float) -> str:
        """Interpret market regime."""
        
        interpretations = {
            "low_volatility": f"Calm market conditions (VIX: {vix_level:.1f}) - typically favorable for risk assets",
            "normal": f"Normal market volatility (VIX: {vix_level:.1f}) - balanced risk environment",
            "high_volatility": f"Elevated market stress (VIX: {vix_level:.1f}) - increased uncertainty and risk aversion"
        }
        
        return interpretations.get(regime, f"Market volatility at {vix_level:.1f}")
    
    def _assess_overall_risk(self, input_dict: Dict[str, float]) -> str:
        """Assess overall risk level from input features."""
        
        risk_indicators = 0
        
        # Check various risk indicators
        vix = input_dict.get("vix", 20)
        if vix > 30:
            risk_indicators += 2
        elif vix > 25:
            risk_indicators += 1
        
        # Check for inverted yield curve
        short_rate = input_dict.get("rate_0.25y", input_dict.get("rate_3m", 0.02))
        long_rate = input_dict.get("rate_10y", input_dict.get("rate_30y", 0.03))
        if short_rate > long_rate:
            risk_indicators += 2
        
        # Assess overall risk
        if risk_indicators >= 3:
            return "high"
        elif risk_indicators >= 1:
            return "medium"
        else:
            return "low"
    
    def _create_consensus_explanation(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Create consensus explanation from multiple methods."""
        
        consensus = {
            "top_consensus_features": [],
            "explanation_agreement": 0.0,
            "confidence": "low"
        }
        
        # Extract feature importance from different methods
        feature_importance_maps = {}
        
        if "shap" in explanations and "feature_importance" in explanations["shap"]:
            feature_importance_maps["shap"] = explanations["shap"]["feature_importance"]
        
        if "lime" in explanations and "feature_importance" in explanations["lime"]:
            feature_importance_maps["lime"] = explanations["lime"]["feature_importance"]
        
        if len(feature_importance_maps) >= 2:
            # Calculate consensus
            all_features = set()
            for importance_map in feature_importance_maps.values():
                all_features.update(importance_map.keys())
            
            # Calculate average importance and agreement
            consensus_importance = {}
            agreement_scores = []
            
            for feature in all_features:
                importances = [
                    importance_map.get(feature, 0)
                    for importance_map in feature_importance_maps.values()
                ]
                
                consensus_importance[feature] = np.mean(importances)
                
                # Calculate agreement (inverse of coefficient of variation)
                if len(importances) > 1 and np.mean(importances) > 0:
                    cv = np.std(importances) / np.mean(importances)
                    agreement_scores.append(1 / (1 + cv))
            
            # Sort by consensus importance
            sorted_consensus = sorted(
                consensus_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            consensus["top_consensus_features"] = [
                {"feature": feature, "consensus_importance": float(importance)}
                for feature, importance in sorted_consensus[:5]
            ]
            
            consensus["explanation_agreement"] = float(np.mean(agreement_scores)) if agreement_scores else 0.0
            
            if consensus["explanation_agreement"] > 0.8:
                consensus["confidence"] = "high"
            elif consensus["explanation_agreement"] > 0.6:
                consensus["confidence"] = "medium"
            else:
                consensus["confidence"] = "low"
        
        return consensus
    
    def generate_explanation_report(
        self,
        input_data: np.ndarray,
        output_path: str = "explanation_report.html"
    ) -> str:
        """Generate comprehensive explanation report."""
        
        explanation = self.explain_prediction_comprehensive(input_data)
        
        # Generate HTML report
        html_content = self._create_html_report(explanation)
        
        with open(output_path, "w") as f:
            f.write(html_content)
        
        logger.info("Explanation report generated", output_path=output_path)
        return output_path
    
    def _create_html_report(self, explanation: Dict[str, Any]) -> str:
        """Create HTML explanation report."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Explanation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .feature-list {{ list-style-type: none; padding: 0; }}
                .feature-item {{ padding: 5px; margin: 2px 0; background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .neutral {{ color: gray; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Explanation Report</h1>
                <p><strong>Prediction:</strong> {explanation.get('prediction', 'N/A')}</p>
                <p><strong>Generated:</strong> {time.ctime(explanation.get('timestamp', time.time()))}</p>
            </div>
        """
        
        # Add SHAP explanation section
        if "shap" in explanation.get("explanations", {}):
            shap_data = explanation["explanations"]["shap"]
            html += f"""
            <div class="section">
                <h2>SHAP Analysis</h2>
                <p><strong>Expected Value:</strong> {shap_data.get('expected_value', 'N/A')}</p>
                <h3>Top Contributing Features:</h3>
                <ul class="feature-list">
            """
            
            for feature in shap_data.get("top_features", [])[:5]:
                contribution = feature.get("contribution", 0)
                css_class = "positive" if contribution > 0 else "negative" if contribution < 0 else "neutral"
                html += f"""
                    <li class="feature-item {css_class}">
                        <strong>{feature.get('feature', 'Unknown')}:</strong> 
                        {feature.get('value', 'N/A')} 
                        (contribution: {contribution:.4f})
                    </li>
                """
            
            html += "</ul></div>"
        
        # Add financial analysis section
        if "financial_analysis" in explanation.get("explanations", {}):
            fin_analysis = explanation["explanations"]["financial_analysis"]
            
            html += """
            <div class="section">
                <h2>Financial Analysis</h2>
            """
            
            # Curve shape analysis
            if "curve_shape_analysis" in fin_analysis:
                curve_analysis = fin_analysis["curve_shape_analysis"]
                html += f"""
                <h3>Yield Curve Analysis</h3>
                <p>{curve_analysis.get('shape_interpretation', 'N/A')}</p>
                <ul>
                    <li>Level: {curve_analysis.get('level', 'N/A'):.4f}</li>
                    <li>Slope: {curve_analysis.get('slope', 'N/A'):.4f}</li>
                    <li>Curvature: {curve_analysis.get('curvature', 'N/A'):.4f}</li>
                </ul>
                """
            
            # Market regime analysis
            if "market_regime_analysis" in fin_analysis:
                market_analysis = fin_analysis["market_regime_analysis"]
                html += f"""
                <h3>Market Regime Analysis</h3>
                <p>{market_analysis.get('regime_interpretation', 'N/A')}</p>
                <p><strong>Regime:</strong> {market_analysis.get('regime', 'N/A')}</p>
                """
            
            html += "</div>"
        
        # Add consensus section
        if "consensus" in explanation:
            consensus = explanation["consensus"]
            html += f"""
            <div class="section">
                <h2>Consensus Analysis</h2>
                <p><strong>Explanation Agreement:</strong> {consensus.get('explanation_agreement', 0):.2%}</p>
                <p><strong>Confidence:</strong> {consensus.get('confidence', 'Unknown')}</p>
                <h3>Top Consensus Features:</h3>
                <ul class="feature-list">
            """
            
            for feature in consensus.get("top_consensus_features", []):
                html += f"""
                    <li class="feature-item">
                        <strong>{feature.get('feature', 'Unknown')}:</strong> 
                        {feature.get('consensus_importance', 0):.4f}
                    </li>
                """
            
            html += "</ul></div>"
        
        html += """
        </body>
        </html>
        """
        
        return html


def create_financial_explainer(
    model: Any,
    feature_names: List[str],
    training_data: Optional[np.ndarray] = None
) -> FinancialModelExplainer:
    """Create financial model explainer."""
    return FinancialModelExplainer(model, feature_names, training_data)


async def main():
    """Main explainability demo."""
    
    # Configure logging
    configure_logging("model_explainer", "INFO", "json")
    
    if not EXPLAINABILITY_AVAILABLE:
        logger.error("SHAP and LIME not available - install with: pip install shap lime")
        return
    
    # Create sample model and data for demonstration
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    feature_names = [f"feature_{i}" for i in range(10)]
    
    # Train sample model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = create_financial_explainer(model, feature_names, X[:100])
    
    # Generate explanation for a sample
    sample_input = X[0]
    explanation = explainer.explain_prediction_comprehensive(sample_input)
    
    print("Explanation generated:")
    print(f"Prediction: {explanation.get('prediction', 'N/A')}")
    
    if "shap" in explanation.get("explanations", {}):
        shap_data = explanation["explanations"]["shap"]
        print("Top SHAP features:")
        for feature in shap_data.get("top_features", [])[:3]:
            print(f"  {feature['feature']}: {feature['contribution']:.4f}")
    
    # Generate report
    report_path = explainer.generate_explanation_report(sample_input)
    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    import time
    asyncio.run(main())
