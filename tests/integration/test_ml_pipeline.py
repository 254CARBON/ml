"""Integration tests for the complete ML pipeline."""

import pytest
import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import httpx
import mlflow
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from training.curve_forecaster.data_generator import CurveDataGenerator
from training.curve_forecaster.model import CurveForecaster
from libs.common.config import BaseConfig
from libs.common.events import create_event_publisher


@pytest.mark.integration
class TestMLPipeline:
    """Test the complete ML pipeline from training to serving."""
    
    @pytest.fixture(scope="class")
    def config(self):
        """Get configuration."""
        return BaseConfig()
    
    @pytest.fixture(scope="class")
    def event_publisher(self, config):
        """Create event publisher."""
        return create_event_publisher(config.ml_redis_url)
    
    @pytest.fixture(scope="class")
    def training_data(self):
        """Generate training data."""
        generator = CurveDataGenerator(seed=42)
        return generator.generate_yield_curve_data(n_samples=500)
    
    @pytest.fixture(scope="class")
    def trained_model(self, training_data, config):
        """Train a model for testing."""
        # Setup MLflow
        mlflow.set_tracking_uri(config.ml_mlflow_tracking_uri)
        
        # Create experiment
        experiment_name = "integration_test"
        try:
            mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            pass  # Experiment already exists
        
        mlflow.set_experiment(experiment_name)
        
        # Train model
        model = CurveForecaster(
            model_type="random_forest",
            lookback_days=10,
            forecast_horizon=3,
            n_estimators=50,
            max_depth=5
        )
        
        with mlflow.start_run(run_name="integration_test_model"):
            model.fit(training_data)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="curve_forecaster"
            )
            
            # Get run info
            run = mlflow.active_run()
            run_id = run.info.run_id
        
        return model, run_id
    
    @pytest.mark.asyncio
    async def test_model_training_and_registration(self, trained_model, config):
        """Test model training and MLflow registration."""
        model, run_id = trained_model
        
        # Verify model is trained
        assert model.is_fitted
        assert len(model.feature_names) > 0
        assert len(model.target_names) > 0
        
        # Verify model is registered in MLflow
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        
        # Check registered model
        try:
            registered_model = client.get_registered_model("curve_forecaster")
            assert registered_model.name == "curve_forecaster"
        except mlflow.exceptions.RestException:
            pytest.skip("MLflow server not available")
    
    @pytest.mark.asyncio
    async def test_model_promotion_workflow(self, trained_model, event_publisher):
        """Test model promotion workflow with events."""
        model, run_id = trained_model
        
        # Promote model to production
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Get latest version
            versions = client.get_latest_versions("curve_forecaster")
            if versions:
                version = versions[0]
                
                # Transition to production
                client.transition_model_version_stage(
                    name="curve_forecaster",
                    version=version.version,
                    stage="Production"
                )
                
                # Publish promotion event
                event_publisher.publish_model_promoted(
                    model_name="curve_forecaster",
                    version=version.version,
                    stage="Production",
                    run_id=run_id
                )
                
                # Verify promotion
                prod_versions = client.get_latest_versions("curve_forecaster", stages=["Production"])
                assert len(prod_versions) > 0
                assert prod_versions[0].current_stage == "Production"
                
        except mlflow.exceptions.RestException:
            pytest.skip("MLflow server not available")
    
    @pytest.mark.asyncio
    async def test_model_serving_api(self):
        """Test model serving API endpoints."""
        base_url = "http://localhost:9005"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test health endpoint
            try:
                response = await client.get(f"{base_url}/health")
                assert response.status_code == 200
                data = response.json()
                assert data["service"] == "model-serving"
            except httpx.ConnectError:
                pytest.skip("Model serving service not available")
            
            # Test models endpoint
            response = await client.get(f"{base_url}/api/v1/models")
            assert response.status_code == 200
            
            # Test prediction endpoint
            prediction_request = {
                "inputs": [
                    {
                        "rate_0.25y": 0.02,
                        "rate_1y": 0.025,
                        "rate_5y": 0.03,
                        "rate_10y": 0.035,
                        "vix": 20.0,
                        "fed_funds": 0.02
                    }
                ]
            }
            
            response = await client.post(
                f"{base_url}/api/v1/predict",
                json=prediction_request
            )
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "latency_ms" in data
            assert len(data["predictions"]) == 1
    
    @pytest.mark.asyncio
    async def test_embedding_service_integration(self):
        """Test embedding service integration."""
        base_url = "http://localhost:9006"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Test health endpoint
                response = await client.get(f"{base_url}/health")
                assert response.status_code == 200
                
                # Test embedding generation
                embed_request = {
                    "items": [
                        {
                            "type": "instrument",
                            "id": "NG_HH_BALMO",
                            "text": "Natural gas Henry Hub balance of month physical forward"
                        },
                        {
                            "type": "curve",
                            "id": "USD_YIELD_CURVE",
                            "text": "US Dollar yield curve government bonds"
                        }
                    ]
                }
                
                response = await client.post(
                    f"{base_url}/api/v1/embed",
                    json=embed_request
                )
                assert response.status_code == 200
                data = response.json()
                assert "vectors" in data
                assert len(data["vectors"]) == 2
                assert len(data["vectors"][0]) > 0  # Check embedding dimension
                
            except httpx.ConnectError:
                pytest.skip("Embedding service not available")
    
    @pytest.mark.asyncio
    async def test_search_service_integration(self):
        """Test search service integration."""
        base_url = "http://localhost:9007"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Test health endpoint
                response = await client.get(f"{base_url}/health")
                assert response.status_code == 200
                
                # First, index some test entities
                index_requests = [
                    {
                        "entity_type": "instrument",
                        "entity_id": "NG_HH_BALMO",
                        "text": "Natural gas Henry Hub balance of month physical forward",
                        "metadata": {"sector": "energy", "commodity": "natural_gas"},
                        "tags": ["energy", "gas", "futures"]
                    },
                    {
                        "entity_type": "curve",
                        "entity_id": "USD_YIELD_CURVE",
                        "text": "US Dollar yield curve government bonds treasury",
                        "metadata": {"currency": "USD", "type": "government"},
                        "tags": ["rates", "government", "usd"]
                    }
                ]
                
                for req in index_requests:
                    response = await client.post(f"{base_url}/api/v1/index", json=req)
                    assert response.status_code == 200
                
                # Wait a bit for indexing
                await asyncio.sleep(2)
                
                # Test search
                search_request = {
                    "query": "natural gas energy",
                    "semantic": True,
                    "limit": 10
                }
                
                response = await client.post(
                    f"{base_url}/api/v1/search",
                    json=search_request
                )
                assert response.status_code == 200
                data = response.json()
                assert "results" in data
                assert "total" in data
                assert "latency_ms" in data
                
            except httpx.ConnectError:
                pytest.skip("Search service not available")
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, training_data):
        """Test complete end-to-end workflow."""
        # This test simulates a complete workflow:
        # 1. Index entities for search
        # 2. Generate embeddings
        # 3. Search for similar entities
        # 4. Make predictions on found entities
        
        services = {
            "embedding": "http://localhost:9006",
            "search": "http://localhost:9007",
            "model_serving": "http://localhost:9005"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Step 1: Index financial instruments
                instruments = [
                    {
                        "entity_type": "instrument",
                        "entity_id": "NG_HH_M1",
                        "text": "Natural gas Henry Hub front month futures contract",
                        "metadata": {"sector": "energy", "commodity": "natural_gas", "maturity": "M1"}
                    },
                    {
                        "entity_type": "instrument", 
                        "entity_id": "CL_WTI_M1",
                        "text": "Crude oil WTI front month futures contract",
                        "metadata": {"sector": "energy", "commodity": "crude_oil", "maturity": "M1"}
                    },
                    {
                        "entity_type": "curve",
                        "entity_id": "USD_LIBOR_CURVE",
                        "text": "US Dollar LIBOR interest rate curve",
                        "metadata": {"currency": "USD", "type": "libor"}
                    }
                ]
                
                # Index entities
                for instrument in instruments:
                    response = await client.post(
                        f"{services['search']}/api/v1/index",
                        json=instrument
                    )
                    if response.status_code != 200:
                        pytest.skip("Search service not available")
                
                # Wait for indexing
                await asyncio.sleep(2)
                
                # Step 2: Search for energy-related instruments
                search_response = await client.post(
                    f"{services['search']}/api/v1/search",
                    json={
                        "query": "energy futures contract",
                        "semantic": True,
                        "filters": {"type": ["instrument"]},
                        "limit": 5
                    }
                )
                
                if search_response.status_code != 200:
                    pytest.skip("Search service not available")
                
                search_results = search_response.json()
                assert len(search_results["results"]) > 0
                
                # Step 3: Make predictions for found instruments
                # Create sample curve data for prediction
                prediction_inputs = []
                for result in search_results["results"][:2]:  # Take first 2 results
                    prediction_inputs.append({
                        "instrument_id": result["entity_id"],
                        "rate_0.25y": 0.02,
                        "rate_1y": 0.025,
                        "rate_5y": 0.03,
                        "rate_10y": 0.035,
                        "vix": 22.0,
                        "fed_funds": 0.025
                    })
                
                prediction_response = await client.post(
                    f"{services['model_serving']}/api/v1/predict",
                    json={"inputs": prediction_inputs}
                )
                
                if prediction_response.status_code != 200:
                    pytest.skip("Model serving service not available")
                
                prediction_data = prediction_response.json()
                assert "predictions" in prediction_data
                assert len(prediction_data["predictions"]) == len(prediction_inputs)
                
                # Step 4: Verify the complete workflow
                assert search_results["total"] > 0
                assert prediction_data["latency_ms"] > 0
                
                print(f"End-to-end test completed successfully:")
                print(f"- Found {search_results['total']} instruments")
                print(f"- Made predictions for {len(prediction_inputs)} instruments")
                print(f"- Search latency: {search_results['latency_ms']:.2f}ms")
                print(f"- Prediction latency: {prediction_data['latency_ms']:.2f}ms")
                
            except httpx.ConnectError as e:
                pytest.skip(f"Service not available: {e}")


@pytest.mark.integration
class TestPerformanceTargets:
    """Test that performance targets are met."""
    
    @pytest.mark.asyncio
    async def test_model_serving_latency(self):
        """Test that model serving meets P95 latency target of 120ms."""
        base_url = "http://localhost:9005"
        latencies = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Warm up
                for _ in range(5):
                    await client.post(
                        f"{base_url}/api/v1/predict",
                        json={"inputs": [{"rate_1y": 0.025, "rate_5y": 0.03}]}
                    )
                
                # Measure latencies
                for _ in range(20):
                    start_time = time.time()
                    response = await client.post(
                        f"{base_url}/api/v1/predict",
                        json={"inputs": [{"rate_1y": 0.025, "rate_5y": 0.03}]}
                    )
                    latency = (time.time() - start_time) * 1000
                    latencies.append(latency)
                    
                    assert response.status_code == 200
                
                # Check P95 latency
                p95_latency = np.percentile(latencies, 95)
                print(f"Model serving P95 latency: {p95_latency:.2f}ms")
                assert p95_latency < 120, f"P95 latency {p95_latency:.2f}ms exceeds target of 120ms"
                
            except httpx.ConnectError:
                pytest.skip("Model serving service not available")
    
    @pytest.mark.asyncio
    async def test_search_latency(self):
        """Test that search meets P95 latency target of 400ms."""
        base_url = "http://localhost:9007"
        latencies = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Index some test data first
                for i in range(10):
                    await client.post(
                        f"{base_url}/api/v1/index",
                        json={
                            "entity_type": "instrument",
                            "entity_id": f"TEST_INSTRUMENT_{i}",
                            "text": f"Test financial instrument {i} for performance testing",
                            "metadata": {"test": True}
                        }
                    )
                
                await asyncio.sleep(2)  # Wait for indexing
                
                # Measure search latencies
                for i in range(20):
                    start_time = time.time()
                    response = await client.post(
                        f"{base_url}/api/v1/search",
                        json={
                            "query": f"financial instrument {i % 5}",
                            "semantic": True,
                            "limit": 10
                        }
                    )
                    latency = (time.time() - start_time) * 1000
                    latencies.append(latency)
                    
                    assert response.status_code == 200
                
                # Check P95 latency
                p95_latency = np.percentile(latencies, 95)
                print(f"Search P95 latency: {p95_latency:.2f}ms")
                assert p95_latency < 400, f"P95 latency {p95_latency:.2f}ms exceeds target of 400ms"
                
            except httpx.ConnectError:
                pytest.skip("Search service not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
