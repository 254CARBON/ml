"""Load testing with Locust for ML platform services."""

import json
import random
import time
from locust import HttpUser, task, between


class ModelServingUser(HttpUser):
    """Load test user for model serving service."""
    
    wait_time = between(1, 3)
    host = "http://localhost:9005"
    
    def on_start(self):
        """Setup for each user."""
        # Check if service is available
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception("Model serving service not available")
    
    @task(10)
    def predict_single(self):
        """Test single prediction endpoint."""
        payload = {
            "inputs": [
                {
                    "rate_0.25y": random.uniform(0.01, 0.05),
                    "rate_1y": random.uniform(0.015, 0.055),
                    "rate_5y": random.uniform(0.02, 0.06),
                    "rate_10y": random.uniform(0.025, 0.065),
                    "vix": random.uniform(15, 35),
                    "fed_funds": random.uniform(0.01, 0.05)
                }
            ]
        }
        
        with self.client.post(
            "/api/v1/predict",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "predictions" in data and "latency_ms" in data:
                    # Check latency target
                    if data["latency_ms"] > 120:
                        response.failure(f"Latency {data['latency_ms']:.2f}ms exceeds 120ms target")
                    else:
                        response.success()
                else:
                    response.failure("Missing required fields in response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(3)
    def predict_batch(self):
        """Test batch prediction endpoint."""
        batch_size = random.randint(2, 10)
        inputs = []
        
        for _ in range(batch_size):
            inputs.append({
                "rate_0.25y": random.uniform(0.01, 0.05),
                "rate_1y": random.uniform(0.015, 0.055),
                "rate_5y": random.uniform(0.02, 0.06),
                "rate_10y": random.uniform(0.025, 0.065),
                "vix": random.uniform(15, 35),
                "fed_funds": random.uniform(0.01, 0.05)
            })
        
        payload = {"inputs": inputs}
        
        with self.client.post(
            "/api/v1/predict",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "predictions" in data and len(data["predictions"]) == batch_size:
                    response.success()
                else:
                    response.failure("Incorrect batch response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def list_models(self):
        """Test list models endpoint."""
        with self.client.get("/api/v1/models", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class EmbeddingServiceUser(HttpUser):
    """Load test user for embedding service."""
    
    wait_time = between(2, 5)
    host = "http://localhost:9006"
    
    def on_start(self):
        """Setup for each user."""
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception("Embedding service not available")
    
    @task(8)
    def generate_embeddings(self):
        """Test embedding generation."""
        entities = [
            "Natural gas Henry Hub futures contract",
            "Crude oil WTI front month",
            "US Treasury 10-year bond yield",
            "EUR/USD foreign exchange rate",
            "S&P 500 equity index futures",
            "Gold futures commodity contract"
        ]
        
        batch_size = random.randint(1, 5)
        items = []
        
        for i in range(batch_size):
            items.append({
                "type": random.choice(["instrument", "curve", "scenario"]),
                "id": f"TEST_{random.randint(1000, 9999)}",
                "text": random.choice(entities)
            })
        
        payload = {"items": items}
        
        with self.client.post(
            "/api/v1/embed",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "vectors" in data and len(data["vectors"]) == batch_size:
                    # Check embedding dimension
                    if len(data["vectors"][0]) > 0:
                        response.success()
                    else:
                        response.failure("Empty embedding vector")
                else:
                    response.failure("Incorrect embedding response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def list_models(self):
        """Test list models endpoint."""
        with self.client.get("/api/v1/models", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class SearchServiceUser(HttpUser):
    """Load test user for search service."""
    
    wait_time = between(1, 4)
    host = "http://localhost:9007"
    
    def on_start(self):
        """Setup for each user."""
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception("Search service not available")
        
        # Index some test data
        self.index_test_data()
    
    def index_test_data(self):
        """Index test data for searching."""
        test_entities = [
            {
                "entity_type": "instrument",
                "entity_id": f"LOAD_TEST_INSTRUMENT_{random.randint(1000, 9999)}",
                "text": "Natural gas Henry Hub futures contract for load testing",
                "metadata": {"sector": "energy", "test": True},
                "tags": ["energy", "gas", "futures", "load_test"]
            },
            {
                "entity_type": "curve",
                "entity_id": f"LOAD_TEST_CURVE_{random.randint(1000, 9999)}",
                "text": "US Treasury yield curve government bonds",
                "metadata": {"currency": "USD", "test": True},
                "tags": ["rates", "government", "treasury", "load_test"]
            }
        ]
        
        for entity in test_entities:
            self.client.post("/api/v1/index", json=entity)
    
    @task(10)
    def search_semantic(self):
        """Test semantic search."""
        queries = [
            "energy futures contract",
            "government bond yield curve",
            "natural gas commodity",
            "treasury interest rates",
            "crude oil futures",
            "foreign exchange rates"
        ]
        
        query = random.choice(queries)
        payload = {
            "query": query,
            "semantic": True,
            "limit": random.randint(5, 20)
        }
        
        with self.client.post(
            "/api/v1/search",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data and "latency_ms" in data:
                    # Check latency target
                    if data["latency_ms"] > 400:
                        response.failure(f"Search latency {data['latency_ms']:.2f}ms exceeds 400ms target")
                    else:
                        response.success()
                else:
                    response.failure("Missing required fields in search response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(5)
    def search_with_filters(self):
        """Test search with filters."""
        payload = {
            "query": "load test",
            "semantic": True,
            "filters": {
                "type": [random.choice(["instrument", "curve"])],
                "tags": ["load_test"]
            },
            "limit": 10
        }
        
        with self.client.post(
            "/api/v1/search",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def index_entity(self):
        """Test entity indexing."""
        entity = {
            "entity_type": random.choice(["instrument", "curve", "scenario"]),
            "entity_id": f"LOAD_TEST_{random.randint(10000, 99999)}",
            "text": f"Load test entity {random.randint(1, 1000)} for performance testing",
            "metadata": {"test": True, "timestamp": time.time()},
            "tags": ["load_test", "performance"]
        }
        
        with self.client.post(
            "/api/v1/index",
            json=entity,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def get_index_stats(self):
        """Test index statistics endpoint."""
        with self.client.get("/api/v1/index/stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class MLPlatformUser(HttpUser):
    """Combined load test user for the entire ML platform."""
    
    wait_time = between(2, 6)
    
    def __init__(self, *args, **kwargs):
        """Initialize base URLs for each service under test."""
        super().__init__(*args, **kwargs)
        self.services = {
            "model_serving": "http://localhost:9005",
            "embedding": "http://localhost:9006", 
            "search": "http://localhost:9007"
        }
    
    def on_start(self):
        """Setup for combined testing."""
        # Check all services
        for service_name, base_url in self.services.items():
            response = self.client.get(f"{base_url}/health")
            if response.status_code != 200:
                raise Exception(f"{service_name} service not available")
    
    @task(5)
    def end_to_end_workflow(self):
        """Test end-to-end workflow across services."""
        # Step 1: Index an entity
        entity = {
            "entity_type": "instrument",
            "entity_id": f"E2E_TEST_{random.randint(1000, 9999)}",
            "text": "End-to-end test financial instrument",
            "metadata": {"test": True, "workflow": "e2e"},
            "tags": ["test", "e2e", "workflow"]
        }
        
        with self.client.post(
            f"{self.services['search']}/api/v1/index",
            json=entity,
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure("Failed to index entity")
                return
        
        # Step 2: Search for the entity
        time.sleep(1)  # Brief wait for indexing
        
        with self.client.post(
            f"{self.services['search']}/api/v1/search",
            json={
                "query": "end-to-end test financial",
                "semantic": True,
                "limit": 5
            },
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure("Failed to search entities")
                return
        
        # Step 3: Generate embeddings
        with self.client.post(
            f"{self.services['embedding']}/api/v1/embed",
            json={
                "items": [
                    {
                        "type": "instrument",
                        "id": entity["entity_id"],
                        "text": entity["text"]
                    }
                ]
            },
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure("Failed to generate embeddings")
                return
        
        # Step 4: Make prediction
        with self.client.post(
            f"{self.services['model_serving']}/api/v1/predict",
            json={
                "inputs": [
                    {
                        "instrument_id": entity["entity_id"],
                        "rate_1y": random.uniform(0.02, 0.04),
                        "rate_5y": random.uniform(0.025, 0.045),
                        "vix": random.uniform(18, 28)
                    }
                ]
            },
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure("Failed to make prediction")


# Custom load test scenarios
class StressTestUser(HttpUser):
    """High-intensity stress test user."""
    
    wait_time = between(0.1, 0.5)  # Very short wait times
    host = "http://localhost:9005"
    
    @task
    def rapid_predictions(self):
        """Rapid-fire predictions to test system limits."""
        payload = {
            "inputs": [
                {
                    "rate_1y": random.uniform(0.02, 0.04),
                    "rate_5y": random.uniform(0.025, 0.045)
                }
            ]
        }
        
        self.client.post("/api/v1/predict", json=payload)


if __name__ == "__main__":
    # Run with: locust -f locustfile.py --host=http://localhost:9005
    pass
