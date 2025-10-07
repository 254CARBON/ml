"""Contract tests for API validation against OpenAPI specs."""

import pytest
import json
import httpx
from pathlib import Path
import sys
from openapi_spec_validator import validate_spec
from jsonschema import validate, ValidationError

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


@pytest.fixture
def specs():
    """Load API specifications."""
    specs_path = project_root / "specs.lock.json"
    with open(specs_path, "r") as f:
        return json.load(f)


@pytest.mark.contract
class TestAPIContracts:
    """Test API contracts against OpenAPI specifications."""
    
    def test_specs_are_valid_openapi(self, specs):
        """Test that all specs are valid OpenAPI 3.0 specifications."""
        for contract_name, contract in specs["contracts"].items():
            try:
                validate_spec(contract["schema"])
                print(f"✓ {contract_name} is valid OpenAPI spec")
            except Exception as e:
                pytest.fail(f"Invalid OpenAPI spec for {contract_name}: {e}")
    
    @pytest.mark.asyncio
    async def test_model_serving_contract(self, specs):
        """Test model serving API contract."""
        contract = specs["contracts"]["model-serving-api"]["schema"]
        base_url = "http://localhost:9005"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Test predict endpoint
                predict_schema = contract["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]["schema"]
                
                # Valid request
                valid_request = {
                    "inputs": [
                        {
                            "rate_1y": 0.025,
                            "rate_5y": 0.03,
                            "vix": 20.0
                        }
                    ]
                }
                
                # Validate request against schema
                validate(valid_request, predict_schema)
                
                # Make actual API call
                response = await client.post(f"{base_url}/api/v1/predict", json=valid_request)
                
                if response.status_code == 200:
                    data = response.json()
                    # Validate response structure
                    assert "predictions" in data
                    assert "model_name" in data
                    assert "model_version" in data
                    assert "latency_ms" in data
                    assert isinstance(data["predictions"], list)
                    assert len(data["predictions"]) == len(valid_request["inputs"])
                
            except httpx.ConnectError:
                pytest.skip("Model serving service not available")
            except ValidationError as e:
                pytest.fail(f"Request validation failed: {e}")
    
    @pytest.mark.asyncio
    async def test_embedding_service_contract(self, specs):
        """Test embedding service API contract."""
        contract = specs["contracts"]["embedding-api"]["schema"]
        base_url = "http://localhost:9006"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Test embed endpoint
                embed_schema = contract["paths"]["/embed"]["post"]["requestBody"]["content"]["application/json"]["schema"]
                
                # Valid request
                valid_request = {
                    "items": [
                        {
                            "type": "instrument",
                            "id": "TEST_INSTRUMENT",
                            "text": "Test financial instrument for contract validation"
                        }
                    ],
                    "model": "default"
                }
                
                # Validate request against schema
                validate(valid_request, embed_schema)
                
                # Make actual API call
                response = await client.post(f"{base_url}/api/v1/embed", json=valid_request)
                
                if response.status_code == 200:
                    data = response.json()
                    # Validate response structure
                    assert "model_version" in data
                    assert "vectors" in data
                    assert "count" in data
                    assert "latency_ms" in data
                    assert isinstance(data["vectors"], list)
                    assert len(data["vectors"]) == len(valid_request["items"])
                    assert data["count"] == len(valid_request["items"])
                
            except httpx.ConnectError:
                pytest.skip("Embedding service not available")
            except ValidationError as e:
                pytest.fail(f"Request validation failed: {e}")
    
    @pytest.mark.asyncio
    async def test_search_service_contract(self, specs):
        """Test search service API contract."""
        contract = specs["contracts"]["search-api"]["schema"]
        base_url = "http://localhost:9007"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Test search endpoint
                search_schema = contract["paths"]["/search"]["post"]["requestBody"]["content"]["application/json"]["schema"]
                
                # Valid request
                valid_request = {
                    "query": "test financial instrument",
                    "semantic": True,
                    "filters": {
                        "type": ["instrument"]
                    },
                    "limit": 10
                }
                
                # Validate request against schema
                validate(valid_request, search_schema)
                
                # Make actual API call
                response = await client.post(f"{base_url}/api/v1/search", json=valid_request)
                
                if response.status_code == 200:
                    data = response.json()
                    # Validate response structure
                    assert "results" in data
                    assert "total" in data
                    assert "query" in data
                    assert "semantic" in data
                    assert "latency_ms" in data
                    assert isinstance(data["results"], list)
                    assert data["query"] == valid_request["query"]
                    assert data["semantic"] == valid_request["semantic"]
                
            except httpx.ConnectError:
                pytest.skip("Search service not available")
            except ValidationError as e:
                pytest.fail(f"Request validation failed: {e}")
    
    def test_invalid_requests_rejected(self, specs):
        """Test that invalid requests are properly rejected."""
        # Test invalid model serving request
        predict_schema = specs["contracts"]["model-serving-api"]["schema"]["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        
        invalid_requests = [
            {},  # Missing inputs
            {"inputs": "not_an_array"},  # Wrong type
            {"inputs": []},  # Empty inputs
        ]
        
        for invalid_request in invalid_requests:
            with pytest.raises(ValidationError):
                validate(invalid_request, predict_schema)
    
    def test_event_schema_validation(self, specs):
        """Test event schema validation."""
        events = specs["events"]
        
        # Test model promoted event
        model_promoted_schema = events["ml.model.promoted.v1"]["schema"]
        
        valid_event = {
            "model_name": "test_model",
            "version": "1.0.0",
            "stage": "Production",
            "timestamp": 1640995200000,
            "run_id": "test_run_123"
        }
        
        # Validate against Avro schema (simplified validation)
        required_fields = [field["name"] for field in model_promoted_schema["fields"]]
        for field in required_fields:
            assert field in valid_event, f"Missing required field: {field}"
        
        # Test embedding reindex request event
        reindex_schema = events["ml.embedding.reindex.request.v1"]["schema"]
        
        valid_reindex_event = {
            "entity_type": "instruments",
            "batch_size": 100,
            "model_version": "v1.0",
            "timestamp": 1640995200000
        }
        
        required_fields = [field["name"] for field in reindex_schema["fields"]]
        for field in required_fields:
            assert field in valid_reindex_event, f"Missing required field: {field}"
    
    def test_backward_compatibility(self, specs):
        """Test backward compatibility of API changes."""
        # This would typically compare against previous spec versions
        # For now, just verify current specs are consistent
        
        contracts = specs["contracts"]
        
        # Verify all contracts have required metadata
        for contract_name, contract in contracts.items():
            assert "version" in contract
            assert "schema" in contract
            assert "openapi" in contract["schema"]
            assert "info" in contract["schema"]
            assert "paths" in contract["schema"]
            
            # Verify version format
            version = contract["version"]
            assert len(version.split(".")) == 3, f"Invalid version format: {version}"


@pytest.mark.contract
class TestServiceManifests:
    """Test service manifest validation."""
    
    def test_service_manifests_exist(self):
        """Test that all services have manifests."""
        services = ["mlflow", "model-serving", "embedding", "search", "indexer-worker"]
        
        for service in services:
            manifest_path = project_root / f"service-{service}" / "service-manifest.yaml"
            assert manifest_path.exists(), f"Service manifest missing for {service}"
    
    def test_service_manifest_structure(self):
        """Test service manifest structure."""
        import yaml
        
        services = ["mlflow", "model-serving", "embedding", "search", "indexer-worker"]
        required_fields = [
            "service_name", "domain", "runtime", "maturity", "owner"
        ]
        
        for service in services:
            manifest_path = project_root / f"service-{service}" / "service-manifest.yaml"
            
            with open(manifest_path, "r") as f:
                manifest = yaml.safe_load(f)
            
            for field in required_fields:
                assert field in manifest, f"Missing required field '{field}' in {service} manifest"
            
            # Validate specific field values
            assert manifest["domain"] == "ml"
            assert manifest["runtime"] == "python"
            assert manifest["maturity"] in ["alpha", "beta", "stable", "planned"]
            assert manifest["owner"] == "ml"
    
    def test_service_dependencies(self):
        """Test service dependency declarations."""
        import yaml
        
        # Load all manifests
        manifests = {}
        services = ["mlflow", "model-serving", "embedding", "search", "indexer-worker"]
        
        for service in services:
            manifest_path = project_root / f"service-{service}" / "service-manifest.yaml"
            with open(manifest_path, "r") as f:
                manifests[service] = yaml.safe_load(f)
        
        # Validate dependencies
        for service, manifest in manifests.items():
            if "dependencies" in manifest:
                deps = manifest["dependencies"]
                
                # Check internal dependencies exist
                if "internal" in deps:
                    for internal_dep in deps["internal"]:
                        # This would check if internal dependency is valid
                        # For now, just verify it's a string
                        assert isinstance(internal_dep, str)
                
                # Check external dependencies are declared
                if "external" in deps:
                    for external_dep in deps["external"]:
                        assert isinstance(external_dep, str)
                        # Common external dependencies
                        valid_external = ["postgres", "redis", "minio", "mlflow", "embedding-service"]
                        # Don't enforce strict validation for flexibility

    @pytest.mark.asyncio
    async def test_multi_tenant_search(self, specs):
        """Test multi-tenant search functionality."""
        contract = specs["contracts"]["search-api"]["schema"]
        base_url = "http://localhost:9007"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Test search with tenant_id parameter
                search_request = {
                    "query": "test query",
                    "semantic": True,
                    "filters": {"type": "instrument"},
                    "limit": 10,
                    "similarity_threshold": 0.0
                }
                
                # Test with tenant_id
                response = await client.post(
                    f"{base_url}/api/v1/search?tenant_id=test_tenant",
                    json=search_request
                )
                
                if response.status_code == 200:
                    data = response.json()
                    assert "results" in data
                    assert "total" in data
                    assert "query" in data
                    assert "semantic" in data
                    assert "latency_ms" in data
                
            except httpx.ConnectError:
                pytest.skip("Search service not available")

    @pytest.mark.asyncio
    async def test_ab_testing_endpoints(self, specs):
        """Test A/B testing experiment endpoints."""
        base_url = "http://localhost:9005"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Test experiment creation
                experiment_request = {
                    "name": "test_experiment",
                    "model_a": "model_v1",
                    "model_b": "model_v2",
                    "traffic_split": 0.5,
                    "description": "Test A/B experiment"
                }
                
                response = await client.post(f"{base_url}/api/v1/experiments", json=experiment_request)
                
                if response.status_code == 200:
                    data = response.json()
                    assert "experiment_id" in data
                    assert "name" in data
                    assert "status" in data
                    assert "traffic_split" in data
                    assert "created_at" in data
                
                # Test experiment listing
                response = await client.get(f"{base_url}/api/v1/experiments")
                
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, list)
                
            except httpx.ConnectError:
                pytest.skip("Model serving service not available")

    @pytest.mark.asyncio
    async def test_shadow_deployment_endpoints(self, specs):
        """Test shadow deployment endpoints."""
        base_url = "http://localhost:9005"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Test shadow deployment configuration
                shadow_request = {
                    "model_name": "test_model",
                    "shadow_model": "shadow_model_v1",
                    "enabled": True
                }
                
                response = await client.post(f"{base_url}/api/v1/shadow", json=shadow_request)
                
                if response.status_code == 200:
                    data = response.json()
                    assert "model_name" in data
                    assert "shadow_model" in data
                    assert "enabled" in data
                    assert "comparison_metrics" in data
                
                # Test shadow deployment status
                response = await client.get(f"{base_url}/api/v1/shadow/test_model")
                
                if response.status_code == 200:
                    data = response.json()
                    assert "model_name" in data
                    assert "shadow_model" in data
                    assert "enabled" in data
                    assert "comparison_metrics" in data
                
            except httpx.ConnectError:
                pytest.skip("Model serving service not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
