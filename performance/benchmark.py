"""Comprehensive benchmarking suite for ML platform services."""

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional
import httpx
import numpy as np
import pandas as pd
import structlog
from concurrent.futures import ThreadPoolExecutor
import psutil

from .profiler import create_system_profiler, create_load_tester, create_performance_optimizer

logger = structlog.get_logger("benchmark")


class MLPlatformBenchmark:
    """Comprehensive benchmark suite for the ML platform."""
    
    def __init__(self, base_urls: Dict[str, str]):
        self.base_urls = base_urls
        self.results: Dict[str, Any] = {}
        self.system_profiler = create_system_profiler(interval=0.5)
        self.performance_optimizer = create_performance_optimizer()
    
    async def run_full_benchmark(
        self,
        duration_seconds: int = 300,
        concurrent_users: int = 50,
        ramp_up_seconds: int = 30
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark across all services."""
        
        logger.info("Starting full ML platform benchmark", 
                   duration=duration_seconds,
                   concurrent_users=concurrent_users)
        
        # Start system profiling
        self.system_profiler.start_profiling()
        
        try:
            # Run service-specific benchmarks in parallel
            benchmark_tasks = [
                self.benchmark_model_serving(duration_seconds, concurrent_users),
                self.benchmark_embedding_service(duration_seconds, concurrent_users // 2),
                self.benchmark_search_service(duration_seconds, concurrent_users),
                self.benchmark_end_to_end_workflow(duration_seconds, concurrent_users // 4)
            ]
            
            service_results = await asyncio.gather(*benchmark_tasks, return_exceptions=True)
            
            # Collect system metrics
            await asyncio.sleep(5)  # Let system settle
            system_stats = self.system_profiler.get_summary_stats()
            
            # Compile results
            self.results = {
                "benchmark_timestamp": time.time(),
                "test_duration_seconds": duration_seconds,
                "concurrent_users": concurrent_users,
                "system_stats": system_stats,
                "service_benchmarks": {
                    "model_serving": service_results[0] if not isinstance(service_results[0], Exception) else {"error": str(service_results[0])},
                    "embedding_service": service_results[1] if not isinstance(service_results[1], Exception) else {"error": str(service_results[1])},
                    "search_service": service_results[2] if not isinstance(service_results[2], Exception) else {"error": str(service_results[2])},
                    "end_to_end": service_results[3] if not isinstance(service_results[3], Exception) else {"error": str(service_results[3])}
                }
            }
            
            # Generate performance analysis
            analysis = self.performance_optimizer.analyze_performance(
                system_stats=system_stats,
                load_test_results=[r for r in service_results if isinstance(r, dict)],
                db_stats={}
            )
            
            self.results["performance_analysis"] = analysis
            
            # Generate summary report
            self.results["summary"] = self._generate_summary_report()
            
            logger.info("Full benchmark completed", 
                       health_score=analysis.get("system_health_score", 0))
            
            return self.results
            
        finally:
            self.system_profiler.stop_profiling()
    
    async def benchmark_model_serving(self, duration: int, users: int) -> Dict[str, Any]:
        """Benchmark model serving service."""
        
        service_url = self.base_urls.get("model_serving", "http://localhost:9005")
        logger.info("Benchmarking model serving", url=service_url, users=users)
        
        # Test different prediction scenarios
        test_scenarios = [
            {
                "name": "single_prediction",
                "endpoint": "/api/v1/predict",
                "payload": {
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
                },
                "weight": 0.7
            },
            {
                "name": "batch_prediction",
                "endpoint": "/api/v1/predict", 
                "payload": {
                    "inputs": [
                        {
                            "rate_1y": 0.025 + i * 0.001,
                            "rate_5y": 0.03 + i * 0.001,
                            "vix": 20.0 + i
                        } for i in range(10)
                    ]
                },
                "weight": 0.3
            }
        ]
        
        results = {}
        load_tester = create_load_tester(service_url)
        
        for scenario in test_scenarios:
            scenario_duration = int(duration * scenario["weight"])
            scenario_users = max(1, int(users * scenario["weight"]))
            
            result = await load_tester.run_load_test(
                endpoint=scenario["endpoint"],
                payload=scenario["payload"],
                concurrent_users=scenario_users,
                duration_seconds=scenario_duration
            )
            
            results[scenario["name"]] = result
        
        # Aggregate results
        total_requests = sum(r["total_requests"] for r in results.values())
        total_errors = sum(r["failed_requests"] for r in results.values())
        all_response_times = []
        
        for result in results.values():
            if "response_times" in result:
                all_response_times.append(result["response_times"]["p95_ms"])
        
        aggregate_result = {
            "service": "model_serving",
            "scenarios": results,
            "aggregate": {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": (total_errors / total_requests) * 100 if total_requests > 0 else 0,
                "p95_latency_ms": np.mean(all_response_times) if all_response_times else 0,
                "throughput_rps": total_requests / duration if duration > 0 else 0
            }
        }
        
        return aggregate_result
    
    async def benchmark_embedding_service(self, duration: int, users: int) -> Dict[str, Any]:
        """Benchmark embedding service."""
        
        service_url = self.base_urls.get("embedding_service", "http://localhost:9006")
        logger.info("Benchmarking embedding service", url=service_url, users=users)
        
        test_scenarios = [
            {
                "name": "single_embedding",
                "endpoint": "/api/v1/embed",
                "payload": {
                    "items": [
                        {
                            "type": "instrument",
                            "id": "TEST_SINGLE",
                            "text": "Natural gas Henry Hub futures contract for benchmarking"
                        }
                    ]
                },
                "weight": 0.4
            },
            {
                "name": "batch_embedding",
                "endpoint": "/api/v1/embed",
                "payload": {
                    "items": [
                        {
                            "type": "instrument",
                            "id": f"TEST_BATCH_{i}",
                            "text": f"Financial instrument {i} for batch embedding benchmark"
                        } for i in range(20)
                    ]
                },
                "weight": 0.6
            }
        ]
        
        results = {}
        load_tester = create_load_tester(service_url)
        
        for scenario in test_scenarios:
            scenario_duration = int(duration * scenario["weight"])
            scenario_users = max(1, int(users * scenario["weight"]))
            
            result = await load_tester.run_load_test(
                endpoint=scenario["endpoint"],
                payload=scenario["payload"],
                concurrent_users=scenario_users,
                duration_seconds=scenario_duration
            )
            
            results[scenario["name"]] = result
        
        # Calculate embedding-specific metrics
        total_embeddings_generated = 0
        for scenario_name, result in results.items():
            if scenario_name == "single_embedding":
                total_embeddings_generated += result["successful_requests"] * 1
            elif scenario_name == "batch_embedding":
                total_embeddings_generated += result["successful_requests"] * 20
        
        aggregate_result = {
            "service": "embedding_service",
            "scenarios": results,
            "aggregate": {
                "total_embeddings_generated": total_embeddings_generated,
                "embeddings_per_second": total_embeddings_generated / duration if duration > 0 else 0,
                "error_rate": sum(r["failed_requests"] for r in results.values()) / sum(r["total_requests"] for r in results.values()) * 100
            }
        }
        
        return aggregate_result
    
    async def benchmark_search_service(self, duration: int, users: int) -> Dict[str, Any]:
        """Benchmark search service."""
        
        service_url = self.base_urls.get("search_service", "http://localhost:9007")
        logger.info("Benchmarking search service", url=service_url, users=users)
        
        # First, index some test data
        await self._setup_search_test_data(service_url)
        
        test_scenarios = [
            {
                "name": "semantic_search",
                "endpoint": "/api/v1/search",
                "payload": {
                    "query": "energy futures natural gas",
                    "semantic": True,
                    "limit": 10
                },
                "weight": 0.6
            },
            {
                "name": "lexical_search",
                "endpoint": "/api/v1/search",
                "payload": {
                    "query": "benchmark test instrument",
                    "semantic": False,
                    "limit": 10
                },
                "weight": 0.2
            },
            {
                "name": "filtered_search",
                "endpoint": "/api/v1/search",
                "payload": {
                    "query": "financial",
                    "semantic": True,
                    "filters": {"type": ["instrument"]},
                    "limit": 20
                },
                "weight": 0.2
            }
        ]
        
        results = {}
        load_tester = create_load_tester(service_url)
        
        for scenario in test_scenarios:
            scenario_duration = int(duration * scenario["weight"])
            scenario_users = max(1, int(users * scenario["weight"]))
            
            result = await load_tester.run_load_test(
                endpoint=scenario["endpoint"],
                payload=scenario["payload"],
                concurrent_users=scenario_users,
                duration_seconds=scenario_duration
            )
            
            results[scenario["name"]] = result
        
        aggregate_result = {
            "service": "search_service",
            "scenarios": results,
            "aggregate": {
                "total_searches": sum(r["total_requests"] for r in results.values()),
                "search_error_rate": sum(r["failed_requests"] for r in results.values()) / sum(r["total_requests"] for r in results.values()) * 100,
                "average_search_latency_ms": np.mean([r["response_times"]["mean_ms"] for r in results.values() if "response_times" in r])
            }
        }
        
        return aggregate_result
    
    async def _setup_search_test_data(self, service_url: str):
        """Setup test data for search benchmarking."""
        
        test_entities = [
            {
                "entity_type": "instrument",
                "entity_id": f"BENCHMARK_INSTRUMENT_{i}",
                "text": f"Financial instrument {i} for benchmark testing energy futures natural gas commodity",
                "metadata": {"sector": "energy", "benchmark": True},
                "tags": ["benchmark", "energy", "futures"]
            } for i in range(100)
        ] + [
            {
                "entity_type": "curve",
                "entity_id": f"BENCHMARK_CURVE_{i}",
                "text": f"Yield curve {i} for benchmark testing government treasury bonds rates",
                "metadata": {"currency": "USD", "benchmark": True},
                "tags": ["benchmark", "rates", "government"]
            } for i in range(50)
        ]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for entity in test_entities:
                try:
                    await client.post(f"{service_url}/api/v1/index", json=entity)
                except:
                    pass  # Ignore indexing errors during setup
        
        # Wait for indexing to complete
        await asyncio.sleep(5)
        logger.info("Search test data setup completed", entities=len(test_entities))
    
    async def benchmark_end_to_end_workflow(self, duration: int, users: int) -> Dict[str, Any]:
        """Benchmark end-to-end workflow across services."""
        
        logger.info("Benchmarking end-to-end workflow", users=users)
        
        workflow_results = []
        
        async def single_workflow():
            """Execute a single end-to-end workflow."""
            workflow_start = time.perf_counter()
            
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    # Step 1: Index an entity
                    entity_id = f"E2E_BENCHMARK_{int(time.time() * 1000)}"
                    index_start = time.perf_counter()
                    
                    index_response = await client.post(
                        f"{self.base_urls['search_service']}/api/v1/index",
                        json={
                            "entity_type": "instrument",
                            "entity_id": entity_id,
                            "text": "End-to-end benchmark financial instrument energy commodity",
                            "metadata": {"benchmark": True},
                            "tags": ["benchmark", "e2e"]
                        }
                    )
                    
                    index_time = (time.perf_counter() - index_start) * 1000
                    
                    if index_response.status_code != 200:
                        return {"success": False, "step": "index", "error": index_response.status_code}
                    
                    # Step 2: Generate embedding
                    embed_start = time.perf_counter()
                    
                    embed_response = await client.post(
                        f"{self.base_urls['embedding_service']}/api/v1/embed",
                        json={
                            "items": [
                                {
                                    "type": "instrument",
                                    "id": entity_id,
                                    "text": "End-to-end benchmark financial instrument"
                                }
                            ]
                        }
                    )
                    
                    embed_time = (time.perf_counter() - embed_start) * 1000
                    
                    if embed_response.status_code != 200:
                        return {"success": False, "step": "embed", "error": embed_response.status_code}
                    
                    # Step 3: Search for similar entities
                    search_start = time.perf_counter()
                    
                    search_response = await client.post(
                        f"{self.base_urls['search_service']}/api/v1/search",
                        json={
                            "query": "benchmark financial instrument",
                            "semantic": True,
                            "limit": 5
                        }
                    )
                    
                    search_time = (time.perf_counter() - search_start) * 1000
                    
                    if search_response.status_code != 200:
                        return {"success": False, "step": "search", "error": search_response.status_code}
                    
                    # Step 4: Make prediction
                    predict_start = time.perf_counter()
                    
                    predict_response = await client.post(
                        f"{self.base_urls['model_serving']}/api/v1/predict",
                        json={
                            "inputs": [
                                {
                                    "rate_1y": 0.025,
                                    "rate_5y": 0.03,
                                    "vix": 20.0
                                }
                            ]
                        }
                    )
                    
                    predict_time = (time.perf_counter() - predict_start) * 1000
                    
                    if predict_response.status_code != 200:
                        return {"success": False, "step": "predict", "error": predict_response.status_code}
                    
                    total_time = (time.perf_counter() - workflow_start) * 1000
                    
                    return {
                        "success": True,
                        "total_time_ms": total_time,
                        "step_times": {
                            "index_ms": index_time,
                            "embed_ms": embed_time,
                            "search_ms": search_time,
                            "predict_ms": predict_time
                        }
                    }
                    
            except Exception as e:
                return {"success": False, "error": str(e), "total_time_ms": (time.perf_counter() - workflow_start) * 1000}
        
        # Run workflows concurrently
        start_time = time.time()
        tasks = []
        
        while time.time() - start_time < duration:
            # Create batch of concurrent workflows
            batch_size = min(users, 20)  # Limit batch size
            batch_tasks = [asyncio.create_task(single_workflow()) for _ in range(batch_size)]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, dict):
                    workflow_results.append(result)
            
            await asyncio.sleep(1)  # Brief pause between batches
        
        # Analyze workflow results
        successful_workflows = [r for r in workflow_results if r.get("success")]
        failed_workflows = [r for r in workflow_results if not r.get("success")]
        
        if successful_workflows:
            total_times = [r["total_time_ms"] for r in successful_workflows]
            step_times = {
                step: [r["step_times"][step] for r in successful_workflows if "step_times" in r]
                for step in ["index_ms", "embed_ms", "search_ms", "predict_ms"]
            }
            
            workflow_stats = {
                "total_workflows": len(workflow_results),
                "successful_workflows": len(successful_workflows),
                "failed_workflows": len(failed_workflows),
                "success_rate": len(successful_workflows) / len(workflow_results) * 100,
                "total_time": {
                    "mean_ms": np.mean(total_times),
                    "p95_ms": np.percentile(total_times, 95),
                    "p99_ms": np.percentile(total_times, 99)
                },
                "step_breakdown": {
                    step: {
                        "mean_ms": np.mean(times) if times else 0,
                        "p95_ms": np.percentile(times, 95) if times else 0
                    }
                    for step, times in step_times.items()
                }
            }
        else:
            workflow_stats = {
                "total_workflows": len(workflow_results),
                "successful_workflows": 0,
                "failed_workflows": len(failed_workflows),
                "success_rate": 0,
                "error": "No successful workflows"
            }
        
        return workflow_stats
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of benchmark results."""
        
        summary = {
            "overall_health_score": 0,
            "performance_targets_met": {},
            "key_metrics": {},
            "recommendations": []
        }
        
        # Check performance targets
        targets = {
            "inference_latency_p95_ms": 120,
            "embedding_latency_p95_ms": 3000,
            "search_latency_p95_ms": 400,
            "throughput_per_service_rps": 150
        }
        
        service_benchmarks = self.results.get("service_benchmarks", {})
        
        # Model serving targets
        if "model_serving" in service_benchmarks:
            ms_results = service_benchmarks["model_serving"]
            if "aggregate" in ms_results:
                p95_latency = ms_results["aggregate"].get("p95_latency_ms", 0)
                throughput = ms_results["aggregate"].get("throughput_rps", 0)
                
                summary["performance_targets_met"]["inference_latency"] = p95_latency <= targets["inference_latency_p95_ms"]
                summary["performance_targets_met"]["model_serving_throughput"] = throughput >= targets["throughput_per_service_rps"]
                summary["key_metrics"]["inference_p95_ms"] = p95_latency
                summary["key_metrics"]["model_serving_throughput_rps"] = throughput
        
        # Embedding service targets
        if "embedding_service" in service_benchmarks:
            es_results = service_benchmarks["embedding_service"]
            # Extract P95 latency from scenarios
            p95_latencies = []
            for scenario in es_results.get("scenarios", {}).values():
                if "response_times" in scenario:
                    p95_latencies.append(scenario["response_times"]["p95_ms"])
            
            if p95_latencies:
                avg_p95 = np.mean(p95_latencies)
                summary["performance_targets_met"]["embedding_latency"] = avg_p95 <= targets["embedding_latency_p95_ms"]
                summary["key_metrics"]["embedding_p95_ms"] = avg_p95
        
        # Search service targets
        if "search_service" in service_benchmarks:
            ss_results = service_benchmarks["search_service"]
            if "aggregate" in ss_results:
                search_latency = ss_results["aggregate"].get("average_search_latency_ms", 0)
                summary["performance_targets_met"]["search_latency"] = search_latency <= targets["search_latency_p95_ms"]
                summary["key_metrics"]["search_latency_ms"] = search_latency
        
        # Calculate overall health score
        targets_met = sum(1 for met in summary["performance_targets_met"].values() if met)
        total_targets = len(summary["performance_targets_met"])
        
        if total_targets > 0:
            summary["overall_health_score"] = (targets_met / total_targets) * 100
        
        # Add system health score from performance analysis
        perf_analysis = self.results.get("performance_analysis", {})
        system_health = perf_analysis.get("system_health_score", 0)
        
        # Combined health score
        summary["combined_health_score"] = (summary["overall_health_score"] + system_health) / 2
        
        # Add recommendations from performance analysis
        if "recommendations" in perf_analysis:
            summary["recommendations"].extend(perf_analysis["recommendations"])
        
        return summary
    
    def save_results(self, output_path: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        import json
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info("Benchmark results saved", path=output_path)
    
    def generate_report(self) -> str:
        """Generate human-readable benchmark report."""
        
        if not self.results:
            return "No benchmark results available"
        
        summary = self.results.get("summary", {})
        
        report_lines = [
            "# ML Platform Benchmark Report",
            f"Generated at: {time.ctime(self.results.get('benchmark_timestamp', time.time()))}",
            f"Test Duration: {self.results.get('test_duration_seconds', 0)} seconds",
            f"Concurrent Users: {self.results.get('concurrent_users', 0)}",
            "",
            "## Overall Health Score",
            f"Combined Health Score: {summary.get('combined_health_score', 0):.1f}/100",
            "",
            "## Performance Targets",
        ]
        
        targets_met = summary.get("performance_targets_met", {})
        for target, met in targets_met.items():
            status = "✅ PASS" if met else "❌ FAIL"
            report_lines.append(f"- {target}: {status}")
        
        report_lines.extend([
            "",
            "## Key Metrics",
        ])
        
        key_metrics = summary.get("key_metrics", {})
        for metric, value in key_metrics.items():
            if isinstance(value, float):
                report_lines.append(f"- {metric}: {value:.2f}")
            else:
                report_lines.append(f"- {metric}: {value}")
        
        report_lines.extend([
            "",
            "## Recommendations",
        ])
        
        recommendations = summary.get("recommendations", [])
        for i, rec in enumerate(recommendations[:10], 1):  # Top 10 recommendations
            report_lines.append(f"{i}. {rec}")
        
        return "\n".join(report_lines)


async def run_comprehensive_benchmark(
    services: Dict[str, str] = None,
    duration: int = 300,
    users: int = 50
) -> Dict[str, Any]:
    """Run comprehensive benchmark of ML platform."""
    
    if services is None:
        services = {
            "model_serving": "http://localhost:9005",
            "embedding_service": "http://localhost:9006",
            "search_service": "http://localhost:9007"
        }
    
    benchmark = MLPlatformBenchmark(services)
    results = await benchmark.run_full_benchmark(duration, users)
    
    # Save results
    benchmark.save_results("performance/benchmark_results.json")
    
    # Generate and save report
    report = benchmark.generate_report()
    with open("performance/benchmark_report.md", "w") as f:
        f.write(report)
    
    print(report)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Platform Benchmark")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--users", type=int, default=50, help="Concurrent users")
    parser.add_argument("--model-serving-url", default="http://localhost:9005")
    parser.add_argument("--embedding-service-url", default="http://localhost:9006")
    parser.add_argument("--search-service-url", default="http://localhost:9007")
    
    args = parser.parse_args()
    
    services = {
        "model_serving": args.model_serving_url,
        "embedding_service": args.embedding_service_url,
        "search_service": args.search_service_url
    }
    
    asyncio.run(run_comprehensive_benchmark(services, args.duration, args.users))
