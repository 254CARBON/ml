"""Performance profiling tools for ML platform services.

This module provides utility classes for system and application‑level profiling
to help understand bottlenecks in services:
- ``SystemProfiler`` samples process CPU/memory/IO at an interval
- ``ApplicationProfiler`` times functions, captures cProfile and memory stats
- ``DatabaseProfiler`` times queries and identifies slow outliers
- ``LoadTester`` generates controlled request load against HTTP endpoints
- ``PerformanceOptimizer`` summarizes stats and suggests remediation steps

All helpers aim to be non‑intrusive and production‑aware (e.g., async support),
while still offering enough detail for local investigations.
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import structlog
import cProfile
import pstats
import io
from memory_profiler import profile as memory_profile
import tracemalloc

logger = structlog.get_logger("profiler")


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    open_files: int
    threads: int
    response_time_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    error_rate: Optional[float] = None


class SystemProfiler:
    """Profiles system resource usage.

    Spawns a lightweight background thread that samples the current process
    using ``psutil`` and records a bounded history for summary statistics.
    """
    
    def __init__(self, interval: float = 1.0, max_samples: int = 1000):
        self.interval = interval
        self.max_samples = max_samples
        self.metrics_history: deque = deque(maxlen=max_samples)
        self.is_profiling = False
        self.profile_thread: Optional[threading.Thread] = None
        self.process = psutil.Process()
    
    def start_profiling(self):
        """Start continuous system profiling."""
        if self.is_profiling:
            return
        
        self.is_profiling = True
        self.profile_thread = threading.Thread(target=self._profile_loop, daemon=True)
        self.profile_thread.start()
        
        logger.info("System profiling started", interval=self.interval)
    
    def stop_profiling(self):
        """Stop system profiling."""
        self.is_profiling = False
        if self.profile_thread:
            self.profile_thread.join(timeout=5)
        
        logger.info("System profiling stopped", samples_collected=len(self.metrics_history))
    
    def _profile_loop(self):
        """Main profiling loop."""
        while self.is_profiling:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.interval)
            except Exception as e:
                logger.error("Error collecting metrics", error=str(e))
                time.sleep(self.interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics.

        Captures CPU%, RSS/%, per‑process IO counters, approximate system
        network counters, and basic process information (open files/threads).
        """
        # CPU and memory
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        
        # Disk I/O
        io_counters = self.process.io_counters()
        disk_read_mb = io_counters.read_bytes / 1024 / 1024
        disk_write_mb = io_counters.write_bytes / 1024 / 1024
        
        # Network I/O (approximate)
        try:
            net_io = psutil.net_io_counters()
            net_sent_mb = net_io.bytes_sent / 1024 / 1024
            net_recv_mb = net_io.bytes_recv / 1024 / 1024
        except:
            net_sent_mb = net_recv_mb = 0
        
        # Process info
        open_files = len(self.process.open_files())
        threads = self.process.num_threads()
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_io_sent_mb=net_sent_mb,
            network_io_recv_mb=net_recv_mb,
            open_files=open_files,
            threads=threads
        )
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from collected metrics."""
        if not self.metrics_history:
            return {}
        
        metrics_list = list(self.metrics_history)
        
        def calc_stats(values):
            if not values:
                return {}
            return {
                "mean": np.mean(values),
                "median": np.median(values),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99),
                "min": np.min(values),
                "max": np.max(values),
                "std": np.std(values)
            }
        
        return {
            "sample_count": len(metrics_list),
            "duration_seconds": metrics_list[-1].timestamp - metrics_list[0].timestamp,
            "cpu_percent": calc_stats([m.cpu_percent for m in metrics_list]),
            "memory_mb": calc_stats([m.memory_mb for m in metrics_list]),
            "memory_percent": calc_stats([m.memory_percent for m in metrics_list]),
            "disk_io_read_mb": calc_stats([m.disk_io_read_mb for m in metrics_list]),
            "disk_io_write_mb": calc_stats([m.disk_io_write_mb for m in metrics_list]),
            "open_files": calc_stats([m.open_files for m in metrics_list]),
            "threads": calc_stats([m.threads for m in metrics_list])
        }


class ApplicationProfiler:
    """Profiles application-specific performance.

    Features
    - Function timing decorator for sync/async callsites
    - cProfile start/stop with top‑N summary
    - Memory snapshots and simple top‑by‑lineno report
    """
    
    def __init__(self):
        self.function_timings: Dict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: List[Any] = []
        self.active_profiles: Dict[str, cProfile.Profile] = {}
    
    def profile_function(self, func_name: str = None):
        """Decorator to profile function execution time.

        Records per‑call latency in milliseconds keyed by fully‑qualified name
        (module.function) so downstream reports can aggregate across calls.
        """
        def decorator(func: Callable):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    # Use a high‑resolution counter for accurate latency
                    start_time = time.perf_counter()
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    finally:
                        duration = (time.perf_counter() - start_time) * 1000
                        self.function_timings[name].append(duration)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    # Synchronous counterpart uses the same pattern
                    start_time = time.perf_counter()
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        duration = (time.perf_counter() - start_time) * 1000
                        self.function_timings[name].append(duration)
                return sync_wrapper
        
        return decorator
    
    def start_cpu_profiling(self, profile_name: str = "default"):
        """Start CPU profiling."""
        if profile_name in self.active_profiles:
            return
        
        profiler = cProfile.Profile()
        profiler.enable()
        self.active_profiles[profile_name] = profiler
        
        logger.info("CPU profiling started", profile_name=profile_name)
    
    def stop_cpu_profiling(self, profile_name: str = "default") -> Optional[str]:
        """Stop CPU profiling and return results.

        Returns a string with the top (cumulative) functions by time for quick
        inspection, or ``None`` if no profile is active.
        """
        if profile_name not in self.active_profiles:
            return None
        
        profiler = self.active_profiles.pop(profile_name)
        profiler.disable()
        
        # Generate report
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative').print_stats(20)  # Top 20 functions
        
        report = s.getvalue()
        logger.info("CPU profiling stopped", profile_name=profile_name)
        return report
    
    def start_memory_profiling(self):
        """Start memory profiling."""
        tracemalloc.start()
        logger.info("Memory profiling started")
    
    def take_memory_snapshot(self, description: str = ""):
        """Take a memory snapshot."""
        if not tracemalloc.is_tracing():
            logger.warning("Memory profiling not started")
            return
        
        snapshot = tracemalloc.take_snapshot()
        self.memory_snapshots.append({
            "timestamp": time.time(),
            "description": description,
            "snapshot": snapshot
        })
        
        logger.info("Memory snapshot taken", description=description)
    
    def get_memory_report(self, top_n: int = 10) -> str:
        """Get memory usage report.

        Uses ``tracemalloc`` snapshot statistics grouped by line number.
        Returns a human‑readable text report listing the top allocation sites.
        """
        if not self.memory_snapshots:
            return "No memory snapshots available"
        
        latest_snapshot = self.memory_snapshots[-1]["snapshot"]
        top_stats = latest_snapshot.statistics('lineno')
        
        report_lines = [f"Top {top_n} memory allocations:"]
        for index, stat in enumerate(top_stats[:top_n], 1):
            report_lines.append(f"{index}. {stat}")
        
        return "\n".join(report_lines)
    
    def get_function_timing_report(self) -> Dict[str, Any]:
        """Get function timing statistics.

        Aggregates recorded timings per function and returns basic statistics
        (count, mean/median, p95/p99, min/max, total time).
        """
        report = {}
        
        for func_name, timings in self.function_timings.items():
            if timings:
                report[func_name] = {
                    "call_count": len(timings),
                    "mean_ms": np.mean(timings),
                    "median_ms": np.median(timings),
                    "p95_ms": np.percentile(timings, 95),
                    "p99_ms": np.percentile(timings, 99),
                    "min_ms": np.min(timings),
                    "max_ms": np.max(timings),
                    "total_ms": np.sum(timings)
                }
        
        return report


class DatabaseProfiler:
    """Profiles database performance."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.query_stats: Dict[str, List[float]] = defaultdict(list)
        self.connection_pool_stats: List[Dict[str, Any]] = []
    
    async def profile_query(self, query: str, params: tuple = None) -> Dict[str, Any]:
        """Profile a database query."""
        import asyncpg
        
        start_time = time.perf_counter()
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # Execute query
            if params:
                result = await conn.fetch(query, *params)
            else:
                result = await conn.fetch(query)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Store timing
            query_key = query[:100]  # First 100 chars as key
            self.query_stats[query_key].append(duration_ms)
            
            await conn.close()
            
            return {
                "query": query_key,
                "duration_ms": duration_ms,
                "row_count": len(result),
                "success": True
            }
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return {
                "query": query[:100],
                "duration_ms": duration_ms,
                "error": str(e),
                "success": False
            }
    
    async def analyze_slow_queries(self, threshold_ms: float = 100) -> List[Dict[str, Any]]:
        """Analyze slow queries."""
        slow_queries = []
        
        for query, timings in self.query_stats.items():
            if timings:
                mean_time = np.mean(timings)
                if mean_time > threshold_ms:
                    slow_queries.append({
                        "query": query,
                        "mean_duration_ms": mean_time,
                        "p95_duration_ms": np.percentile(timings, 95),
                        "call_count": len(timings),
                        "total_time_ms": np.sum(timings)
                    })
        
        # Sort by total time
        slow_queries.sort(key=lambda x: x["total_time_ms"], reverse=True)
        
        logger.info("Slow query analysis completed", 
                   slow_query_count=len(slow_queries),
                   threshold_ms=threshold_ms)
        
        return slow_queries
    
    def get_query_stats_report(self) -> Dict[str, Any]:
        """Get comprehensive query statistics report."""
        total_queries = sum(len(timings) for timings in self.query_stats.values())
        total_time = sum(sum(timings) for timings in self.query_stats.values())
        
        query_details = {}
        for query, timings in self.query_stats.items():
            if timings:
                query_details[query] = {
                    "call_count": len(timings),
                    "mean_ms": np.mean(timings),
                    "p95_ms": np.percentile(timings, 95),
                    "total_ms": np.sum(timings),
                    "percentage_of_total": (np.sum(timings) / total_time) * 100 if total_time > 0 else 0
                }
        
        return {
            "total_queries": total_queries,
            "total_time_ms": total_time,
            "unique_queries": len(self.query_stats),
            "average_query_time_ms": total_time / total_queries if total_queries > 0 else 0,
            "query_details": query_details
        }


class LoadTester:
    """Load testing utilities for ML services."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
    
    async def run_load_test(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        concurrent_users: int = 10,
        duration_seconds: int = 60,
        ramp_up_seconds: int = 10
    ) -> Dict[str, Any]:
        """Run load test against an endpoint."""
        
        logger.info("Starting load test", 
                   endpoint=endpoint,
                   concurrent_users=concurrent_users,
                   duration_seconds=duration_seconds)
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Results tracking
        response_times = []
        error_count = 0
        success_count = 0
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def make_request():
            async with semaphore:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    request_start = time.perf_counter()
                    try:
                        response = await client.post(f"{self.base_url}{endpoint}", json=payload)
                        response_time = (time.perf_counter() - request_start) * 1000
                        
                        if response.status_code == 200:
                            nonlocal success_count
                            success_count += 1
                            response_times.append(response_time)
                        else:
                            nonlocal error_count
                            error_count += 1
                            
                    except Exception as e:
                        error_count += 1
                        logger.debug("Request failed", error=str(e))
        
        # Generate load
        tasks = []
        current_time = time.time()
        
        while current_time < end_time:
            # Ramp up gradually
            if current_time < start_time + ramp_up_seconds:
                ramp_progress = (current_time - start_time) / ramp_up_seconds
                active_users = max(1, int(concurrent_users * ramp_progress))
            else:
                active_users = concurrent_users
            
            # Create tasks for active users
            for _ in range(active_users):
                task = asyncio.create_task(make_request())
                tasks.append(task)
            
            await asyncio.sleep(0.1)  # 100ms between batches
            current_time = time.time()
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate statistics
        total_requests = success_count + error_count
        duration = time.time() - start_time
        
        results = {
            "endpoint": endpoint,
            "duration_seconds": duration,
            "total_requests": total_requests,
            "successful_requests": success_count,
            "failed_requests": error_count,
            "error_rate": (error_count / total_requests) * 100 if total_requests > 0 else 0,
            "throughput_rps": total_requests / duration if duration > 0 else 0,
            "concurrent_users": concurrent_users
        }
        
        if response_times:
            results["response_times"] = {
                "mean_ms": np.mean(response_times),
                "median_ms": np.median(response_times),
                "p95_ms": np.percentile(response_times, 95),
                "p99_ms": np.percentile(response_times, 99),
                "min_ms": np.min(response_times),
                "max_ms": np.max(response_times)
            }
        
        logger.info("Load test completed", **results)
        return results
    
    async def run_stress_test(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        max_users: int = 100,
        step_size: int = 10,
        step_duration: int = 30
    ) -> List[Dict[str, Any]]:
        """Run stress test with increasing load."""
        
        logger.info("Starting stress test", 
                   endpoint=endpoint,
                   max_users=max_users,
                   step_size=step_size)
        
        stress_results = []
        
        for users in range(step_size, max_users + 1, step_size):
            logger.info(f"Testing with {users} concurrent users")
            
            result = await self.run_load_test(
                endpoint=endpoint,
                payload=payload,
                concurrent_users=users,
                duration_seconds=step_duration,
                ramp_up_seconds=5
            )
            
            result["stress_test_step"] = users
            stress_results.append(result)
            
            # Check if we've hit a breaking point
            if result["error_rate"] > 10:  # 10% error rate
                logger.warning("High error rate detected, stopping stress test", 
                             users=users, 
                             error_rate=result["error_rate"])
                break
            
            # Brief pause between steps
            await asyncio.sleep(5)
        
        logger.info("Stress test completed", steps=len(stress_results))
        return stress_results


class PerformanceOptimizer:
    """Analyzes performance data and suggests optimizations."""
    
    def __init__(self):
        self.optimization_rules = {
            "high_cpu": {
                "threshold": 80,
                "suggestions": [
                    "Consider horizontal scaling",
                    "Profile CPU-intensive functions",
                    "Implement caching for expensive operations",
                    "Optimize algorithms and data structures"
                ]
            },
            "high_memory": {
                "threshold": 80,
                "suggestions": [
                    "Implement memory pooling",
                    "Reduce batch sizes",
                    "Clear unused caches",
                    "Optimize data structures"
                ]
            },
            "high_latency": {
                "threshold": 200,  # ms
                "suggestions": [
                    "Add connection pooling",
                    "Implement request batching",
                    "Add caching layers",
                    "Optimize database queries"
                ]
            },
            "low_throughput": {
                "threshold": 50,  # rps
                "suggestions": [
                    "Increase worker processes",
                    "Optimize request handling",
                    "Implement async processing",
                    "Add load balancing"
                ]
            }
        }
    
    def analyze_performance(
        self,
        system_stats: Dict[str, Any],
        load_test_results: List[Dict[str, Any]],
        db_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance data and provide recommendations."""
        
        recommendations = []
        performance_issues = []
        
        # Analyze system metrics
        if system_stats.get("cpu_percent", {}).get("p95", 0) > self.optimization_rules["high_cpu"]["threshold"]:
            performance_issues.append("high_cpu")
            recommendations.extend(self.optimization_rules["high_cpu"]["suggestions"])
        
        if system_stats.get("memory_percent", {}).get("p95", 0) > self.optimization_rules["high_memory"]["threshold"]:
            performance_issues.append("high_memory")
            recommendations.extend(self.optimization_rules["high_memory"]["suggestions"])
        
        # Analyze load test results
        for result in load_test_results:
            response_times = result.get("response_times", {})
            p95_latency = response_times.get("p95_ms", 0)
            throughput = result.get("throughput_rps", 0)
            
            if p95_latency > self.optimization_rules["high_latency"]["threshold"]:
                performance_issues.append("high_latency")
                recommendations.extend(self.optimization_rules["high_latency"]["suggestions"])
            
            if throughput < self.optimization_rules["low_throughput"]["threshold"]:
                performance_issues.append("low_throughput")
                recommendations.extend(self.optimization_rules["low_throughput"]["suggestions"])
        
        # Analyze database performance
        if db_stats:
            avg_query_time = db_stats.get("average_query_time_ms", 0)
            if avg_query_time > 50:  # 50ms average
                recommendations.extend([
                    "Optimize database indexes",
                    "Consider query result caching",
                    "Analyze and optimize slow queries",
                    "Consider database connection pooling"
                ])
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        analysis_result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "performance_issues": performance_issues,
            "recommendations": recommendations,
            "priority_actions": self._prioritize_recommendations(performance_issues, recommendations),
            "system_health_score": self._calculate_health_score(system_stats, load_test_results),
            "detailed_analysis": {
                "system_stats": system_stats,
                "load_test_summary": self._summarize_load_tests(load_test_results),
                "database_performance": db_stats
            }
        }
        
        logger.info("Performance analysis completed", 
                   issues=len(performance_issues),
                   recommendations=len(recommendations))
        
        return analysis_result
    
    def _prioritize_recommendations(self, issues: List[str], recommendations: List[str]) -> List[str]:
        """Prioritize recommendations based on severity."""
        priority_map = {
            "high_cpu": 1,
            "high_memory": 1,
            "high_latency": 2,
            "low_throughput": 3
        }
        
        # Sort issues by priority
        sorted_issues = sorted(issues, key=lambda x: priority_map.get(x, 10))
        
        # Return top recommendations
        return recommendations[:5]
    
    def _calculate_health_score(
        self,
        system_stats: Dict[str, Any],
        load_test_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall system health score."""
        score = 100.0
        
        # Deduct points for high resource usage
        cpu_p95 = system_stats.get("cpu_percent", {}).get("p95", 0)
        memory_p95 = system_stats.get("memory_percent", {}).get("p95", 0)
        
        if cpu_p95 > 80:
            score -= 20
        elif cpu_p95 > 60:
            score -= 10
        
        if memory_p95 > 80:
            score -= 20
        elif memory_p95 > 60:
            score -= 10
        
        # Deduct points for high latency/low throughput
        for result in load_test_results:
            error_rate = result.get("error_rate", 0)
            response_times = result.get("response_times", {})
            p95_latency = response_times.get("p95_ms", 0)
            
            if error_rate > 5:
                score -= 15
            elif error_rate > 1:
                score -= 5
            
            if p95_latency > 500:
                score -= 15
            elif p95_latency > 200:
                score -= 10
        
        return max(0.0, score)
    
    def _summarize_load_tests(self, load_test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize load test results."""
        if not load_test_results:
            return {}
        
        total_requests = sum(r.get("total_requests", 0) for r in load_test_results)
        total_errors = sum(r.get("failed_requests", 0) for r in load_test_results)
        
        all_response_times = []
        all_throughputs = []
        
        for result in load_test_results:
            if "response_times" in result:
                all_response_times.extend([result["response_times"]["mean_ms"]])
            if "throughput_rps" in result:
                all_throughputs.append(result["throughput_rps"])
        
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "overall_error_rate": (total_errors / total_requests) * 100 if total_requests > 0 else 0,
            "average_response_time_ms": np.mean(all_response_times) if all_response_times else 0,
            "average_throughput_rps": np.mean(all_throughputs) if all_throughputs else 0,
            "test_count": len(load_test_results)
        }


def create_system_profiler(interval: float = 1.0) -> SystemProfiler:
    """Create system profiler."""
    return SystemProfiler(interval=interval)


def create_application_profiler() -> ApplicationProfiler:
    """Create application profiler."""
    return ApplicationProfiler()


def create_database_profiler(db_url: str) -> DatabaseProfiler:
    """Create database profiler."""
    return DatabaseProfiler(db_url)


def create_load_tester(base_url: str) -> LoadTester:
    """Create load tester."""
    return LoadTester(base_url)


def create_performance_optimizer() -> PerformanceOptimizer:
    """Create performance optimizer."""
    return PerformanceOptimizer()
