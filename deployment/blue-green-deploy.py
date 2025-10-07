#!/usr/bin/env python3
"""Blue-green deployment script for ML platform services."""

import asyncio
import argparse
import time
import subprocess
import json
from typing import Dict, List, Optional
import httpx
import structlog

logger = structlog.get_logger("blue_green_deploy")


class BlueGreenDeployer:
    """Handles blue-green deployments for ML platform services."""
    
    def __init__(
        self,
        namespace_blue: str = "ml-platform-blue",
        namespace_green: str = "ml-platform-green",
        namespace_prod: str = "ml-platform-production"
    ):
        self.namespace_blue = namespace_blue
        self.namespace_green = namespace_green
        self.namespace_prod = namespace_prod
        self.kubectl_timeout = 300  # 5 minutes
    
    async def deploy_to_slot(
        self,
        slot: str,
        image_tag: str,
        services: List[str] = None
    ) -> bool:
        """Deploy services to blue or green slot."""
        
        if services is None:
            services = ["model-serving", "embedding-service", "search-service"]
        
        namespace = self.namespace_blue if slot == "blue" else self.namespace_green
        
        logger.info(f"Deploying to {slot} slot", namespace=namespace, services=services, image_tag=image_tag)
        
        try:
            # Update image tags for all services
            for service in services:
                image_name = f"ghcr.io/254carbon/ml-platform-{service}:{image_tag}"
                
                cmd = [
                    "kubectl", "set", "image",
                    f"deployment/{service}",
                    f"{service}={image_name}",
                    "-n", namespace
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.kubectl_timeout)
                
                if result.returncode != 0:
                    logger.error(f"Failed to update {service} image", error=result.stderr)
                    return False
                
                logger.info(f"Updated {service} image", image=image_name)
            
            # Wait for rollout to complete
            for service in services:
                cmd = ["kubectl", "rollout", "status", f"deployment/{service}", "-n", namespace, "--timeout=300s"]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.kubectl_timeout)
                
                if result.returncode != 0:
                    logger.error(f"Rollout failed for {service}", error=result.stderr)
                    return False
                
                logger.info(f"Rollout completed for {service}")
            
            logger.info(f"Deployment to {slot} slot completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Deployment to {slot} slot failed", error=str(e))
            return False
    
    async def validate_deployment(self, slot: str, services: List[str] = None) -> bool:
        """Validate deployment in the specified slot."""
        
        if services is None:
            services = ["model-serving", "embedding-service", "search-service"]
        
        namespace = self.namespace_blue if slot == "blue" else self.namespace_green
        
        logger.info(f"Validating {slot} slot deployment", namespace=namespace)
        
        try:
            # Get service endpoints
            service_urls = {}
            for service in services:
                # Port forward to test services
                port_map = {
                    "model-serving": 9005,
                    "embedding-service": 9006,
                    "search-service": 9007
                }
                
                port = port_map.get(service, 8080)
                service_urls[service] = f"http://localhost:{port + (100 if slot == 'green' else 0)}"
            
            # Test each service
            async with httpx.AsyncClient(timeout=30.0) as client:
                for service, url in service_urls.items():
                    try:
                        # Health check
                        response = await client.get(f"{url}/health")
                        if response.status_code != 200:
                            logger.error(f"Health check failed for {service}", status=response.status_code)
                            return False
                        
                        # Basic functionality test
                        if service == "model-serving":
                            test_response = await client.post(
                                f"{url}/api/v1/predict",
                                json={"inputs": [{"rate_1y": 0.025, "rate_5y": 0.03}]}
                            )
                            if test_response.status_code != 200:
                                logger.error(f"Prediction test failed for {service}")
                                return False
                        
                        elif service == "embedding-service":
                            test_response = await client.post(
                                f"{url}/api/v1/embed",
                                json={"items": [{"text": "test", "type": "instrument"}]}
                            )
                            if test_response.status_code != 200:
                                logger.error(f"Embedding test failed for {service}")
                                return False
                        
                        elif service == "search-service":
                            test_response = await client.post(
                                f"{url}/api/v1/search",
                                json={"query": "test", "limit": 5}
                            )
                            if test_response.status_code != 200:
                                logger.error(f"Search test failed for {service}")
                                return False
                        
                        logger.info(f"Validation passed for {service}")
                        
                    except Exception as e:
                        logger.error(f"Validation failed for {service}", error=str(e))
                        return False
            
            logger.info(f"All services validated successfully in {slot} slot")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed for {slot} slot", error=str(e))
            return False
    
    async def switch_traffic(self, target_slot: str, percentage: int = 100) -> bool:
        """Switch traffic to target slot."""
        
        logger.info(f"Switching {percentage}% traffic to {target_slot} slot")
        
        try:
            # Update service selectors to point to target slot
            services = ["model-serving", "embedding-service", "search-service"]
            target_namespace = self.namespace_blue if target_slot == "blue" else self.namespace_green
            
            for service in services:
                # Update active service selector
                cmd = [
                    "kubectl", "patch", "service", f"{service}-active",
                    "-n", self.namespace_prod,
                    "-p", json.dumps({
                        "spec": {
                            "selector": {
                                "app": service,
                                "deployment-slot": target_slot
                            }
                        }
                    })
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to switch traffic for {service}", error=result.stderr)
                    return False
                
                logger.info(f"Traffic switched for {service} to {target_slot}")
            
            # Update ingress canary weight for gradual rollout
            if percentage < 100:
                cmd = [
                    "kubectl", "patch", "ingress", "ml-platform-canary-ingress",
                    "-n", self.namespace_prod,
                    "-p", json.dumps({
                        "metadata": {
                            "annotations": {
                                "nginx.ingress.kubernetes.io/canary-weight": str(percentage)
                            }
                        }
                    })
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error("Failed to update canary weight", error=result.stderr)
                    return False
            
            logger.info(f"Traffic switch to {target_slot} completed")
            return True
            
        except Exception as e:
            logger.error(f"Traffic switch failed", error=str(e))
            return False
    
    async def rollback(self, previous_slot: str) -> bool:
        """Rollback to previous deployment slot."""
        
        logger.info(f"Rolling back to {previous_slot} slot")
        
        try:
            # Switch traffic back
            success = await self.switch_traffic(previous_slot, 100)
            
            if success:
                logger.info(f"Rollback to {previous_slot} completed successfully")
            else:
                logger.error(f"Rollback to {previous_slot} failed")
            
            return success
            
        except Exception as e:
            logger.error("Rollback failed", error=str(e))
            return False
    
    async def get_current_slot(self) -> str:
        """Get the currently active deployment slot."""
        try:
            cmd = [
                "kubectl", "get", "service", "model-serving-active",
                "-n", self.namespace_prod,
                "-o", "jsonpath={.spec.selector.deployment-slot}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                current_slot = result.stdout.strip()
                return current_slot if current_slot in ["blue", "green"] else "blue"
            else:
                logger.warning("Could not determine current slot, defaulting to blue")
                return "blue"
                
        except Exception as e:
            logger.error("Failed to get current slot", error=str(e))
            return "blue"
    
    async def deploy_with_validation(
        self,
        image_tag: str,
        services: List[str] = None,
        validation_timeout: int = 300,
        traffic_percentage: int = 10
    ) -> bool:
        """Deploy with validation and gradual traffic shift."""
        
        logger.info("Starting blue-green deployment with validation", image_tag=image_tag)
        
        try:
            # Determine current and target slots
            current_slot = await self.get_current_slot()
            target_slot = "green" if current_slot == "blue" else "blue"
            
            logger.info("Deployment slots determined", current=current_slot, target=target_slot)
            
            # Step 1: Deploy to target slot
            deploy_success = await self.deploy_to_slot(target_slot, image_tag, services)
            if not deploy_success:
                logger.error("Deployment failed, aborting")
                return False
            
            # Step 2: Validate deployment
            validation_success = await self.validate_deployment(target_slot, services)
            if not validation_success:
                logger.error("Validation failed, aborting deployment")
                return False
            
            # Step 3: Gradual traffic shift
            logger.info(f"Starting gradual traffic shift to {target_slot}")
            
            # Start with small percentage
            await self.switch_traffic(target_slot, traffic_percentage)
            await asyncio.sleep(60)  # Monitor for 1 minute
            
            # Check metrics and errors
            metrics_ok = await self._check_metrics(target_slot)
            if not metrics_ok:
                logger.error("Metrics check failed, rolling back")
                await self.rollback(current_slot)
                return False
            
            # Increase traffic gradually
            for percentage in [25, 50, 75, 100]:
                logger.info(f"Increasing traffic to {percentage}%")
                await self.switch_traffic(target_slot, percentage)
                await asyncio.sleep(120)  # Monitor for 2 minutes
                
                metrics_ok = await self._check_metrics(target_slot)
                if not metrics_ok:
                    logger.error(f"Metrics check failed at {percentage}%, rolling back")
                    await self.rollback(current_slot)
                    return False
            
            logger.info("Blue-green deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error("Blue-green deployment failed", error=str(e))
            # Attempt rollback
            current_slot = await self.get_current_slot()
            await self.rollback(current_slot)
            return False
    
    async def _check_metrics(self, slot: str) -> bool:
        """Check key metrics for the deployed slot."""
        try:
            # This would check Prometheus metrics
            # For now, just return True as a placeholder
            logger.info(f"Checking metrics for {slot} slot")
            
            # In production, this would:
            # 1. Query Prometheus for error rates
            # 2. Check latency percentiles
            # 3. Verify throughput is within expected ranges
            # 4. Check for any critical alerts
            
            return True
            
        except Exception as e:
            logger.error("Metrics check failed", error=str(e))
            return False


async def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Blue-green deployment for ML platform")
    parser.add_argument("--image-tag", required=True, help="Image tag to deploy")
    parser.add_argument("--services", nargs="+", help="Services to deploy")
    parser.add_argument("--validation-timeout", type=int, default=300, help="Validation timeout in seconds")
    parser.add_argument("--traffic-percentage", type=int, default=10, help="Initial traffic percentage")
    parser.add_argument("--rollback", action="store_true", help="Rollback to previous slot")
    parser.add_argument("--current-slot", help="Current slot for rollback")
    
    args = parser.parse_args()
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    deployer = BlueGreenDeployer()
    
    if args.rollback:
        if not args.current_slot:
            current_slot = await deployer.get_current_slot()
            previous_slot = "green" if current_slot == "blue" else "blue"
        else:
            previous_slot = args.current_slot
        
        success = await deployer.rollback(previous_slot)
        if success:
            print(f"✅ Rollback to {previous_slot} completed")
        else:
            print(f"❌ Rollback to {previous_slot} failed")
            exit(1)
    else:
        success = await deployer.deploy_with_validation(
            image_tag=args.image_tag,
            services=args.services,
            validation_timeout=args.validation_timeout,
            traffic_percentage=args.traffic_percentage
        )
        
        if success:
            print(f"✅ Blue-green deployment of {args.image_tag} completed successfully")
        else:
            print(f"❌ Blue-green deployment of {args.image_tag} failed")
            exit(1)


if __name__ == "__main__":
    asyncio.run(main())
