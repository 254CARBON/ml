"""GPU detection and allocation for embedding generation."""

import os
import platform
import subprocess
from typing import Dict, List, Optional, Tuple
import torch
import structlog

logger = structlog.get_logger("gpu_detector")


class GPUDetector:
    """Detects and manages GPU resources."""
    
    def __init__(self):
        """Prepare GPU detection state and caches."""
        self.gpu_info: Dict[str, Any] = {}
        self.available_devices: List[str] = []
        self.current_device: Optional[str] = None
        self._detection_complete = False
    
    def detect_gpus(self) -> Dict[str, Any]:
        """Detect available GPU resources."""
        if self._detection_complete:
            return self.gpu_info
        
        gpu_info = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "cuda_available": False,
            "mps_available": False,  # Apple Metal Performance Shaders
            "gpu_count": 0,
            "devices": [],
            "recommended_device": "cpu"
        }
        
        try:
            # Check CUDA availability
            if torch.cuda.is_available():
                gpu_info["cuda_available"] = True
                gpu_info["gpu_count"] = torch.cuda.device_count()
                
                for i in range(gpu_info["gpu_count"]):
                    device_props = torch.cuda.get_device_properties(i)
                    device_info = {
                        "id": i,
                        "name": device_props.name,
                        "memory_total": device_props.total_memory,
                        "memory_free": torch.cuda.get_device_properties(i).total_memory,
                        "compute_capability": f"{device_props.major}.{device_props.minor}",
                        "type": "cuda"
                    }
                    gpu_info["devices"].append(device_info)
                    self.available_devices.append(f"cuda:{i}")
                
                gpu_info["recommended_device"] = "cuda:0"
                
            # Check MPS (Apple Silicon) availability
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info["mps_available"] = True
                gpu_info["gpu_count"] = 1
                
                device_info = {
                    "id": 0,
                    "name": "Apple GPU",
                    "memory_total": self._get_apple_gpu_memory(),
                    "memory_free": self._get_apple_gpu_memory(),
                    "compute_capability": "mps",
                    "type": "mps"
                }
                gpu_info["devices"].append(device_info)
                self.available_devices.append("mps")
                gpu_info["recommended_device"] = "mps"
            
            # Fallback to CPU
            if not gpu_info["cuda_available"] and not gpu_info["mps_available"]:
                gpu_info["recommended_device"] = "cpu"
                self.available_devices.append("cpu")
        
        except Exception as e:
            logger.error("GPU detection failed", error=str(e))
            gpu_info["recommended_device"] = "cpu"
            self.available_devices.append("cpu")
        
        self.gpu_info = gpu_info
        self._detection_complete = True
        
        logger.info(
            "GPU detection completed",
            cuda_available=gpu_info["cuda_available"],
            mps_available=gpu_info["mps_available"],
            gpu_count=gpu_info["gpu_count"],
            recommended_device=gpu_info["recommended_device"]
        )
        
        return gpu_info
    
    def _get_apple_gpu_memory(self) -> int:
        """Estimate Apple GPU memory."""
        try:
            # Try to get system memory as a proxy
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    total_memory = int(result.stdout.strip())
                    # Estimate GPU memory as fraction of total memory
                    return int(total_memory * 0.6)  # Rough estimate
        except Exception:
            pass
        
        return 8 * 1024 * 1024 * 1024  # 8GB default estimate
    
    def select_device(self, preference: str = "auto") -> str:
        """Select the best available device."""
        if not self._detection_complete:
            self.detect_gpus()
        
        if preference == "cpu":
            self.current_device = "cpu"
        elif preference == "gpu":
            if self.gpu_info["cuda_available"]:
                self.current_device = "cuda:0"
            elif self.gpu_info["mps_available"]:
                self.current_device = "mps"
            else:
                logger.warning("GPU requested but not available, falling back to CPU")
                self.current_device = "cpu"
        else:  # auto
            self.current_device = self.gpu_info["recommended_device"]
        
        logger.info("Device selected", device=self.current_device, preference=preference)
        return self.current_device
    
    def get_device_info(self, device: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific device."""
        if not self._detection_complete:
            self.detect_gpus()
        
        if device == "cpu":
            return {
                "id": "cpu",
                "name": "CPU",
                "type": "cpu",
                "memory_total": "system_memory",
                "memory_free": "system_memory"
            }
        
        for device_info in self.gpu_info["devices"]:
            if device == f"{device_info['type']}:{device_info['id']}" or device == device_info['type']:
                return device_info
        
        return None
    
    def get_memory_usage(self, device: str) -> Tuple[int, int]:
        """Get memory usage for a device (used, total)."""
        try:
            if device.startswith("cuda:"):
                device_id = int(device.split(":")[1])
                return torch.cuda.memory_allocated(device_id), torch.cuda.max_memory_allocated(device_id)
            elif device == "mps":
                # MPS doesn't have direct memory query, return estimates
                return 0, self._get_apple_gpu_memory()
            else:
                # CPU - return system memory info
                import psutil
                memory = psutil.virtual_memory()
                return memory.used, memory.total
                
        except Exception as e:
            logger.warning("Failed to get memory usage", device=device, error=str(e))
            return 0, 0
    
    def optimize_batch_size(self, device: str, base_batch_size: int = 32) -> int:
        """Optimize batch size based on available memory."""
        try:
            if device == "cpu":
                # For CPU, use larger batch sizes
                return min(base_batch_size * 4, 256)
            
            elif device.startswith("cuda:"):
                # For CUDA, check available memory
                device_id = int(device.split(":")[1])
                memory_free = torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_allocated(device_id)
                
                # Rough estimation: 1GB can handle ~32 batch size for typical embeddings
                memory_gb = memory_free / (1024 ** 3)
                optimal_batch_size = int(memory_gb * 32)
                
                return min(max(optimal_batch_size, 8), 512)
            
            elif device == "mps":
                # For Apple Silicon, use moderate batch sizes
                return min(base_batch_size * 2, 128)
            
            else:
                return base_batch_size
                
        except Exception as e:
            logger.warning("Batch size optimization failed", device=device, error=str(e))
            return base_batch_size
    
    def clear_cache(self, device: str):
        """Clear GPU cache if applicable."""
        try:
            if device.startswith("cuda:"):
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared", device=device)
            elif device == "mps":
                # MPS cache clearing
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                logger.info("MPS cache cleared", device=device)
        except Exception as e:
            logger.warning("Failed to clear cache", device=device, error=str(e))


# Global GPU detector instance
_gpu_detector: Optional[GPUDetector] = None


def get_gpu_detector() -> GPUDetector:
    """Get or create GPU detector instance."""
    global _gpu_detector
    if _gpu_detector is None:
        _gpu_detector = GPUDetector()
    return _gpu_detector


def detect_optimal_device(preference: str = "auto") -> str:
    """Detect optimal device for ML workloads."""
    detector = get_gpu_detector()
    return detector.select_device(preference)


def get_device_memory_info(device: str) -> Tuple[int, int]:
    """Get device memory information."""
    detector = get_gpu_detector()
    return detector.get_memory_usage(device)


def optimize_batch_size_for_device(device: str, base_batch_size: int = 32) -> int:
    """Optimize batch size for the given device."""
    detector = get_gpu_detector()
    return detector.optimize_batch_size(device, base_batch_size)
