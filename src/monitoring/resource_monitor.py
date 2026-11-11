"""System resource monitoring using psutil.

Monitors CPU, RAM, GPU (where available), process memory, and disk usage.
GPU metrics are reported as unavailable when no GPU monitoring interface is
detected -- never replaced with invented values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


@dataclass
class ResourceSnapshot:
    """A point-in-time snapshot of system resources."""

    cpu_percent: Optional[float] = None
    ram_percent: Optional[float] = None
    ram_available_mb: Optional[float] = None
    process_memory_mb: Optional[float] = None
    disk_percent: Optional[float] = None
    gpu_available: bool = False
    gpu_utilization: Optional[float] = None
    gpu_memory_mb: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "ram_percent": self.ram_percent,
            "ram_available_mb": self.ram_available_mb,
            "process_memory_mb": self.process_memory_mb,
            "disk_percent": self.disk_percent,
            "gpu_available": self.gpu_available,
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_mb": self.gpu_memory_mb,
        }


class ResourceMonitor:
    """Monitors system resources via psutil and optional GPU interfaces."""

    def __init__(self, check_gpu: bool = True) -> None:
        """Initialize the resource monitor.

        Args:
            check_gpu: Whether to attempt GPU monitoring via PyTorch or
                nvidia-smi when available.
        """
        self._check_gpu = check_gpu
        self._history: list[ResourceSnapshot] = []

    def is_psutil_available(self) -> bool:
        """Return True if psutil is available."""
        return _PSUTIL_AVAILABLE

    def get_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current system resources."""
        snapshot = ResourceSnapshot()

        if _PSUTIL_AVAILABLE:
            try:
                snapshot.cpu_percent = psutil.cpu_percent(interval=None)
            except Exception:
                pass
            try:
                vm = psutil.virtual_memory()
                snapshot.ram_percent = vm.percent
                snapshot.ram_available_mb = vm.available / (1024 * 1024)
            except Exception:
                pass
            try:
                proc = psutil.Process()
                snapshot.process_memory_mb = (
                    proc.memory_info().rss / (1024 * 1024)
                )
            except Exception:
                pass
            try:
                du = psutil.disk_usage("/")
                snapshot.disk_percent = du.percent
            except Exception:
                pass

        if self._check_gpu:
            gpu = self._check_gpu_metrics()
            snapshot.gpu_available = gpu["available"]
            snapshot.gpu_utilization = gpu.get("utilization")
            snapshot.gpu_memory_mb = gpu.get("memory_used_mb")

        self._history.append(snapshot)
        return snapshot

    def get_history(self) -> list[ResourceSnapshot]:
        """Return the snapshot history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear snapshot history."""
        self._history.clear()

    def get_average(self, attr: str) -> Optional[float]:
        """Return the average value of a resource attribute over history."""
        values = [
            getattr(s, attr)
            for s in self._history
            if getattr(s, attr) is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def _check_gpu_metrics(self) -> dict[str, Any]:
        """Check GPU metrics via PyTorch if available."""
        try:
            import torch

            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                props = torch.cuda.get_device_properties(device)
                mem_used = torch.cuda.memory_allocated(device) / (1024 * 1024)
                mem_total = props.total_memory / (1024 * 1024)
                # Utilization is not directly available via torch API.
                return {
                    "available": True,
                    "utilization": None,
                    "memory_used_mb": mem_used,
                    "memory_total_mb": mem_total,
                }
        except Exception:
            pass
        return {"available": False, "utilization": None, "memory_used_mb": None}
