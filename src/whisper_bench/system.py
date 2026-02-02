"""Hardware monitoring module for resource usage during benchmarks."""

from __future__ import annotations

import os
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from subprocess import Popen

# Try to import pynvml for NVIDIA GPU monitoring
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class CpuInfo:
    """CPU information."""

    model: str
    cores_physical: int
    cores_logical: int


@dataclass
class GpuInfo:
    """GPU information."""

    name: str
    memory_total_mb: float
    driver_version: str


@dataclass
class SystemInfo:
    """Static system information collected once at start."""

    os_name: str
    os_version: str
    cpu: CpuInfo
    memory_total_gb: float
    gpu: GpuInfo | None = None


@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement."""

    timestamp: float
    cpu_percent: float
    memory_rss_mb: float
    cpu_temp_c: float | None = None
    gpu_utilization_percent: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_temp_c: float | None = None


@dataclass
class SampleHardwareMetrics:
    """Aggregated hardware metrics for a single sample."""

    cpu_percent_mean: float
    cpu_percent_max: float
    memory_rss_mean_mb: float
    memory_rss_peak_mb: float
    cpu_temp_max_c: float | None = None
    gpu_utilization_mean: float | None = None
    gpu_utilization_max: float | None = None
    gpu_memory_peak_mb: float | None = None
    gpu_temp_max_c: float | None = None


@dataclass
class AggregateHardwareMetrics:
    """Hardware metrics aggregated across all samples."""

    cpu_percent_mean: float
    cpu_percent_max: float
    memory_rss_mean_mb: float
    memory_rss_peak_mb: float
    cpu_temp_max_c: float | None = None
    gpu_utilization_mean: float | None = None
    gpu_utilization_max: float | None = None
    gpu_memory_peak_mb: float | None = None
    gpu_temp_max_c: float | None = None


def get_cpu_model() -> str:
    """Get CPU model name, with ARM/Raspberry Pi support."""
    # Try /proc/cpuinfo first (Linux)
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        content = cpuinfo_path.read_text()
        # ARM devices (Raspberry Pi) use "Model" or "Hardware" fields
        for line in content.split("\n"):
            if line.startswith("Model") and ":" in line:
                return line.split(":", 1)[1].strip()
            if line.startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
        # Fallback for ARM
        for line in content.split("\n"):
            if line.startswith("Hardware") and ":" in line:
                return line.split(":", 1)[1].strip()

    # macOS
    if platform.system() == "Darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

    return platform.processor() or "Unknown"


def get_cpu_temperature() -> float | None:
    """Get current CPU temperature in Celsius.

    Cross-platform implementation with Raspberry Pi support.
    """
    # 1. Try Raspberry Pi thermal zone (ARM Linux)
    rpi_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if rpi_path.exists():
        try:
            return int(rpi_path.read_text().strip()) / 1000.0
        except (ValueError, OSError):
            pass

    # 2. Try psutil sensors (x86 Linux)
    if hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Try common sensor names
                for name in ["coretemp", "k10temp", "cpu_thermal", "cpu-thermal"]:
                    if name in temps and temps[name]:
                        return max(r.current for r in temps[name])
                # Fallback: return first available sensor
                for sensors in temps.values():
                    if sensors:
                        return max(r.current for r in sensors)
        except Exception:
            pass

    # 3. Unavailable
    return None


def get_gpu_info() -> GpuInfo | None:
    """Get NVIDIA GPU information if available."""
    if not PYNVML_AVAILABLE:
        return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8")

        return GpuInfo(
            name=name,
            memory_total_mb=memory.total / (1024 * 1024),
            driver_version=driver,
        )
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def get_gpu_metrics() -> tuple[float | None, float | None, float | None]:
    """Get current GPU utilization, memory used, and temperature.

    Returns:
        Tuple of (utilization_percent, memory_used_mb, temperature_c)
    """
    if not PYNVML_AVAILABLE:
        return None, None, None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            temp = None

        return (
            float(utilization.gpu),
            memory.used / (1024 * 1024),
            float(temp) if temp is not None else None,
        )
    except Exception:
        return None, None, None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def get_system_info() -> SystemInfo:
    """Collect static system information once at startup."""
    cpu_info = CpuInfo(
        model=get_cpu_model(),
        cores_physical=psutil.cpu_count(logical=False) or 1,
        cores_logical=psutil.cpu_count(logical=True) or 1,
    )

    memory = psutil.virtual_memory()

    return SystemInfo(
        os_name=platform.system(),
        os_version=platform.release(),
        cpu=cpu_info,
        memory_total_gb=memory.total / (1024**3),
        gpu=get_gpu_info(),
    )


class ResourceMonitor:
    """Context manager for monitoring resources during subprocess execution.

    Usage:
        with ResourceMonitor(interval_ms=100) as monitor:
            subprocess.run(cmd)
        metrics = monitor.get_metrics()
    """

    def __init__(
        self,
        interval_ms: int = 100,
        process: Popen | None = None,
    ):
        self.interval_ms = interval_ms
        self.process = process
        self._snapshots: list[ResourceSnapshot] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._target_pid: int | None = None

    def start(self, pid: int | None = None) -> None:
        """Start monitoring. Optionally track a specific PID."""
        self._target_pid = pid
        self._stop_event.clear()
        self._snapshots = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def _monitor_loop(self) -> None:
        """Background thread that collects resource snapshots."""
        interval_sec = self.interval_ms / 1000.0

        while not self._stop_event.is_set():
            try:
                snapshot = self._take_snapshot()
                if snapshot:
                    self._snapshots.append(snapshot)
            except Exception:
                pass  # Ignore errors in monitoring

            self._stop_event.wait(interval_sec)

    def _take_snapshot(self) -> ResourceSnapshot | None:
        """Take a single resource measurement."""
        try:
            # Get overall CPU percent
            cpu_percent = psutil.cpu_percent(interval=None)

            # Get memory for target process or current process
            if self._target_pid:
                try:
                    proc = psutil.Process(self._target_pid)
                    memory_rss = proc.memory_info().rss / (1024 * 1024)
                except psutil.NoSuchProcess:
                    memory_rss = 0.0
            else:
                # Monitor all child processes of current process
                current = psutil.Process()
                memory_rss = current.memory_info().rss / (1024 * 1024)
                for child in current.children(recursive=True):
                    try:
                        memory_rss += child.memory_info().rss / (1024 * 1024)
                    except psutil.NoSuchProcess:
                        pass

            cpu_temp = get_cpu_temperature()
            gpu_util, gpu_mem, gpu_temp = get_gpu_metrics()

            return ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_rss_mb=memory_rss,
                cpu_temp_c=cpu_temp,
                gpu_utilization_percent=gpu_util,
                gpu_memory_used_mb=gpu_mem,
                gpu_temp_c=gpu_temp,
            )
        except Exception:
            return None

    def get_metrics(self) -> SampleHardwareMetrics | None:
        """Compute aggregated metrics from collected snapshots."""
        if not self._snapshots:
            return None

        cpu_percents = [s.cpu_percent for s in self._snapshots]
        memory_rss = [s.memory_rss_mb for s in self._snapshots]
        cpu_temps = [s.cpu_temp_c for s in self._snapshots if s.cpu_temp_c is not None]
        gpu_utils = [
            s.gpu_utilization_percent
            for s in self._snapshots
            if s.gpu_utilization_percent is not None
        ]
        gpu_mems = [
            s.gpu_memory_used_mb
            for s in self._snapshots
            if s.gpu_memory_used_mb is not None
        ]
        gpu_temps = [s.gpu_temp_c for s in self._snapshots if s.gpu_temp_c is not None]

        return SampleHardwareMetrics(
            cpu_percent_mean=sum(cpu_percents) / len(cpu_percents),
            cpu_percent_max=max(cpu_percents),
            memory_rss_mean_mb=sum(memory_rss) / len(memory_rss),
            memory_rss_peak_mb=max(memory_rss),
            cpu_temp_max_c=max(cpu_temps) if cpu_temps else None,
            gpu_utilization_mean=sum(gpu_utils) / len(gpu_utils) if gpu_utils else None,
            gpu_utilization_max=max(gpu_utils) if gpu_utils else None,
            gpu_memory_peak_mb=max(gpu_mems) if gpu_mems else None,
            gpu_temp_max_c=max(gpu_temps) if gpu_temps else None,
        )


def compute_aggregate_hardware_metrics(
    samples: list[SampleHardwareMetrics],
) -> AggregateHardwareMetrics | None:
    """Aggregate hardware metrics across all samples."""
    if not samples:
        return None

    cpu_means = [s.cpu_percent_mean for s in samples]
    cpu_maxes = [s.cpu_percent_max for s in samples]
    mem_means = [s.memory_rss_mean_mb for s in samples]
    mem_peaks = [s.memory_rss_peak_mb for s in samples]
    cpu_temps = [s.cpu_temp_max_c for s in samples if s.cpu_temp_max_c is not None]
    gpu_utils_mean = [
        s.gpu_utilization_mean for s in samples if s.gpu_utilization_mean is not None
    ]
    gpu_utils_max = [
        s.gpu_utilization_max for s in samples if s.gpu_utilization_max is not None
    ]
    gpu_mem_peaks = [
        s.gpu_memory_peak_mb for s in samples if s.gpu_memory_peak_mb is not None
    ]
    gpu_temps = [s.gpu_temp_max_c for s in samples if s.gpu_temp_max_c is not None]

    return AggregateHardwareMetrics(
        cpu_percent_mean=sum(cpu_means) / len(cpu_means),
        cpu_percent_max=max(cpu_maxes),
        memory_rss_mean_mb=sum(mem_means) / len(mem_means),
        memory_rss_peak_mb=max(mem_peaks),
        cpu_temp_max_c=max(cpu_temps) if cpu_temps else None,
        gpu_utilization_mean=(
            sum(gpu_utils_mean) / len(gpu_utils_mean) if gpu_utils_mean else None
        ),
        gpu_utilization_max=max(gpu_utils_max) if gpu_utils_max else None,
        gpu_memory_peak_mb=max(gpu_mem_peaks) if gpu_mem_peaks else None,
        gpu_temp_max_c=max(gpu_temps) if gpu_temps else None,
    )
