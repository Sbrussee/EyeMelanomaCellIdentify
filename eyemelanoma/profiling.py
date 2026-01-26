"""Resource profiling helpers for memory, I/O, and GPU usage."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

import resource


@dataclass(frozen=True)
class ResourceProfile:
    """Resource usage measurements for a single sampling window."""

    max_rss_mb: float
    io_write_mb_s: float
    io_read_mb_s: float
    gpu_memory_mb: float | None


def _safe_int(value: str, default: int) -> int:
    """Parse an integer from a string, returning a default on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ResourceLogger:
    """Configurable logger for resource usage snapshots."""

    enabled: bool
    log_every: int
    sample_size_mb: int = 1

    @classmethod
    def from_env(cls) -> "ResourceLogger":
        """Build a logger configured by environment variables."""
        enabled = os.getenv("EYEMELANOMA_PROFILE_RESOURCES", "0") == "1"
        log_every = _safe_int(os.getenv("EYEMELANOMA_PROFILE_EVERY", "2000"), 2000)
        sample_size_mb = _safe_int(os.getenv("EYEMELANOMA_PROFILE_IO_MB", "1"), 1)
        return cls(
            enabled=enabled,
            log_every=max(1, log_every),
            sample_size_mb=max(1, sample_size_mb),
        )

    def log(self, stage: str) -> None:
        """Log a resource usage snapshot for a pipeline stage."""
        if not self.enabled:
            return
        profile = profile_resource_usage(sample_size_mb=self.sample_size_mb)
        gpu = profile.gpu_memory_mb if profile.gpu_memory_mb is not None else "n/a"
        print(
            f"[RESOURCE] {stage}: max_rss_mb={profile.max_rss_mb:.1f}, "
            f"io_write_mb_s={profile.io_write_mb_s:.1f}, io_read_mb_s={profile.io_read_mb_s:.1f}, "
            f"gpu_mem_mb={gpu}"
        )

    def maybe_log_every(self, stage_prefix: str, index: int) -> None:
        """Log resource usage every ``log_every`` iterations for a stage prefix."""
        if not self.enabled:
            return
        if index > 0 and index % self.log_every == 0:
            self.log(f"{stage_prefix}:{index}")


def _bytes_to_mb(value: float) -> float:
    """Convert bytes to megabytes."""
    return float(value) / (1024.0 * 1024.0)


def _measure_io_throughput_mb_s(sample_size_mb: int) -> tuple[float, float]:
    """Measure approximate I/O throughput by writing and reading a temp file."""
    sample_size_mb = max(1, int(sample_size_mb))
    sample_bytes = sample_size_mb * 1024 * 1024
    payload = b"0" * sample_bytes
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "io_sample.bin")
        start = time.perf_counter()
        with open(path, "wb") as handle:
            handle.write(payload)
        write_time = max(time.perf_counter() - start, 1e-6)

        start = time.perf_counter()
        with open(path, "rb") as handle:
            _ = handle.read()
        read_time = max(time.perf_counter() - start, 1e-6)

    write_mb_s = float(sample_size_mb) / write_time
    read_mb_s = float(sample_size_mb) / read_time
    return write_mb_s, read_mb_s


def _measure_gpu_memory_mb() -> float | None:
    """Report current GPU memory usage if PyTorch is available."""
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None
    return _bytes_to_mb(float(torch.cuda.memory_allocated()))


def profile_resource_usage(sample_size_mb: int = 1) -> ResourceProfile:
    """Collect a lightweight snapshot of memory, I/O, and GPU usage."""
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_mb = float(rss_kb) / 1024.0
    write_mb_s, read_mb_s = _measure_io_throughput_mb_s(sample_size_mb)
    gpu_memory_mb = _measure_gpu_memory_mb()
    return ResourceProfile(
        max_rss_mb=max_rss_mb,
        io_write_mb_s=write_mb_s,
        io_read_mb_s=read_mb_s,
        gpu_memory_mb=gpu_memory_mb,
    )
