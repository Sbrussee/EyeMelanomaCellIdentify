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
