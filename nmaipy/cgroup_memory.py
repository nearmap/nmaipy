"""
Cgroup-aware resource utilities for containerized environments.

In Kubernetes pods, psutil reports the host node's resources,
not the container's cgroup limits. This module provides functions to read
the actual container resource limits from cgroup files.

Supports both cgroup v1 and v2 (unified hierarchy).
Currently covers memory and CPU resources.
"""

import logging
import os
from typing import Optional, Tuple

import psutil

logger = logging.getLogger(__name__)

# Cgroup v2 paths (unified hierarchy, newer K8s)
CGROUP_V2_MEMORY_MAX = "/sys/fs/cgroup/memory.max"
CGROUP_V2_MEMORY_CURRENT = "/sys/fs/cgroup/memory.current"
CGROUP_V2_CPU_MAX = "/sys/fs/cgroup/cpu.max"

# Cgroup v1 paths (legacy, older K8s)
CGROUP_V1_MEMORY_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
CGROUP_V1_MEMORY_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
CGROUP_V1_CPU_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
CGROUP_V1_CPU_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


def is_running_in_container() -> bool:
    """
    Detect if we're running inside a container with cgroup memory limits.

    Returns:
        True if cgroup memory files are accessible, False otherwise.
    """
    return os.path.exists(CGROUP_V2_MEMORY_MAX) or os.path.exists(
        CGROUP_V1_MEMORY_LIMIT
    )


def get_cgroup_memory_limit_bytes() -> Optional[int]:
    """
    Get the container's memory limit from cgroup files.

    Tries cgroup v2 first, then falls back to v1.

    Returns:
        Memory limit in bytes, or None if not in a container or unlimited.
    """
    # Try cgroup v2 first (newer K8s versions)
    if os.path.exists(CGROUP_V2_MEMORY_MAX):
        try:
            with open(CGROUP_V2_MEMORY_MAX, "r") as f:
                value = f.read().strip()
                if value == "max":
                    # "max" means no limit set
                    return None
                limit = int(value)
                # Check if effectively unlimited (very large number)
                if limit > 1e18:
                    return None
                return limit
        except (IOError, ValueError) as e:
            logger.debug(f"Could not read cgroup v2 memory limit: {e}")

    # Try cgroup v1 (older K8s versions)
    if os.path.exists(CGROUP_V1_MEMORY_LIMIT):
        try:
            with open(CGROUP_V1_MEMORY_LIMIT, "r") as f:
                limit = int(f.read().strip())
                # Check if effectively unlimited (very large number, often 9223372036854771712)
                if limit > 1e18:
                    return None
                return limit
        except (IOError, ValueError) as e:
            logger.debug(f"Could not read cgroup v1 memory limit: {e}")

    return None


def get_cgroup_memory_usage_bytes() -> Optional[int]:
    """
    Get the container's current memory usage from cgroup files.

    Tries cgroup v2 first, then falls back to v1.

    Returns:
        Current memory usage in bytes, or None if not in a container.
    """
    # Try cgroup v2 first
    if os.path.exists(CGROUP_V2_MEMORY_CURRENT):
        try:
            with open(CGROUP_V2_MEMORY_CURRENT, "r") as f:
                return int(f.read().strip())
        except (IOError, ValueError) as e:
            logger.debug(f"Could not read cgroup v2 memory usage: {e}")

    # Try cgroup v1
    if os.path.exists(CGROUP_V1_MEMORY_USAGE):
        try:
            with open(CGROUP_V1_MEMORY_USAGE, "r") as f:
                return int(f.read().strip())
        except (IOError, ValueError) as e:
            logger.debug(f"Could not read cgroup v1 memory usage: {e}")

    return None


def get_cgroup_memory_limit_gb() -> Optional[float]:
    """
    Get the container's memory limit in GB.

    Returns:
        Memory limit in GB, or None if not in a container or unlimited.
    """
    limit_bytes = get_cgroup_memory_limit_bytes()
    if limit_bytes is None:
        return None
    return limit_bytes / (1024**3)


def get_cgroup_memory_usage_gb() -> Optional[float]:
    """
    Get the container's current memory usage in GB.

    Returns:
        Current memory usage in GB, or None if not in a container.
    """
    usage_bytes = get_cgroup_memory_usage_bytes()
    if usage_bytes is None:
        return None
    return usage_bytes / (1024**3)


def get_memory_info_cgroup_aware() -> Tuple[float, float]:
    """
    Get memory usage and total, respecting cgroup limits when in a container.

    Returns:
        Tuple of (used_gb, total_gb) - uses cgroup values in containers,
        falls back to psutil for bare metal.
    """
    if is_running_in_container():
        usage_gb = get_cgroup_memory_usage_gb()
        limit_gb = get_cgroup_memory_limit_gb()
        if usage_gb is not None and limit_gb is not None:
            return (usage_gb, limit_gb)

    # Fall back to psutil
    mem = psutil.virtual_memory()
    used_gb = (mem.total - mem.available) / (1024**3)
    total_gb = mem.total / (1024**3)
    return (used_gb, total_gb)


def get_cgroup_cpu_limit() -> Optional[float]:
    """
    Get the container's CPU limit from cgroup files.

    Tries cgroup v2 first, then falls back to v1.

    Returns:
        Number of CPUs allocated (e.g., 2.0 for quota=200000/period=100000),
        or None if not in a container or unlimited.
    """
    # Try cgroup v2 first
    if os.path.exists(CGROUP_V2_CPU_MAX):
        try:
            with open(CGROUP_V2_CPU_MAX, "r") as f:
                value = f.read().strip()
                parts = value.split()
                if len(parts) == 2:
                    max_val, period = parts
                    if max_val == "max":
                        return None  # Unlimited
                    return int(max_val) / int(period)
        except (IOError, ValueError, ZeroDivisionError) as e:
            logger.debug(f"Could not read cgroup v2 CPU limit: {e}")

    # Try cgroup v1
    if os.path.exists(CGROUP_V1_CPU_QUOTA):
        try:
            with open(CGROUP_V1_CPU_QUOTA, "r") as f:
                quota = int(f.read().strip())
            if quota == -1:
                return None  # Unlimited
            period = 100000  # Default period
            if os.path.exists(CGROUP_V1_CPU_PERIOD):
                with open(CGROUP_V1_CPU_PERIOD, "r") as f:
                    period = int(f.read().strip())
            return quota / period
        except (IOError, ValueError, ZeroDivisionError) as e:
            logger.debug(f"Could not read cgroup v1 CPU limit: {e}")

    return None


def get_cpu_info_cgroup_aware() -> Tuple[float, float]:
    """
    Get CPU usage percentage and effective CPU count, respecting cgroup limits.

    cpu_percent is system-wide utilization (0-100%), normalized across all cores.
    100% means all cores fully utilized. On a 16-core machine at 50%, roughly
    8 cores worth of work is being done. This is the metric to watch when
    deciding if more parallelism can be pushed.

    Note: psutil.cpu_percent(interval=None) returns 0.0 on its first call
    (needs a baseline). Call psutil.cpu_percent(interval=None) once at startup
    to prime it before using this function.

    Returns:
        Tuple of (cpu_percent, cpu_count) where:
        - cpu_percent: System-wide CPU usage 0-100% (non-blocking)
        - cpu_count: Effective number of CPUs (cgroup-aware in containers)
    """
    cpu_pct = psutil.cpu_percent(interval=None)

    if is_running_in_container():
        cgroup_cpus = get_cgroup_cpu_limit()
        if cgroup_cpus is not None:
            return (cpu_pct, cgroup_cpus)

    return (cpu_pct, float(psutil.cpu_count() or 1))
