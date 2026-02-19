"""
Base Exporter class for parallel chunked processing of AOI data.

Provides common infrastructure for exporters that need to:
- Split large datasets into manageable chunks
- Process chunks in parallel using multiprocessing
- Track progress with shared counters across processes
- Handle process pool failures with retry logic
- Monitor memory usage
- Cache chunk results to avoid redundant work

Subclasses implement process_chunk() to define API-specific processing logic.
"""

import concurrent.futures
import json
import multiprocessing
import os
import platform
import sys
import tempfile
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import psutil
from tqdm import tqdm

from nmaipy import log, storage
from nmaipy.cgroup_memory import (
    get_cgroup_cpu_limit,
    get_cgroup_memory_limit_gb,
    get_cgroup_memory_usage_gb,
    get_cpu_info_cgroup_aware,
    get_memory_info_cgroup_aware,
    is_running_in_container,
)
from nmaipy.constants import API_WARMUP_INTERVAL_SECONDS

logger = log.get_logger()


class BaseExporter(ABC):
    """
    Abstract base class for exporters with chunked parallel processing.

    Provides infrastructure for:
    - Chunking input GeoDataFrames with cache checking
    - Parallel processing with ProcessPoolExecutor
    - Shared progress counters across worker processes
    - Dynamic progress bars that update during processing
    - Retry logic for process pool failures
    - Memory usage monitoring
    - Managing chunk and final output directories

    Subclasses must implement:
    - process_chunk(): Process a single chunk of data
    - get_chunk_output_file(): Return path to chunk output file for cache checking
    """

    # Shared tqdm configuration for progress bars. The "+" in the format indicates
    # total can increase during gridding.
    _TQDM_CONFIG = dict(
        desc="API requests",
        file=sys.stdout,
        position=0,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}+ [{elapsed}<{remaining}, {rate_fmt}]",
        mininterval=5.0,
        maxinterval=10.0,
        smoothing=0.1,
        unit=" requests",
    )

    @staticmethod
    def _format_progress_description(chunks_done, chunks_total, lat_str="Warmup"):
        """Build progress bar description with memory and CPU info."""
        used_gb, total_gb = get_memory_info_cgroup_aware()
        cpu_pct, cpu_count = get_cpu_info_cgroup_aware()
        cpu_count_str = f"{cpu_count:.0f}" if cpu_count == int(cpu_count) else f"{cpu_count:.1f}"
        return (
            f"API requests - {used_gb:.1f}/{total_gb:.1f}GB | "
            f"CPU {cpu_pct:.0f}% ({cpu_count_str}) | "
            f"{lat_str} | Chunks: {chunks_done}/{chunks_total}"
        )

    def __init__(
        self,
        output_dir: str,
        processes: int = 4,
        chunk_size: int = 500,
        log_level: str = "INFO",
    ):
        """
        Initialize BaseExporter.

        Args:
            output_dir: Directory for output files
            processes: Number of parallel processes for chunk processing
            chunk_size: Number of AOIs to process in a single chunk
            log_level: Logging level
        """
        self.output_dir = str(output_dir)
        self.is_s3_output = storage.is_s3_path(self.output_dir)
        self.processes = processes
        self.chunk_size = chunk_size
        self.log_level = log_level

        # Configure logging
        log.configure_logger(self.log_level)
        self.logger = log.get_logger()

        # Create output directories (no-op for S3 where directories are virtual)
        self.chunk_path = storage.join_path(self.output_dir, "chunks")
        self.final_path = storage.join_path(self.output_dir, "final")
        storage.ensure_directory(self.output_dir)
        storage.ensure_directory(self.chunk_path)
        storage.ensure_directory(self.final_path)

        # For S3 output, create a local staging directory for files that require
        # local I/O (e.g. pyarrow ParquetWriter for streaming large geoparquet)
        if self.is_s3_output:
            self._local_staging_dir = tempfile.mkdtemp(prefix="nmaipy_staging_")
            self._local_final_staging = os.path.join(self._local_staging_dir, "final")
            os.makedirs(self._local_final_staging, exist_ok=True)
        else:
            self._local_staging_dir = None
            self._local_final_staging = None

    def _save_config(self, config: Dict[str, Any], config_name: str = "export_config.json"):
        """
        Save export configuration to the final output directory.

        Creates a JSON file with all export parameters and metadata, useful for
        reproducibility and debugging. Saved at the start of export so it's
        available even if the export fails.

        Args:
            config: Dictionary of export parameters
            config_name: Name of the config file (default: export_config.json)
        """
        try:
            import nmaipy

            nmaipy_version = getattr(nmaipy, "__version__", "unknown")
        except Exception:
            nmaipy_version = "unknown"

        # Build metadata
        metadata = {
            "export_started_at": datetime.now(timezone.utc).isoformat(),
            "nmaipy_version": nmaipy_version,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }

        # Combine metadata and config
        full_config = {
            "_metadata": metadata,
            "parameters": config,
        }

        config_path = storage.join_path(self.final_path, config_name)

        try:
            storage.write_json(config_path, full_config, indent=2)
            self.logger.debug(f"Saved export config to {config_path}")
        except Exception as e:
            self.logger.warning(f"Could not save export config: {e}")

    @abstractmethod
    def process_chunk(self, chunk_id: str, aoi_gdf: gpd.GeoDataFrame, **kwargs):
        """
        Process a single chunk of AOI data.

        This method must be implemented by subclasses to define their specific
        processing logic (e.g., calling different APIs, applying transformations).

        Args:
            chunk_id: Unique identifier for this chunk
            aoi_gdf: GeoDataFrame containing AOIs to process
            **kwargs: Additional parameters (e.g., classes_df, progress_counters)

        Returns:
            Implementation-specific return value (often None, with results saved to files)

        Raises:
            Any exceptions should be raised to be caught by run_parallel()
        """
        pass

    @abstractmethod
    def get_chunk_output_file(self, chunk_id: str) -> str:
        """
        Get the path to the main output file for a chunk.

        Used for cache checking to skip already-processed chunks.

        Args:
            chunk_id: Unique identifier for this chunk

        Returns:
            Path to the chunk's main output file (string, may be S3 URI)
        """
        pass

    def split_into_chunks(
        self, aoi_gdf: gpd.GeoDataFrame, file_stem: str, check_cache: bool = True
    ) -> Tuple[List[Tuple[int, gpd.GeoDataFrame]], int, int]:
        """
        Split a GeoDataFrame into chunks for parallel processing.

        Optionally filters out already-processed chunks by checking for existing output files.

        Args:
            aoi_gdf: GeoDataFrame to split
            file_stem: Base filename for chunk IDs
            check_cache: If True, skip chunks with existing output files

        Returns:
            Tuple of:
            - List of (chunk_index, chunk_gdf) tuples to process
            - Number of skipped chunks
            - Number of skipped AOIs
        """
        # Filter AOIs with zero area if geometry-based
        if isinstance(aoi_gdf, gpd.GeoDataFrame):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Geometry is in a geographic CRS.",
                )
                aoi_gdf = aoi_gdf[aoi_gdf.area > 0]

        num_chunks = max(len(aoi_gdf) // self.chunk_size, 1)
        self.logger.info(
            f"Processing {len(aoi_gdf)} AOIs using {self.processes} processes, "
            f"divided into {num_chunks} chunks (chunk_size={self.chunk_size})"
        )

        # Split into chunks
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'GeoDataFrame.swapaxes' is deprecated",
            )
            warnings.filterwarnings(
                "ignore",
                message="'DataFrame.swapaxes' is deprecated",
            )
            chunks = np.array_split(aoi_gdf, num_chunks)

        # Filter out cached chunks if requested
        chunks_to_process = []
        skipped_chunks = 0
        skipped_aois = 0

        for i, batch in enumerate(chunks):
            chunk_id = f"{file_stem}_{str(i).zfill(4)}"
            if check_cache:
                outfile = self.get_chunk_output_file(chunk_id)
                if storage.file_exists(outfile):
                    skipped_chunks += 1
                    skipped_aois += len(batch)
                    continue
            chunks_to_process.append((i, batch))

        if skipped_chunks > 0:
            self.logger.info(
                f"Found {skipped_chunks} cached chunks, will process "
                f"{len(aoi_gdf) - skipped_aois} AOIs (skipping {skipped_aois})"
            )

        return chunks_to_process, skipped_chunks, skipped_aois

    def run_parallel(
        self,
        chunks_to_process: List[Tuple[int, gpd.GeoDataFrame]],
        file_stem: str,
        initial_aoi_count: int,
        use_progress_tracking: bool = True,
        **process_chunk_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Process chunks in parallel using ProcessPoolExecutor.

        Supports:
        - Shared progress counters for cross-process tracking
        - Dynamic tqdm progress bar
        - Retry logic for BrokenProcessPool
        - Memory usage monitoring
        - Per-chunk error tracking
        - Latency statistics collection

        Args:
            chunks_to_process: List of (chunk_index, chunk_gdf) tuples
            file_stem: Base filename for chunk IDs
            initial_aoi_count: Total number of AOIs (for initial progress estimate)
            use_progress_tracking: Enable shared progress counters and dynamic tqdm
            **process_chunk_kwargs: Additional kwargs to pass to process_chunk()

        Returns:
            List of latency_stats dicts from each chunk (None entries for chunks without stats)
        """
        if len(chunks_to_process) == 0:
            self.logger.info("No chunks to process (all cached or empty)")
            return []

        jobs = []
        job_to_chunk = {}  # Track which chunk each job corresponds to
        max_retries = 3
        PROCESS_POOL_RETRY_DELAY = 5  # seconds between ProcessPool retries

        # Create shared progress counters if enabled
        progress_counters = None
        if use_progress_tracking:
            manager = multiprocessing.Manager()
            # Estimate 1 request per AOI initially (may grow if gridding occurs)
            progress_counters = manager.dict(
                {
                    "total": initial_aoi_count,
                    "completed": 0,
                    "lock": manager.Lock(),
                }
            )
            # Add progress_counters to kwargs if subclass expects it
            if "progress_counters" not in process_chunk_kwargs:
                process_chunk_kwargs["progress_counters"] = progress_counters

        # Prime psutil CPU tracking (first call always returns 0.0, needs a baseline)
        psutil.cpu_percent(interval=None)

        for attempt in range(max_retries):
            try:
                with ProcessPoolExecutor(max_workers=self.processes) as executor:
                    try:
                        # Submit all chunks
                        # Track warmup start time to gradually ramp up concurrency
                        warmup_start_time = time.time()

                        # Log warmup plan once at the start
                        if API_WARMUP_INTERVAL_SECONDS > 0 and self.processes > 1:
                            warmup_duration = (self.processes - 1) * API_WARMUP_INTERVAL_SECONDS
                            self.logger.info(
                                f"API warmup: ramping parallel workers from 1 to {self.processes} "
                                f"over {warmup_duration:.0f}s"
                            )

                        # Create progress bar BEFORE submission loop so it shows during warmup
                        pbar = None
                        if use_progress_tracking and progress_counters is not None:
                            with progress_counters["lock"]:
                                initial_total = progress_counters["total"]
                            pbar = tqdm(total=initial_total, **self._TQDM_CONFIG)

                        chunks_submitted = 0
                        for i, batch in chunks_to_process:
                            chunk_id = f"{file_stem}_{str(i).zfill(4)}"

                            # API warmup: gradually ramp up concurrency to allow API autoscaling
                            # Start with 1 worker, add 1 more every warmup_interval seconds
                            if API_WARMUP_INTERVAL_SECONDS > 0 and i < self.processes:
                                while True:
                                    elapsed = time.time() - warmup_start_time
                                    # Max allowed concurrent = 1 + floor(elapsed / interval), capped at processes
                                    max_concurrent = min(
                                        1 + int(elapsed // API_WARMUP_INTERVAL_SECONDS),
                                        self.processes
                                    )
                                    active_jobs = sum(1 for j in jobs if not j.done())

                                    if active_jobs < max_concurrent:
                                        break
                                    else:
                                        # At capacity for current warmup stage - update progress while waiting
                                        if pbar is not None and progress_counters is not None:
                                            self._update_pbar_during_warmup(
                                                pbar, progress_counters, chunks_submitted, len(chunks_to_process)
                                            )
                                        time.sleep(1.0)

                            self.logger.debug(
                                f"Submitting chunk {chunk_id} with {len(batch)} AOIs "
                                f"(indices {batch.index.min()}-{batch.index.max()})"
                            )
                            job = executor.submit(
                                self.process_chunk,
                                chunk_id,
                                batch,
                                **process_chunk_kwargs,
                            )
                            jobs.append(job)
                            job_to_chunk[job] = (
                                chunk_id,
                                i,
                                batch.index.min(),
                                batch.index.max(),
                            )
                            chunks_submitted += 1

                        # Process jobs with progress tracking
                        if use_progress_tracking and progress_counters is not None:
                            all_latency_stats = self._monitor_progress_with_tqdm(
                                jobs,
                                job_to_chunk,
                                progress_counters,
                                len(chunks_to_process),
                                executor,
                                pbar=pbar,
                            )
                        else:
                            if pbar is not None:
                                pbar.close()
                            all_latency_stats = self._monitor_progress_simple(jobs, job_to_chunk)

                        return all_latency_stats  # Success - exit retry loop

                    except KeyboardInterrupt:
                        self.logger.warning(
                            "Interrupted by user (Ctrl+C) - shutting down processes..."
                        )
                        # Close progress bar if it exists
                        if pbar is not None:
                            pbar.close()
                        # Cancel all pending jobs
                        for job in jobs:
                            job.cancel()
                        executor.shutdown(wait=False)
                        raise

                    finally:
                        executor.shutdown(wait=True)

            except BrokenProcessPool as e:
                self._handle_broken_process_pool(
                    e, attempt, max_retries, PROCESS_POOL_RETRY_DELAY
                )
                if attempt < max_retries - 1:
                    jobs = []  # Reset jobs list for retry
                    job_to_chunk = {}  # Reset job tracking for retry
                else:
                    raise

        # Should not reach here (either returns on success or raises on failure)
        return []

    def _update_pbar_during_warmup(
        self,
        pbar: tqdm,
        progress_counters: Dict[str, Any],
        chunks_submitted: int,
        total_chunks: int,
    ) -> None:
        """
        Update the progress bar during warmup phase.

        Called while waiting to submit more chunks during API warmup.
        Shows current progress even though not all chunks are submitted yet.
        """
        lock_acquired = progress_counters["lock"].acquire(timeout=0.01)
        if lock_acquired:
            try:
                requests_completed = progress_counters["completed"]
                requests_total = progress_counters["total"]
            finally:
                progress_counters["lock"].release()

            if pbar.total != requests_total:
                pbar.total = requests_total
            pbar.n = requests_completed

            pbar.set_description(
                self._format_progress_description(chunks_submitted, total_chunks)
            )
            pbar.refresh()

    def _monitor_progress_with_tqdm(
        self,
        jobs: List[concurrent.futures.Future],
        job_to_chunk: Dict[concurrent.futures.Future, Tuple[str, int, Any, Any]],
        progress_counters: Dict[str, Any],
        num_jobs: int,
        executor: ProcessPoolExecutor,
        pbar: Optional[tqdm] = None,
    ) -> List[Dict[str, Any]]:
        """
        Monitor job progress with dynamic tqdm progress bar.

        Updates progress based on shared counters that can grow during gridding.

        Args:
            jobs: List of submitted futures
            job_to_chunk: Mapping from future to chunk info
            progress_counters: Shared counters dict with 'total', 'completed', 'lock'
            num_jobs: Total number of chunks (can be updated dynamically)
            executor: The ProcessPoolExecutor
            pbar: Optional existing tqdm progress bar (created if not provided)

        Returns:
            List of latency_stats dicts from each completed chunk
        """
        completed_jobs = 0
        last_progress_check = time.time()
        PROGRESS_CHECK_INTERVAL = 0.5  # Check shared counters every 0.5 seconds
        all_latency_stats = []  # Collect latency stats from each chunk
        latest_latency_stats = None  # Track latest for progress bar display

        # Get initial total
        with progress_counters["lock"]:
            initial_total = progress_counters["total"]

        # Use existing pbar or create new one
        pbar_created = pbar is None
        if pbar_created:
            pbar = tqdm(total=initial_total, **self._TQDM_CONFIG)

        try:
            while completed_jobs < num_jobs:
                # Check if any jobs have completed (non-blocking)
                done, _ = wait(
                    jobs,
                    timeout=0.1,
                    return_when=FIRST_COMPLETED,
                )

                for j in done:
                    if j in jobs:  # Make sure we haven't already processed this
                        try:
                            result = j.result()  # Block until result fully transferred
                            completed_jobs += 1

                            # Collect latency stats from chunk result
                            if isinstance(result, dict) and "latency_stats" in result:
                                latency_stats = result.get("latency_stats")
                                if latency_stats is not None:
                                    all_latency_stats.append(latency_stats)
                                    latest_latency_stats = latency_stats

                            # Update progress bar immediately
                            lock_acquired = progress_counters["lock"].acquire(
                                timeout=0.01
                            )
                            if lock_acquired:
                                try:
                                    requests_completed = progress_counters["completed"]
                                    requests_total = progress_counters["total"]
                                finally:
                                    progress_counters["lock"].release()

                                if pbar.total != requests_total:
                                    pbar.total = requests_total
                                pbar.n = requests_completed

                            # Build latency string for progress bar
                            if latest_latency_stats:
                                lat_str = f"P50={latest_latency_stats['p50']:.0f}ms"
                            else:
                                lat_str = "Lat: ---"

                            pbar.set_description(
                                self._format_progress_description(
                                    completed_jobs, num_jobs, lat_str=lat_str
                                )
                            )
                            pbar.refresh()

                        except Exception as e:
                            chunk_info = job_to_chunk.get(j, ("unknown", -1, -1, -1))
                            chunk_id, chunk_idx, min_aoi, max_aoi = chunk_info
                            completed_jobs += 1
                            self.logger.error(
                                f"FAILURE TO COMPLETE JOB - Chunk: {chunk_id} "
                                f"(index {chunk_idx}), AOI range: {min_aoi}-{max_aoi}, "
                                f"Error: {e}"
                            )
                            self.logger.error(f"Traceback: {traceback.format_exc()}")
                            executor.shutdown(wait=False)
                            raise
                        finally:
                            jobs.remove(j)  # Remove from pending jobs list

                # Periodically check shared counters and update progress bar
                current_time = time.time()
                if current_time - last_progress_check >= PROGRESS_CHECK_INTERVAL:
                    # Try to acquire lock with timeout
                    lock_acquired = progress_counters["lock"].acquire(timeout=0.1)

                    if lock_acquired:
                        try:
                            requests_completed = progress_counters["completed"]
                            requests_total = progress_counters["total"]
                        finally:
                            progress_counters["lock"].release()

                        # Update total if it changed (due to gridding)
                        if pbar.total != requests_total:
                            pbar.total = requests_total

                        # Build latency string for progress bar
                        if latest_latency_stats:
                            lat_str = f"P50={latest_latency_stats['p50']:.0f}ms"
                        else:
                            lat_str = "Lat: ---"

                        # Update position and description
                        pbar.n = requests_completed
                        pbar.set_description(
                            self._format_progress_description(
                                completed_jobs, num_jobs, lat_str=lat_str
                            )
                        )

                    # Always refresh, even if we couldn't get the lock
                    pbar.refresh()
                    last_progress_check = current_time

            # Final update to show 100% completion
            with progress_counters["lock"]:
                requests_completed = progress_counters["completed"]
                requests_total = progress_counters["total"]

            # Build final latency string
            if latest_latency_stats:
                lat_str = f"P50={latest_latency_stats['p50']:.0f}ms"
            else:
                lat_str = "Lat: ---"

            pbar.n = requests_completed
            pbar.total = requests_total
            pbar.set_description(
                self._format_progress_description(
                    num_jobs, num_jobs, lat_str=lat_str
                )
            )
            pbar.refresh()

        finally:
            # Close pbar only if we created it
            if pbar_created:
                pbar.close()

        return all_latency_stats

    def _monitor_progress_simple(
        self,
        jobs: List[concurrent.futures.Future],
        job_to_chunk: Dict[concurrent.futures.Future, Tuple[str, int, Any, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Simple progress monitoring with tqdm over completed jobs.

        Used when progress tracking is disabled or not supported.

        Returns:
            List of latency_stats dicts from each completed chunk
        """
        self.logger.info(f"Processing {len(jobs)} chunks...")
        all_latency_stats = []
        for job, _ in tqdm(list(zip(jobs, range(len(jobs)))), desc="Processing chunks"):
            try:
                result = job.result()  # Wait for completion and raise any exceptions
                # Collect latency stats from chunk result
                if isinstance(result, dict) and "latency_stats" in result:
                    latency_stats = result.get("latency_stats")
                    if latency_stats is not None:
                        all_latency_stats.append(latency_stats)
            except Exception as e:
                chunk_info = job_to_chunk.get(job, ("unknown", -1, -1, -1))
                chunk_id = chunk_info[0]
                self.logger.error(f"Chunk {chunk_id} failed: {e}")
                raise
        return all_latency_stats

    def _handle_broken_process_pool(
        self, error: Exception, attempt: int, max_retries: int, retry_delay: int
    ):
        """
        Handle BrokenProcessPool errors with diagnostic logging.

        Args:
            error: The BrokenProcessPool exception
            attempt: Current attempt number (0-indexed)
            max_retries: Maximum number of retry attempts
            retry_delay: Seconds to wait before retrying
        """
        import resource

        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Get resource limits
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)

        # Count open file descriptors (Linux/Mac)
        try:
            import os

            pid = os.getpid()
            if os.path.exists(f"/proc/{pid}/fd"):
                fd_count = len(os.listdir(f"/proc/{pid}/fd"))
            else:
                fd_count = "unknown"
        except Exception:
            fd_count = "unknown"

        self.logger.error(f"BrokenProcessPool diagnostic info:")

        # Log cgroup info if in container
        if is_running_in_container():
            cgroup_usage = get_cgroup_memory_usage_gb()
            cgroup_limit = get_cgroup_memory_limit_gb()
            if cgroup_usage is not None and cgroup_limit is not None:
                cgroup_percent = (cgroup_usage / cgroup_limit) * 100
                self.logger.error(
                    f"  Container Memory: {cgroup_usage:.1f}GB used of {cgroup_limit:.1f}GB ({cgroup_percent:.1f}%)"
                )
        self.logger.error(
            f"  Host Memory: {mem.used/1024**3:.1f}GB used of {mem.total/1024**3:.1f}GB ({mem.percent}%)"
        )
        self.logger.error(
            f"  Swap: {swap.used/1024**3:.1f}GB used of {swap.total/1024**3:.1f}GB ({swap.percent}%)"
        )
        self.logger.error(f"  File descriptors: {fd_count} open (limit: {soft_limit})")
        self.logger.error(f"  Active processes: {self.processes}")
        cpu_pct = psutil.cpu_percent(interval=0.1)  # Blocking OK in error path
        self.logger.error(
            f"  CPU: {cpu_pct:.1f}% across {psutil.cpu_count()} host cores"
        )
        if is_running_in_container():
            cgroup_cpus = get_cgroup_cpu_limit()
            if cgroup_cpus is not None:
                self.logger.error(f"  Container CPU limit: {cgroup_cpus:.1f} CPUs")

        if attempt < max_retries - 1:
            self.logger.warning(
                f"Process pool broken, attempt {attempt + 1}/{max_retries}, "
                f"retrying after {retry_delay}s delay..."
            )
            time.sleep(retry_delay)
        else:
            self.logger.error(
                f"Process pool broken after {max_retries} attempts, giving up"
            )

    @staticmethod
    def configure_worker_logging(log_level: str):
        """
        Configure logging for worker processes.

        Should be called at the start of process_chunk() in subclasses.

        Args:
            log_level: Logging level to use
        """
        if multiprocessing.current_process().name != "MainProcess":
            log.configure_logger(log_level)
