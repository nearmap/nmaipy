"""
Shared infrastructure for Nearmap AI API clients.

This module provides common functionality used across different API clients:
- Retry logic with exponential backoff
- API key security (filtering from logs)
- Error handling classes
- Session management with connection pooling
- Request caching infrastructure
"""
import contextlib
import gzip
import hashlib
import json
import logging
import os
import re
import ssl
import threading
from http import HTTPStatus
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import urllib3
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nmaipy import log
from nmaipy.constants import (
    BACKOFF_FACTOR,
    DUMMY_STATUS_CODE,
    MAX_RETRIES,
    READ_TIMEOUT_SECONDS,
    TIMEOUT_SECONDS,
)

# Load environment variables from .env file
load_dotenv()

# Get API key, with fallback to empty string
API_KEY = os.getenv("API_KEY", "")

logger = log.get_logger()

# Latency histogram bucket boundaries in milliseconds
# - Fine resolution <200ms (target response time)
# - SLA boundary at 2s
# - Slow request range 10-60s
# - 60s+ for timeouts
LATENCY_BUCKETS = [0, 50, 100, 150, 200, 300, 500, 1000, 2000, 5000, 10000, 30000, 60000, float("inf")]


class APIKeyFilter(logging.Filter):
    """Logging filter to remove API keys from log messages for security."""

    def filter(self, record):
        # Clean API key from the message if present
        if hasattr(record, 'getMessage'):
            msg = record.getMessage()
            # Pattern to match API key in URLs
            msg = re.sub(r'apikey=[^&\s]+', 'apikey=REMOVED', msg)
            # Pattern to match API key in various formats (be aggressive about cleaning)
            msg = re.sub(r'(["\']?api[_-]?key["\']?\s*[:=]\s*["\']?)[^"\'\s,}]+', r'\1REMOVED', msg, flags=re.IGNORECASE)
            record.msg = msg
            record.args = ()  # Clear args to prevent formatting issues
        return True


# Apply the filter to urllib3 logger to prevent API key leakage
urllib3_logger = logging.getLogger('urllib3.connectionpool')
urllib3_logger.addFilter(APIKeyFilter())

# Also apply to requests logger for completeness
requests_logger = logging.getLogger('requests')
requests_logger.addFilter(APIKeyFilter())

# Also apply to nmaipy logger for defense in depth
nmaipy_logger = logging.getLogger('nmaipy')
nmaipy_logger.addFilter(APIKeyFilter())


def sanitize_error_message(message: str) -> str:
    """
    Sanitize error messages for aggregation by truncating URL query parameters.

    This allows grouping of similar errors that differ only in URL parameters
    (e.g., different coordinates, timestamps) so they can be counted together
    in error summaries.

    Args:
        message: Error message that may contain URLs

    Returns:
        Message with URL query parameters replaced by "..."

    Example:
        >>> sanitize_error_message("Error at https://api.example.com/endpoint?lat=1.23&lon=4.56")
        'Error at https://api.example.com/endpoint?...'
    """
    if not message or not isinstance(message, str):
        return message
    # Match URLs and truncate after the "?"
    return re.sub(r'(https?://[^\s?]+)\?[^\s]*', r'\1?...', message)


def format_error_summary_table(status_counts, message_counts, max_message_len=160):
    """
    Format error summary as an ASCII table for logging.

    Args:
        status_counts: pandas Series of status code counts (or None)
        message_counts: pandas Series of message counts (or None)
        max_message_len: Maximum length for message text before truncation

    Returns:
        str: Formatted ASCII table string
    """
    lines = []

    # Status codes table
    if status_counts is not None and len(status_counts) > 0:
        lines.append("  Status Codes:")
        lines.append(f"  {'Code':<10} {'Count':>8}")
        lines.append(f"  {'-' * 10} {'-' * 8}")
        for code, count in status_counts.items():
            lines.append(f"  {code:<10} {count:>8}")

    # Messages table
    if message_counts is not None and len(message_counts) > 0:
        if lines:
            lines.append("")
        lines.append("  Error Messages:")
        lines.append(f"  {'Count':>6}  {'Message'}")
        lines.append(f"  {'-' * 6}  {'-' * max_message_len}")
        for msg, count in message_counts.items():
            truncated_msg = msg if len(msg) <= max_message_len else msg[: max_message_len - 3] + "..."
            lines.append(f"  {count:>6}  {truncated_msg}")

    return "\n" + "\n".join(lines) if lines else ""


class RetryRequest(Retry):
    """
    Inherited retry request with controlled backoff timing.

    BACKOFF_MAX set to 20s to allow more time for transient failures to resolve.
    BACKOFF_MIN set to 2s to provide more breathing room before retrying.
    """

    BACKOFF_MIN = 2.0  # Minimum backoff time in seconds
    BACKOFF_MAX = 10    # Maximum backoff time in seconds

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add all connection-related errors to retry on
        self.RETRY_AFTER_STATUS_CODES = frozenset({
            HTTPStatus.TOO_MANY_REQUESTS,      # 429
            HTTPStatus.INTERNAL_SERVER_ERROR,   # 500
            HTTPStatus.BAD_GATEWAY,            # 502
            HTTPStatus.SERVICE_UNAVAILABLE,     # 503
        })

        # Add connection errors that should trigger retries
        self.RETRY_ON_EXCEPTIONS = frozenset({
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            RemoteDisconnected,  # From http.client
            requests.exceptions.ProxyError,
            requests.exceptions.SSLError,
            urllib3.exceptions.SSLError,  # Added to catch SSL errors from urllib3
            requests.exceptions.Timeout,
            urllib3.exceptions.ProtocolError,
            EOFError,  # Sometimes occurs with RemoteDisconnected
            ConnectionResetError,  # Python built-in exception
            ssl.SSLEOFError,  # Explicit SSL EOF error
        })

    def increment(self, method=None, url=None, response=None, error=None, _pool=None, _stacktrace=None):
        """Override to log on first retry attempt"""
        result = super().increment(method, url, response, error, _pool, _stacktrace)

        # Log on first retry only (history is a tuple, check length)
        if result and len(result.history) == 1:
            if response is not None:
                reason = f"HTTP {response.status} {response.reason}"
            elif error is not None:
                reason = f"{type(error).__name__}: {str(error)[:100]}"
            else:
                reason = "unknown reason"

            # Clean URL of API key for logging
            clean_url = url
            if url and 'apikey=' in url:
                clean_url = url.split('apikey=')[0] + 'apikey=***'

            logger.info(f"{reason} causing retry of request {clean_url}")

        return result

    def new_timeout(self, *args, **kwargs):
        """Override to enforce backoff time between BACKOFF_MIN and BACKOFF_MAX"""
        timeout = super().new_timeout(*args, **kwargs)
        return min(max(timeout, self.BACKOFF_MIN), self.BACKOFF_MAX)  # Clamp between min and max seconds

    @classmethod
    def from_int(cls, retries, **kwargs):
        """Helper to create retry config with better defaults"""
        kwargs.setdefault('backoff_factor', BACKOFF_FACTOR)
        kwargs.setdefault('status_forcelist', [429, 500, 502, 503, 504])
        kwargs.setdefault('respect_retry_after_header', True)
        return super().from_int(retries, **kwargs)


class APIError(Exception):
    """
    Base error class for all Nearmap AI API errors.
    Captures response status, body, and error messages.
    """

    def __init__(self, response, request_string, text="Query Not Attempted", message="Error with Query"):
        if response is None:
            self.status_code = DUMMY_STATUS_CODE
            self.text = text
            self.message = message
        else:
            try:
                self.status_code = response.status_code
                self.text = response.text
            except AttributeError:
                self.status_code = response["status_code"]
                self.text = response["text"]
            try:
                err_body = response.json()
                self.message = err_body["message"] if "message" in err_body else err_body.get("error", "")
            except json.JSONDecodeError:
                self.message = "JSONDecodeError"
            except ValueError:
                self.message = "ValueError"
            except requests.exceptions.ChunkedEncodingError:
                self.message = "ChunkedEncodingError"
            except AttributeError:
                self.message = ""
        self.request_string = request_string

    def __str__(self):
        """Return a concise error message without the full response body"""
        return f"{self.__class__.__name__}: {self.status_code} - {self.message}"


class AIFeatureAPIError(APIError):
    """
    Error responses for logging from AI Feature API.
    Maintains backward compatibility with existing code.
    """
    pass


class RoofAgeAPIError(APIError):
    """Error responses specific to the Roof Age API."""
    pass


class APIRequestSizeError(APIError):
    """
    Generic error indicating a request is too large for the API to handle.

    This can occur when:
    - AOI geometry is too complex or has too many vertices
    - AOI area exceeds API limits
    - Server times out due to request complexity (504 Gateway Timeout)

    When this error occurs, the client should consider:
    - Splitting the AOI into smaller chunks (gridding)
    - Simplifying the geometry
    - Reducing the query scope
    """

    status_codes = (HTTPStatus.GATEWAY_TIMEOUT, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
    error_codes = ("AOI_EXCEEDS_MAX_SIZE",)  # API error codes that indicate size issues

    def __str__(self):
        """Return a concise error message"""
        return f"APIRequestSizeError: {self.status_code} - Request too large, consider gridding"


class APIGridError(Exception):
    """
    Generic error indicating that a gridded request failed.

    This occurs when an AOI was split into a grid for parallel processing,
    but one or more grid cells failed to process successfully.
    """

    def __init__(self, status_code_error_mode, message=""):
        self.status_code = status_code_error_mode
        self.text = "Gridding and re-requesting failed on one or more grid cell queries."
        self.request_string = ""
        self.message = message

    def __str__(self):
        """Return a concise error message"""
        return f"APIGridError: {self.status_code} - {self.message}"


def clean_api_key_from_string(text: str) -> str:
    """
    Remove API keys from a string for safe logging.

    Args:
        text: String that may contain API keys

    Returns:
        String with API keys replaced with 'REMOVED'
    """
    if not text:
        return text
    # Pattern to match API key in URLs
    text = re.sub(r'apikey=[^&\s]+', 'apikey=REMOVED', text)
    # Pattern to match API key in various formats
    text = re.sub(r'(["\']?api[_-]?key["\']?\s*[:=]\s*["\']?)[^"\'\s,}]+', r'\1REMOVED', text, flags=re.IGNORECASE)
    return text


class BaseApiClient:
    """
    Base class for Nearmap AI API clients.

    Provides common functionality:
    - Session management with connection pooling
    - Request retry logic
    - Caching infrastructure
    - API key handling
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        overwrite_cache: Optional[bool] = False,
        compress_cache: Optional[bool] = False,
        threads: Optional[int] = 10,
        maxretry: int = MAX_RETRIES,
    ):
        """
        Initialize base API client.

        Args:
            api_key: Nearmap API key. If not provided, reads from API_KEY environment variable
            cache_dir: Directory to use as a payload cache
            overwrite_cache: Whether to overwrite cached values
            compress_cache: Whether to use gzip compression (.json.gz) or raw json (.json)
            threads: Number of threads for concurrent execution
            maxretry: Number of retries for failed requests
        """
        # Initialize thread-safety attributes
        self._sessions = []
        self._thread_local = threading.local()
        self._lock = threading.Lock()

        # API key handling
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("API_KEY", None)
        if self.api_key is None:
            raise ValueError(
                "No API KEY provided. Provide a key when initializing the client or set an API_KEY "
                "environment variable"
            )

        # Cache configuration
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        elif overwrite_cache:
            raise ValueError(f"No cache dir specified, but overwrite cache set to True.")

        self.overwrite_cache = overwrite_cache
        self.compress_cache = compress_cache
        self.threads = threads
        self.maxretry = maxretry

        # Latency and request tracking
        # Note: These are safe because each chunk runs in a separate process (ProcessPoolExecutor),
        # so each API client instance is isolated - no cross-thread access occurs.
        self._latencies = []
        self._retry_count = 0
        self._timeout_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def __del__(self):
        """Cleanup when instance is destroyed"""
        self.cleanup()

    def cleanup(self):
        """Clean up all sessions"""
        if hasattr(self, "_lock") and hasattr(self, "_sessions"):
            with self._lock:
                for session in self._sessions:
                    try:
                        session.close()
                    except Exception:
                        pass
                self._sessions.clear()
        else:  # Fallback if attributes don't exist
            if hasattr(self, "_sessions"):
                for session in self._sessions:
                    try:
                        session.close()
                    except Exception:
                        pass
                self._sessions.clear()

    @contextlib.contextmanager
    def _session_scope(self, status_forcelist=None):
        """
        Context manager for session lifecycle.
        Creates fresh session each time to prevent resource accumulation.

        Args:
            status_forcelist: List of HTTP status codes to retry on. Defaults to [429, 500, 502, 503, 504]

        Yields:
            requests.Session: Configured session with retry logic
        """
        if status_forcelist is None:
            status_forcelist = [
                HTTPStatus.TOO_MANY_REQUESTS,      # 429
                HTTPStatus.INTERNAL_SERVER_ERROR,   # 500
                HTTPStatus.BAD_GATEWAY,            # 502
                HTTPStatus.SERVICE_UNAVAILABLE,     # 503
                HTTPStatus.GATEWAY_TIMEOUT,        # 504
            ]

        # Always create a fresh session to prevent resource accumulation
        session = requests.Session()

        retries = RetryRequest(
            total=self.maxretry,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=status_forcelist,
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
            connect=self.maxretry,
            read=self.maxretry,
            redirect=self.maxretry,
        )

        # Dynamic pool sizing: match pool size to thread count
        # Min 10 for basic concurrency, max 50 to prevent file handle exhaustion
        pool_size = min(max(self.threads, 10), 50)
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_maxsize=pool_size,
            pool_connections=pool_size,
            pool_block=True
        )
        session.mount("https://", adapter)

        # Store timeout for use in requests (connect timeout, read timeout)
        session._timeout = (TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)

        try:
            yield session
        finally:
            # Always close the session to prevent resource leaks
            try:
                session.close()
            except Exception:
                pass

    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get the cache file path for a given key.

        Args:
            cache_key: Unique identifier for the cached item

        Returns:
            Path to the cache file
        """
        if self.cache_dir is None:
            raise ValueError("Cache directory not configured")

        # Hash the key to create a safe filename
        key_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        extension = ".json.gz" if self.compress_cache else ".json"
        return self.cache_dir / f"{key_hash}{extension}"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """
        Load data from cache if it exists and cache is enabled.

        Args:
            cache_key: Unique identifier for the cached item

        Returns:
            Cached data dict, or None if not found or cache disabled
        """
        if self.cache_dir is None or self.overwrite_cache:
            return None

        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            if self.compress_cache:
                with gzip.open(cache_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        """
        Save data to cache.

        Args:
            cache_key: Unique identifier for the cached item
            data: Data to cache (must be JSON-serializable)
        """
        if self.cache_dir is None:
            return

        cache_path = self._get_cache_path(cache_key)
        try:
            if self.compress_cache:
                with gzip.open(cache_path, 'wt', encoding='utf-8') as f:
                    json.dump(data, f)
            else:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def _sanitize_path_component(self, text: str) -> str:
        """
        Sanitize a string for use as a path component.

        Removes/replaces characters that are problematic in file paths.

        Args:
            text: String to sanitize

        Returns:
            Safe string for use in file paths
        """
        if not text:
            return "_empty_"
        # Replace problematic characters with underscore
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(text))
        # Replace multiple underscores with single
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores and whitespace
        sanitized = sanitized.strip('_ \t')
        # Truncate to reasonable length
        return sanitized[:50] if sanitized else "_empty_"

    def _get_address_cache_path(
        self,
        country: str,
        state: str,
        city: str,
        zipcode: str,
        cache_key: str
    ) -> Path:
        """
        Get cache path for address-based queries.

        Uses nested directory structure: cache_dir/country/state/city/zip/<hash>.json

        This structure is shared between Feature API and Roof Age API for consistency.

        Args:
            country: Country code (e.g., 'US', 'AU')
            state: State/province code
            city: City name
            zipcode: Postal/zip code
            cache_key: Full cache key to hash for filename

        Returns:
            Path to the cache file
        """
        if self.cache_dir is None:
            raise ValueError("Cache directory not configured")

        extension = ".json.gz" if self.compress_cache else ".json"
        key_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]

        cache_subdir = (
            self.cache_dir
            / self._sanitize_path_component(country)
            / self._sanitize_path_component(state)
            / self._sanitize_path_component(city)
            / self._sanitize_path_component(zipcode)
        )
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / f"{key_hash}{extension}"

    def _clean_api_key(self, text: str) -> str:
        """
        Remove API keys from text for safe logging.

        Args:
            text: String that may contain API keys

        Returns:
            String with API keys replaced with 'REMOVED'
        """
        return clean_api_key_from_string(text)

    def get_latency_stats(self) -> Optional[Dict[str, Any]]:
        """
        Calculate latency statistics for this API client instance.

        Returns a dict containing:
        - mean: Arithmetic mean
        - p50, p90, p95, p99: Percentiles
        - min, max: Extremes
        - count: Number of successful requests (latency samples)
        - histogram: Bucket counts for aggregation across chunks
        - retry_count: Number of retries that occurred
        - timeout_count: Number of requests that timed out
        - cache_hits: Number of cache hits
        - cache_misses: Number of cache misses (API calls made)

        Returns None if no latencies have been recorded.
        """
        if not self._latencies:
            return None

        arr = np.array(self._latencies)
        n = len(arr)

        # Histogram buckets for cross-chunk aggregation
        hist, _ = np.histogram(arr, bins=LATENCY_BUCKETS)

        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": n,
            "histogram": hist.tolist(),
            "retry_count": self._retry_count,
            "timeout_count": self._timeout_count,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }


def collect_latency_stats_from_apis(
    api_clients: List[Optional["BaseApiClient"]],
    chunk_id: str,
    chunk_start_time: str,
    chunk_end_time: str,
    total_duration_ms: float,
) -> Optional[Dict[str, Any]]:
    """
    Collect and combine latency statistics from multiple API clients.

    This helper function aggregates latency data from one or more API client instances
    (e.g., FeatureApi and RoofAgeApi) into a single stats dict suitable for saving.

    Args:
        api_clients: List of API client instances (None entries are skipped)
        chunk_id: Identifier for this chunk
        chunk_start_time: ISO format timestamp when chunk processing started
        chunk_end_time: ISO format timestamp when chunk processing ended
        total_duration_ms: Total wall-clock time for chunk processing in milliseconds

    Returns:
        Dict with combined latency stats, or None if no latencies were recorded
    """
    combined_latencies = []
    combined_retry_count = 0
    combined_timeout_count = 0
    combined_cache_hits = 0
    combined_cache_misses = 0

    for api in api_clients:
        if api is not None:
            combined_latencies.extend(api._latencies)
            combined_retry_count += api._retry_count
            combined_timeout_count += api._timeout_count
            combined_cache_hits += api._cache_hits
            combined_cache_misses += api._cache_misses

    if not combined_latencies:
        return None

    arr = np.array(combined_latencies)
    n = len(arr)
    hist, _ = np.histogram(arr, bins=LATENCY_BUCKETS)

    return {
        "chunk_id": chunk_id,
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": n,
        "histogram": hist.tolist(),
        "retry_count": combined_retry_count,
        "timeout_count": combined_timeout_count,
        "cache_hits": combined_cache_hits,
        "cache_misses": combined_cache_misses,
        "start_time": chunk_start_time,
        "end_time": chunk_end_time,
        "total_duration_ms": total_duration_ms,
    }


class GriddedApiClient(BaseApiClient):
    """
    Base class for API clients that support automatic gridding of large AOIs.

    Extends BaseApiClient with the ability to automatically split large geometries
    into grid cells for parallel processing. This is useful for APIs that have
    size limits on individual requests.

    Features:
    - Automatic gridding when AOI exceeds size threshold
    - Parallel processing of grid cells with thread pool
    - Semaphore-based concurrency control to prevent resource exhaustion
    - Result combination with deduplication
    - Proactive and reactive gridding strategies

    Subclasses should implement:
    - Query methods that may need gridding support
    - _combine_grid_results() if custom result combination is needed
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        overwrite_cache: Optional[bool] = False,
        compress_cache: Optional[bool] = False,
        threads: Optional[int] = 10,
        maxretry: int = MAX_RETRIES,
        grid_cell_size: float = 0.002,
        max_aoi_area_sqm: float = 1_000_000,
        max_concurrent_gridding: Optional[int] = None,
    ):
        """
        Initialize GriddedApiClient.

        Args:
            api_key: Nearmap API key
            cache_dir: Directory for caching
            overwrite_cache: Whether to overwrite cache
            compress_cache: Whether to compress cache files
            threads: Number of concurrent threads
            maxretry: Number of retries for failed requests
            grid_cell_size: Size of grid cells in degrees (default ~200m)
            max_aoi_area_sqm: Max AOI area before gridding (default 1 sq km)
            max_concurrent_gridding: Max concurrent gridding operations (default threads/5)
        """
        super().__init__(
            api_key=api_key,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            compress_cache=compress_cache,
            threads=threads,
            maxretry=maxretry,
        )

        self.grid_cell_size = grid_cell_size
        self.max_aoi_area_sqm = max_aoi_area_sqm

        # Semaphore to limit concurrent gridding operations
        # This prevents too many file handles being opened when many large AOIs grid simultaneously
        if max_concurrent_gridding is None:
            max_concurrent_gridding = max(1, self.threads // 5)
        self._gridding_semaphore = threading.Semaphore(max_concurrent_gridding)
        logger.debug(f"Initialized gridding semaphore with limit of {max_concurrent_gridding} concurrent AOIs")

    def should_grid_aoi(self, geometry, region: str = "us") -> bool:
        """
        Determine if an AOI should be pre-emptively gridded based on area.

        Args:
            geometry: Polygon or MultiPolygon in API_CRS (EPSG:4326)
            region: Country/region code for CRS selection (default: 'us')

        Returns:
            True if geometry area exceeds threshold and should be gridded
        """
        from nmaipy.constants import AREA_CRS
        import geopandas as gpd

        # Calculate area in appropriate projected CRS
        gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
        area_sqm = gdf.to_crs(AREA_CRS.get(region, AREA_CRS["us"])).area.iloc[0]

        return area_sqm > self.max_aoi_area_sqm


def generate_curl_command(url: str, body: dict = None, method: str = "POST", timeout: int = 90) -> str:
    """
    Generate a curl command for debugging HTTP requests.

    Provides all information needed to reproduce an API request from the command line.
    API keys are automatically sanitized for safety.

    Args:
        url: The request URL (with API key in query params if applicable)
        body: Optional JSON body for POST requests
        method: HTTP method (POST or GET)
        timeout: Request timeout in seconds

    Returns:
        A formatted curl command string with API key sanitized

    Example:
        >>> url = "https://api.nearmap.com/ai/features/v4/features.json?apikey=SECRET"
        >>> body = {"aoi": {"type": "Polygon", "coordinates": [...]}}
        >>> cmd = generate_curl_command(url, body)
        >>> print(cmd)  # API key will be replaced with 'APIKEYREMOVED'
    """
    # Clean the API key from the URL
    clean_url = clean_api_key_from_string(url)

    if method == "POST" and body:
        # Format the JSON body nicely
        json_body = json.dumps(body, indent=2)

        # Build the curl command
        curl_cmd = f"""curl -X POST '{clean_url}' \\
  -H 'Content-Type: application/json' \\
  -H 'Accept: application/json' \\
  --max-time {timeout} \\
  --data '{json_body}'

# To use this command:
# 1. Replace 'APIKEYREMOVED' or 'REMOVED' with your actual API key
# 2. Save the geometry to a file if it's too large for command line
# 3. Use --data @filename.json for large payloads
# 4. Add -v for verbose output to debug connection issues"""
    else:
        # GET request
        curl_cmd = f"""curl -X GET '{clean_url}' \\
  -H 'Accept: application/json' \\
  --max-time {timeout}

# To use this command:
# 1. Replace 'APIKEYREMOVED' or 'REMOVED' with your actual API key
# 2. Add -v for verbose output to debug connection issues"""

    return curl_cmd


def format_error_message(request_string: str, response: requests.Response) -> str:
    """
    Format a detailed error message for failed API requests.

    Args:
        request_string: The original request URL/string
        response: HTTP response object from failed request

    Returns:
        Formatted error message with status, response text, and sanitized request

    Example:
        >>> msg = format_error_message(url, response)
        >>> logger.error(msg)
    """
    clean_request = clean_api_key_from_string(request_string)
    return f"\n{clean_request=}\n\n{response.status_code=}\n\n{response.text}\n\n"


# =============================================================================
# Latency Statistics Helper Functions
# =============================================================================


def percentile_from_histogram(hist: np.ndarray, buckets: List[float], percentile: float) -> float:
    """
    Compute percentile from histogram via linear interpolation.

    Args:
        hist: Array of bucket counts
        buckets: Bucket boundaries (len = len(hist) + 1)
        percentile: Percentile to compute (0-100)

    Returns:
        Estimated percentile value in milliseconds
    """
    cumsum = np.cumsum(hist)
    total = cumsum[-1]
    if total == 0:
        return 0.0

    target = total * percentile / 100

    # Find bucket containing the percentile
    idx = np.searchsorted(cumsum, target)
    if idx >= len(hist):
        idx = len(hist) - 1

    # Linear interpolation within bucket
    lower_cum = cumsum[idx - 1] if idx > 0 else 0
    bucket_count = hist[idx]
    if bucket_count > 0:
        fraction = (target - lower_cum) / bucket_count
    else:
        fraction = 0

    # Handle infinity in last bucket
    lower_bound = buckets[idx]
    upper_bound = buckets[idx + 1]
    if upper_bound == float("inf"):
        # For the infinity bucket, just return the lower bound
        return float(lower_bound)

    return float(lower_bound + fraction * (upper_bound - lower_bound))


def compute_global_latency_stats(
    chunk_stats: List[Dict], n_bootstrap: int = 1000, seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Compute global latency statistics from per-chunk stats with bootstrap confidence intervals.

    Args:
        chunk_stats: List of dicts from get_latency_stats() for each chunk
        n_bootstrap: Number of bootstrap samples for confidence intervals
        seed: Random seed for reproducibility (default: 42, None for non-deterministic)

    Returns:
        Dict with global mean, percentiles, and 95% confidence intervals
    """
    if not chunk_stats:
        return {}

    merged_hist = np.sum([np.array(s["histogram"]) for s in chunk_stats], axis=0)
    total_count = sum(s["count"] for s in chunk_stats)

    if total_count == 0:
        return {"count": 0}

    global_mean = sum(s["mean"] * s["count"] for s in chunk_stats) / total_count

    global_p50 = percentile_from_histogram(merged_hist, LATENCY_BUCKETS, 50)
    global_p90 = percentile_from_histogram(merged_hist, LATENCY_BUCKETS, 90)
    global_p95 = percentile_from_histogram(merged_hist, LATENCY_BUCKETS, 95)
    global_p99 = percentile_from_histogram(merged_hist, LATENCY_BUCKETS, 99)

    # Bootstrap CIs: resample chunks, merge histograms, compute percentiles
    rng = np.random.default_rng(seed)
    p50_samples, p90_samples, p95_samples, p99_samples = [], [], [], []
    chunk_stats_arr = np.array(chunk_stats, dtype=object)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(chunk_stats), size=len(chunk_stats))
        resampled = chunk_stats_arr[indices]
        hist = np.sum([np.array(s["histogram"]) for s in resampled], axis=0)
        p50_samples.append(percentile_from_histogram(hist, LATENCY_BUCKETS, 50))
        p90_samples.append(percentile_from_histogram(hist, LATENCY_BUCKETS, 90))
        p95_samples.append(percentile_from_histogram(hist, LATENCY_BUCKETS, 95))
        p99_samples.append(percentile_from_histogram(hist, LATENCY_BUCKETS, 99))

    return {
        "mean": global_mean,
        "p50": global_p50,
        "p90": global_p90,
        "p95": global_p95,
        "p99": global_p99,
        "p50_ci": (float(np.percentile(p50_samples, 2.5)), float(np.percentile(p50_samples, 97.5))),
        "p90_ci": (float(np.percentile(p90_samples, 2.5)), float(np.percentile(p90_samples, 97.5))),
        "p95_ci": (float(np.percentile(p95_samples, 2.5)), float(np.percentile(p95_samples, 97.5))),
        "p99_ci": (float(np.percentile(p99_samples, 2.5)), float(np.percentile(p99_samples, 97.5))),
        "count": total_count,
    }


def _get_latency_bucket_names() -> List[str]:
    """Get histogram bucket column names based on LATENCY_BUCKETS."""
    bucket_names = []
    for i in range(len(LATENCY_BUCKETS) - 1):
        lower = int(LATENCY_BUCKETS[i])
        upper = LATENCY_BUCKETS[i + 1]
        if upper == float("inf"):
            bucket_names.append(f"bucket_{lower}_plus")
        else:
            bucket_names.append(f"bucket_{lower}_{int(upper)}")
    return bucket_names


def _stats_to_row(stats: Dict) -> Dict:
    """Convert a latency stats dict to a flat row dict for DataFrame."""
    if stats is None:
        return None

    total_duration_ms = stats.get("total_duration_ms", 0)
    count = stats.get("count", 0)
    if total_duration_ms > 0 and count > 0:
        rps = count / (total_duration_ms / 1000)
    else:
        rps = 0.0

    row = {
        "chunk_id": stats.get("chunk_id", ""),
        "count": count,
        "mean": stats.get("mean", 0),
        "p50": stats.get("p50", 0),
        "p90": stats.get("p90", 0),
        "p95": stats.get("p95", 0),
        "p99": stats.get("p99", 0),
        "min": stats.get("min", 0),
        "max": stats.get("max", 0),
        "retry_count": stats.get("retry_count", 0),
        "timeout_count": stats.get("timeout_count", 0),
        "cache_hits": stats.get("cache_hits", 0),
        "cache_misses": stats.get("cache_misses", 0),
        "start_time": stats.get("start_time", ""),
        "end_time": stats.get("end_time", ""),
        "total_duration_ms": total_duration_ms,
        "rps": round(rps, 2),
    }

    histogram = stats.get("histogram", [0] * (len(LATENCY_BUCKETS) - 1))
    bucket_names = _get_latency_bucket_names()
    for name, count in zip(bucket_names, histogram):
        row[name] = count

    return row


def _get_latency_csv_columns() -> List[str]:
    """Get the ordered list of columns for latency CSV files."""
    bucket_names = _get_latency_bucket_names()
    return (
        ["chunk_id", "count", "mean", "p50", "p90", "p95", "p99", "min", "max"]
        + ["retry_count", "timeout_count", "cache_hits", "cache_misses"]
        + ["start_time", "end_time", "total_duration_ms", "rps"]
        + bucket_names
    )


def write_latency_csv(chunk_stats: List[Dict], csv_path) -> None:
    """
    Write per-chunk latency statistics to a CSV file using pandas.

    Args:
        chunk_stats: List of dicts with chunk_id and latency stats
        csv_path: Path to write the CSV file
    """
    rows = [_stats_to_row(s) for s in chunk_stats if s is not None]
    if not rows:
        return

    df = pd.DataFrame(rows)
    columns = _get_latency_csv_columns()
    df = df[columns]
    df.to_csv(csv_path, index=False)


def save_chunk_latency_stats(stats: Dict, chunk_path: Path, chunk_id: str) -> None:
    """
    Save latency stats for a single chunk to a sidecar parquet file.

    Args:
        stats: Dict with latency stats from a single chunk
        chunk_path: Path to the chunk directory
        chunk_id: Identifier for this chunk
    """
    if stats is None:
        return

    row = _stats_to_row(stats)
    if row is None:
        return

    df = pd.DataFrame([row])
    outfile = chunk_path / f"latency_{chunk_id}.parquet"
    df.to_parquet(outfile, index=False)


def combine_chunk_latency_stats(chunk_path: Path, file_stem: str, output_csv_path: Path) -> List[Dict]:
    """
    Combine per-chunk latency parquet files into a final CSV and return stats list.

    Args:
        chunk_path: Path to directory containing per-chunk latency files
        file_stem: Base filename stem for matching latency files (e.g., "mydata")
        output_csv_path: Path to write the combined CSV file

    Returns:
        List of latency stats dicts suitable for compute_global_latency_stats()
    """
    latency_files = list(chunk_path.glob(f"latency_{file_stem}_*.parquet"))
    if not latency_files:
        return []

    dfs = []
    for lf in latency_files:
        try:
            dfs.append(pd.read_parquet(lf))
        except Exception:
            pass  # Skip corrupted files

    if not dfs:
        return []

    combined_df = pd.concat(dfs, ignore_index=True)

    columns = _get_latency_csv_columns()
    available_columns = [c for c in columns if c in combined_df.columns]
    combined_df = combined_df[available_columns]
    combined_df.to_csv(output_csv_path, index=False)

    bucket_names = _get_latency_bucket_names()
    stats_list = []
    for _, row in combined_df.iterrows():
        stats = {
            "chunk_id": row["chunk_id"],
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "p50": float(row["p50"]),
            "p90": float(row["p90"]),
            "p95": float(row["p95"]),
            "p99": float(row["p99"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
            "histogram": [int(row[name]) for name in bucket_names if name in row],
        }
        stats_list.append(stats)

    return stats_list


def read_latency_csv(csv_path) -> List[Dict]:
    """
    Read latency stats from a CSV file and convert to list of dicts.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of latency stats dicts suitable for compute_global_latency_stats()
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    bucket_names = _get_latency_bucket_names()

    stats_list = []
    for _, row in df.iterrows():
        stats = {
            "chunk_id": row["chunk_id"],
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "p50": float(row["p50"]),
            "p90": float(row["p90"]),
            "p95": float(row["p95"]),
            "p99": float(row["p99"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
            "histogram": [int(row[name]) for name in bucket_names if name in row],
        }
        stats_list.append(stats)

    return stats_list
