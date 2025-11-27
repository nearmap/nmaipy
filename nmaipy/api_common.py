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
from typing import Dict, Optional, Tuple

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
                    except:
                        pass
                self._sessions.clear()
        else:  # Fallback if attributes don't exist
            if hasattr(self, "_sessions"):
                for session in self._sessions:
                    try:
                        session.close()
                    except:
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
            except:
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
