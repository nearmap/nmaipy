import concurrent.futures
import contextlib
import gzip
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from http import HTTPStatus
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely.geometry
import stringcase
from requests.adapters import HTTPAdapter
from shapely.geometry import MultiPolygon, Polygon, shape, GeometryCollection
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse, quote

from nmaipy import log, geometry_utils
from nmaipy.api_common import (
    APIError,
    AIFeatureAPIError,
    APIRequestSizeError,
    APIGridError,
    GriddedApiClient,
    RetryRequest,
    clean_api_key_from_string,
    generate_curl_command,
    format_error_message,
)
from nmaipy.api_common import API_KEY
from nmaipy.constants import (
    ADDRESS_FIELDS,
    AOI_EXCEEDS_MAX_SIZE,
    AOI_ID_COLUMN_NAME,
    API_CRS,
    AREA_CRS,
    BACKOFF_FACTOR,
    CHUNKED_ENCODING_RETRY_DELAY,
    CONNECTED_CLASS_IDS,
    DUMMY_STATUS_CODE,
    GRID_SIZE_DEGREES,
    LAT_LONG_CRS,
    MAX_AOI_AREA_SQM_BEFORE_GRIDDING,
    MAX_RETRIES,
    POLYGON_TOO_COMPLEX,
    READ_TIMEOUT_SECONDS,
    ROLLUP_SURVEY_DATE_ID,
    ROLLUP_SYSTEM_VERSION_ID,
    SINCE_COL_NAME,
    SLOW_REQUEST_THRESHOLD_SECONDS,
    SQUARED_METERS_TO_SQUARED_FEET,
    SURVEY_RESOURCE_ID_COL_NAME,
    TIMEOUT_SECONDS,
    UNTIL_COL_NAME,
)


logger = log.get_logger()


# Backward compatibility aliases - use generic versions from api_common
AIFeatureAPIGridError = APIGridError
AIFeatureAPIRequestSizeError = APIRequestSizeError


# Add these helper functions at the top of ai_offline_parcel.py
def close_sessions(sessions):
    """Helper function to properly close request sessions"""
    for session in sessions:
        try:
            session.close()
        except:
            pass


def cleanup_executor(executor):
    """Helper function to cleanup executor and its sessions"""
    try:
        # Get any sessions stored on the executor's threads
        sessions = []
        for thread in executor._threads:
            if hasattr(thread, "_local") and hasattr(thread._local, "session"):
                sessions.append(thread._local.session)

        # Close all sessions
        close_sessions(sessions)

        # Shutdown the executor
        executor.shutdown(wait=True)
    except:
        pass


class FeatureApi(GriddedApiClient):
    """
    Client for the Nearmap AI Feature API.

    Inherits from GriddedApiClient for automatic gridding support.
    """
    
    # Thread-local storage for executor reuse in gridding scenarios
    # This prevents nested thread pool creation
    _thread_local = threading.local()

    SOURCE_CRS = LAT_LONG_CRS
    FLOAT_COLS = [
        "fidelity",
        "confidence",
        "areaSqm",
        "clippedAreaSqm",
        "unclippedAreaSqm",
        "areaSqft",
        "clippedAreaSqft",
        "unclippedAreaSqft",
    ]
    API_TYPE_FEATURES = "features"
    API_TYPE_ROLLUPS = "rollups"
    POOL_SIZE = 10
    
    # HTTP status codes for retry logic
    RETRY_STATUS_CODES_BASE = [
        HTTPStatus.TOO_MANY_REQUESTS,      # 429
        HTTPStatus.BAD_GATEWAY,            # 502
        HTTPStatus.SERVICE_UNAVAILABLE,    # 503
        HTTPStatus.INTERNAL_SERVER_ERROR,  # 500
    ]
    RETRY_STATUS_CODES_WITH_TIMEOUT = RETRY_STATUS_CODES_BASE + [
        HTTPStatus.GATEWAY_TIMEOUT,        # 504
    ]

    def __init__(
        self,
        api_key: Optional[str] = API_KEY,
        bulk_mode: Optional[bool] = True,
        cache_dir: Optional[Path] = None,
        overwrite_cache: Optional[bool] = False,
        compress_cache: Optional[bool] = False,
        threads: Optional[int] = 10,
        alpha: Optional[bool] = False,
        beta: Optional[bool] = False,
        prerelease: Optional[bool] = False,
        only3d: Optional[bool] = False,
        url_root: Optional[str] = "api.nearmap.com/ai/features/v4/bulk",
        system_version_prefix: Optional[str] = "gen6-",
        system_version: Optional[str] = None,
        aoi_grid_min_pct: Optional[int] = 100,
        aoi_grid_inexact: Optional[bool] = False,
        parcel_mode: Optional[bool] = True,
        maxretry: int = MAX_RETRIES,
        rapid: Optional[bool] = False,
        order: Optional[str] = None,
        exclude_tiles_with_occlusion: Optional[bool] = False,
        progress_counters: Optional[dict] = None,
        grid_size: Optional[float] = GRID_SIZE_DEGREES,
    ):
        """
        Initialize FeatureApi class

        Args:
            api_key: Nearmap API key. If not defined the environment variable will be used
            cache_dir: Directory to use as a payload cache
            overwrite_cache: Set to overwrite values stored in the cache
            compress_cache: Whether to use gzip compression (.json.gz) or save raw json text (.json).
            threads: Number of threads to spawn for concurrent execution
            alpha: Include alpha features
            beta: Include beta features
            prerelease: Include prerelease features
            only3d: Only return features with 3D coverage
            url_root: The root URL for the API. Default is the bulk API.
            system_version_prefix: Prefix for the system version (e.g. "gen6-" to restrict to gen 6 results)
            system_version: System version to use (e.g. "gen6-glowing_grove-1.0" to restrict to exact version matches)
            aoi_grid_min_pct: Minimum percentage of sub-gridded squares the AOI must get valid responses from.
            aoi_grid_inexact: Accept grids combined from multiple dates/survey IDs.
            parcel_mode: When set to True, uses the API's parcel mode which filters features based on parcel boundaries.
            maxretry: Number of retries to attempt on a failed request
            rapid: When True, rapid survey resources will be considered (for damage classification)
            order: Specify "earliest" or "latest" for date-based requests (defaults to "latest")
            exclude_tiles_with_occlusion: When True, ignores survey resources with occluded tiles
            progress_counters: Optional dict with 'total' and 'completed' counters for tracking progress across processes
            grid_size: Grid cell size in degrees for subdividing large AOIs (default ~200m)
        """
        # Initialize thread-safety attributes first
        self._sessions = []
        self._thread_local = threading.local()
        self._lock = threading.Lock()

        # Store progress counters for cross-process progress tracking
        self.progress_counters = progress_counters

        if not bulk_mode:
            url_root = "api.nearmap.com/ai/features/v4"

        URL_ROOT = f"https://{url_root}"
        self.FEATURES_URL = URL_ROOT + "/features.json"
        self.ROLLUPS_CSV_URL = URL_ROOT + "/rollups.csv"
        self.FEATURES_SURVEY_RESOURCE_URL = URL_ROOT + "/surveyresources"
        self.CLASSES_URL = URL_ROOT + "/classes.json"
        self.PACKS_URL = URL_ROOT + "/packs.json"

        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("API_KEY", None)
        if self.api_key is None:
            raise ValueError(
                "No API KEY provided. Provide a key when initializing FeatureApi or set an environmental " "variable"
            )
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        elif overwrite_cache:
            raise ValueError(f"No cache dir specified, but overwrite cache set to True.")

        self.overwrite_cache = overwrite_cache
        self.compress_cache = compress_cache
        self.threads = threads
        self.bulk_mode = bulk_mode
        self.alpha = alpha
        self.beta = beta
        self.prerelease = prerelease
        self.only3d = only3d
        self.system_version_prefix = system_version_prefix
        self.system_version = system_version
        self.aoi_grid_min_pct = aoi_grid_min_pct
        self.aoi_grid_inexact = aoi_grid_inexact
        self.parcel_mode = parcel_mode
        self.maxretry = maxretry
        self.rapid = rapid
        self.order = order
        self.exclude_tiles_with_occlusion = exclude_tiles_with_occlusion
        self.grid_size = grid_size

        # Semaphore to limit concurrent gridding operations
        # This prevents too many file handles being opened when many large AOIs grid simultaneously
        # 1/5th prevents file handle exhaustion while still allowing parallelism
        max_concurrent_gridding = max(1, self.threads // 5)
        self._gridding_semaphore = threading.Semaphore(max_concurrent_gridding)
        logger.debug(f"Initialized gridding semaphore with limit of {max_concurrent_gridding} concurrent AOIs")

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

    def _increment_progress(self):
        """Increment progress counter immediately after each request completes"""
        if self.progress_counters is None:
            return

        with self.progress_counters['lock']:
            self.progress_counters['completed'] += 1

    @contextlib.contextmanager
    def _session_scope(self, in_gridding_mode=False):
        """Context manager for session lifecycle - creates fresh session each time to prevent accumulation"""
        # Always create a fresh session to prevent resource accumulation
        session = requests.Session()

        # Configure retries based on context: skip 504 retries initially, retry within gridding
        status_forcelist = self.RETRY_STATUS_CODES_WITH_TIMEOUT if in_gridding_mode else self.RETRY_STATUS_CODES_BASE

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
        # Dynamic pool sizing: match pool size to thread count to prevent blocking
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
        # Note: session.timeout is not a real attribute - we need to pass timeout to each request
        session._timeout = (TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)

        try:
            yield session
        finally:
            # Always close the session to prevent resource leaks
            try:
                session.close()
            except:
                pass

    def _get_feature_api_results_as_data(self, base_url: str) -> Tuple[requests.Response, Dict]:
        """
        Return a result from one of the base URLS (such as packs or classes)
        """
        with self._session_scope() as session:
            request_string = f"{base_url}?apikey={self.api_key}"
            if self.alpha:
                request_string += "&alpha=true"
            if self.beta:
                request_string += "&beta=true"
            response = session.get(request_string, timeout=session._timeout)
            if not response.ok:
                raise RuntimeError(f"\n{self._clean_api_key(request_string)=}\n\n{response.status_code=}\n\n{response.text}\n\n")
            return response, response.json()

    def get_packs(self) -> Dict[str, List[str]]:
        """
        Get packs with class IDs
        """
        response, data = self._get_feature_api_results_as_data(self.PACKS_URL)
        return {p["code"]: [c["id"] for c in p["featureClasses"]] for p in data["packs"]}

    def get_feature_classes(self, packs: List[str] = None) -> pd.DataFrame:
        """
        Get the feature class IDs and descriptions as a dataframe.

        Args:
            packs: If defined, classes will be filtered to the set of packs
        """

        # Request data
        t1 = time.monotonic()
        response, data = self._get_feature_api_results_as_data(self.CLASSES_URL)
        response_time_ms = (time.monotonic() - t1) * 1e3
        logger.debug(f"{response_time_ms:.1f}ms response time for classes.json")
        df_classes = pd.DataFrame(data["classes"]).set_index("id")

        # Filter classes to packs
        if packs:
            pack_classes = self.get_packs()
            if diff := set(packs) - set(pack_classes.keys()):
                raise ValueError(f"Unknown packs: {diff}")
            all_classes = list(set([class_id for p in packs for class_id in pack_classes[p]]))
            # Strip out any classes that we don't get a valid description for from the "packs" endpoint.
            all_classes = [c for c in all_classes if c in df_classes.index]
            df_classes = df_classes.loc[all_classes]
            df_classes = df_classes.query("type == 'Feature'")  # Filter out non-feature classes

        return df_classes




    def _clean_api_key(self, request_string: str) -> str:
        """
        Remove the API key from a request string using proper URL parsing.
        This ensures URL-encoded API keys are also properly redacted.
        """
        result = request_string

        # First, try to parse as URL and clean any apikey parameters
        try:
            parsed = urlparse(request_string)

            # If we have query parameters, clean them
            if parsed.query or '?' in request_string:
                # Handle case where there's no scheme (e.g., "/path?query")
                if not parsed.scheme and '?' in request_string:
                    path_part, query_part = request_string.split('?', 1)
                    query_params = parse_qsl(query_part, keep_blank_values=True)
                else:
                    query_params = parse_qsl(parsed.query, keep_blank_values=True)

                # Replace any apikey parameter value
                cleaned_params = [
                    ('apikey', 'APIKEYREMOVED') if k.lower() == 'apikey' else (k, v)
                    for k, v in query_params
                ]

                # Reconstruct the URL
                if not parsed.scheme and '?' in request_string:
                    result = f"{path_part}?{urlencode(cleaned_params)}"
                else:
                    result = urlunparse((
                        parsed.scheme,
                        parsed.netloc,
                        parsed.path,
                        parsed.params,
                        urlencode(cleaned_params),
                        parsed.fragment
                    ))
        except:
            pass  # If URL parsing fails, continue with simple replacement

        # Always do simple replacement as a catch-all
        # This handles API keys in non-URL contexts or different formats
        if hasattr(self, 'api_key') and self.api_key:
            result = result.replace(self.api_key, "APIKEYREMOVED")

        return result


    def _request_cache_path(self, request_string: str) -> Path:
        """
        Hash a request string to create a cache path.
        """
        request_string = self._clean_api_key(request_string)
        request_hash = hashlib.md5(request_string.encode()).hexdigest()
        lon, lat = geometry_utils.extract_coords_for_cache(request_string)
        ext = "json.gz" if self.compress_cache else "json"
        return self.cache_dir / lon / lat / f"{request_hash}.{ext}"
        
    def _post_request_cache_path(self, url: str, body: dict) -> Path:
        """
        Hash a POST request URL and body to create a cache path.

        Uses lon/lat based caching for geometry queries, or country/state/city/zip
        based caching for address-only queries (shared with Roof Age API).
        """
        from urllib.parse import parse_qs, urlparse

        # Clean API key from URL
        clean_url = self._clean_api_key(url)

        # Convert body to a stable string representation for cache key
        body_str = json.dumps(body, sort_keys=True)
        combined_str = clean_url + body_str
        request_hash = hashlib.md5(combined_str.encode()).hexdigest()

        ext = "json.gz" if self.compress_cache else "json"

        # Check if there's geometry in the body - use lon/lat based caching
        if "aoi" in body and body["aoi"].get("type") in ["Polygon", "MultiPolygon"]:
            # Get first coordinate from first polygon
            if body["aoi"]["type"] == "Polygon":
                coords = body["aoi"]["coordinates"][0][0]
            else:  # MultiPolygon
                coords = body["aoi"]["coordinates"][0][0][0]

            lon, lat = str(int(float(coords[0]))), str(int(float(coords[1])))
            cache_subdir = self.cache_dir / lon / lat
            cache_subdir.mkdir(parents=True, exist_ok=True)
            return cache_subdir / f"{request_hash}.{ext}"

        # No geometry - check for address fields in URL for address-based caching
        parsed = urlparse(url)
        params = parse_qs(parsed.query)

        # Extract address components (params are lists, get first value)
        country = params.get("country", [""])[0]
        state = params.get("state", [""])[0]
        city = params.get("city", [""])[0]
        zipcode = params.get("zip", [""])[0]

        # If we have address fields, use address-based caching (shared with Roof Age API)
        if country or state or city or zipcode:
            return self._get_address_cache_path(
                country=country or "_unknown_",
                state=state or "_unknown_",
                city=city or "_unknown_",
                zipcode=zipcode or "_unknown_",
                cache_key=combined_str
            )

        # Fallback: hash-based flat structure
        return self.cache_dir / f"{request_hash}.{ext}"


    def _handle_response_errors(self, response: requests.Response, request_string: str):
        """
        Handle errors returned from the feature API
        """
        clean_request_string = self._clean_api_key(request_string)
        if response is None:
            raise AIFeatureAPIError(response, clean_request_string)
        elif not response.ok:
            raise AIFeatureAPIError(response, clean_request_string)

    def _write_to_cache(self, path, payload):
        """
        Write a payload to the cache. To make the write atomic, data is first written to a temp file and then renamed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # Generate temp file name based on whether we're compressing
        temp_filename = f"{str(uuid.uuid4())}.tmp"
        if self.compress_cache:
            temp_filename = f"{temp_filename}.gz"
        temp_path = self.cache_dir / temp_filename
        
        try:
            if self.compress_cache:
                with gzip.open(temp_path, "wb") as f:
                    f.write(json.dumps(payload).encode("utf-8"))
            else:
                with open(temp_path, "w") as f:
                    json.dump(payload, f)
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

        
    def _create_post_request(
        self,
        base_url: str,
        geometry: Union[Polygon, MultiPolygon],
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        address_fields: Optional[Dict[str, str]] = None,
        survey_resource_id: Optional[str] = None,
        region: Optional[str] = None,
        param_dic: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, dict, bool]:
        """
        Create parameters for a POST request with given parameters
        base_url: Need to choose one of: self.FEATURES_URL, self.ROLLUPS_CSV_URL

        Returns:
            - url: The URL to send the POST request to
            - body: The JSON body for the POST request
            - exact: Whether the geometry is exact or had to be approximated (always True for POST requests)
        """
        # Determine the correct URL
        url = base_url
        if survey_resource_id is not None:
            url = f"{self.FEATURES_SURVEY_RESOURCE_URL}/{survey_resource_id}/features.json"

        # Add apikey as query parameter
        url = f"{url}?apikey={self.api_key}"

        # Create request body - only include 'aoi' in the body
        body = {}

        # Convert geometry to GeoJSON format
        if geometry is not None:
            body["aoi"] = shapely.geometry.mapping(geometry)

        # Add all other parameters as query parameters, not in body
        if self.alpha:
            url += "&alpha=true"
        if self.beta:
            url += "&beta=true"
        if self.prerelease:
            url += "&prerelease=true"
        if self.only3d:
            url += "&3dCoverage=true"
        if self.parcel_mode:
            url += "&parcelMode=true"
        if self.system_version_prefix is not None:
            url += f"&systemVersionPrefix={self.system_version_prefix}"
        if self.system_version is not None:
            url += f"&systemVersion={self.system_version}"
        if self.rapid:
            url += "&rapid=true"
        if self.order is not None:
            url += f"&order={self.order}"
        if self.exclude_tiles_with_occlusion:
            url += "&excludeTilesWithOcclusion=true"

        # Add dates as query parameters if given
        if ((since is not None) or (until is not None)) and (survey_resource_id is not None):
            logger.debug(
                f"Request made with survey_resource_id {survey_resource_id} and either since or until - ignoring dates."
            )
        elif (since is not None) or (until is not None):
            if since:
                if not isinstance(since, str):
                    raise ValueError("Since must be a string")
                url += f"&since={since}"
            if until:
                if not isinstance(until, str):
                    raise ValueError("Until must be a string")
                url += f"&until={until}"

        # Add packs as query parameters if given
        if packs:
            pack_param = packs if isinstance(packs, str) else ",".join(packs)
            url += f"&packs={pack_param}"

        # Add classes as query parameters if given
        if classes:
            class_param = classes if isinstance(classes, str) else ",".join(classes)
            url += f"&classes={class_param}"

        # Add include as query parameters if given
        if include:
            include_param = include if isinstance(include, str) else ",".join(include)
            url += f"&include={include_param}"

        # Add address fields as query parameters if given (for address-based queries)
        if address_fields:
            # Add country parameter for address-based queries (API requires uppercase)
            # The region parameter uses the same values as AREA_CRS keys (au, ca, nz, us)
            if region:
                region_lower = region.lower()
                if region_lower not in AREA_CRS:
                    valid_regions = ", ".join(sorted(AREA_CRS.keys()))
                    raise ValueError(
                        f"Invalid region '{region}'. Must be one of: {valid_regions}"
                    )
                url += f"&country={region_lower.upper()}"
            for field in ADDRESS_FIELDS:
                if field in address_fields and address_fields[field]:
                    # URL-encode address values to handle spaces, apostrophes, etc.
                    encoded_value = quote(str(address_fields[field]))
                    url += f"&{field}={encoded_value}"

        # Add custom parameters from param_dic if provided
        if param_dic:
            for key, value in param_dic.items():
                # URL-encode values to handle special characters
                encoded_value = quote(str(value))
                url += f"&{key}={encoded_value}"

        # With POST requests, we always get the exact geometry processed
        exact = True

        return url, body, exact

    def get_features(
        self,
        geometry: Union[Polygon, MultiPolygon],
        region: str,
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        address_fields: Optional[Dict[str, str]] = None,
        survey_resource_id: Optional[str] = None,
        in_gridding_mode: bool = False,
        param_dic: Optional[Dict[str, str]] = None,
    ):
        """
        Get features for the provided geometry.

        Args:
            geometry: The polygon or multipolygon to query
            region: Country code
            packs: List of AI packs to include
            classes: List of feature classes to include
            since: Start date for the query
            until: End date for the query
            address_fields: Address fields for address-based queries
            survey_resource_id: ID of the specific survey to query
            param_dic: Optional dictionary of custom parameters to add to the API request

        Returns:
            API response as a dictionary
        """
        data = self._get_results(
            geometry=geometry,
            region=region,
            packs=packs,
            classes=classes,
            include=include,
            since=since,
            until=until,
            address_fields=address_fields,
            survey_resource_id=survey_resource_id,
            result_type=self.API_TYPE_FEATURES,
            in_gridding_mode=in_gridding_mode,
            param_dic=param_dic,
        )
        return data

    def get_rollup(
        self,
        geometry: Union[Polygon, MultiPolygon],
        region: str,
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        address_fields: Optional[Dict[str, str]] = None,
        survey_resource_id: Optional[str] = None,
        param_dic: Optional[Dict[str, str]] = None,
    ):
        """
        Get rollup data for the provided geometry.

        Args:
            geometry: The polygon or multipolygon to query
            region: Country code
            packs: List of AI packs to include
            classes: List of feature classes to include
            since: Start date for the query
            until: End date for the query
            address_fields: Address fields for address-based queries
            survey_resource_id: ID of the specific survey to query
            param_dic: Optional dictionary of custom parameters to add to the API request

        Returns:
            API response as CSV text
        """
        data = self._get_results(
            geometry=geometry,
            region=region,
            packs=packs,
            classes=classes,
            since=since,
            until=until,
            address_fields=address_fields,
            survey_resource_id=survey_resource_id,
            result_type=self.API_TYPE_ROLLUPS,
            param_dic=param_dic,
        )
        return data

    def _get_results(
        self,
        geometry: Union[Polygon, MultiPolygon],
        region: str,
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        address_fields: Optional[Dict[str, str]] = None,
        survey_resource_id: Optional[str] = None,
        result_type: str = API_TYPE_FEATURES,
        in_gridding_mode: bool = False,
        param_dic: Optional[Dict[str, str]] = None,
    ):
        """
        Get feature data for an AOI. If a cache is configured, the cache will be checked before using the API.

        Args:
            geometry: AOI in EPSG4326
            region: Country code, used for recalculating areas.
            packs: List of AI packs
            since: Earliest date to pull data for
            until: Latest date to pull data for
            address_fields: Fields for an address based query (rather than query AOI based query).
            survey_resource_id: The ID of the survey resource id if an exact survey is requested for the pull.
                               NB: This is NOT the survey ID from coverage - it is the id of the AI resource attached to that survey.
            result_type: Type of API endpoint (features or rollups)
            param_dic: Optional dictionary of custom parameters to add to the API request

        Returns:
            API response as a Dictionary
        """
        with self._session_scope(in_gridding_mode) as session:
            # Determine the base URL based on the result type and packs
            if result_type == self.API_TYPE_FEATURES:
                base_url = self.FEATURES_URL
            elif result_type == self.API_TYPE_ROLLUPS:
                base_url = self.ROLLUPS_CSV_URL
                
            # Use POST request with JSON body for better geometry handling
            url, body, exact = self._create_post_request(
                base_url=base_url,
                geometry=geometry,
                packs=packs,
                classes=classes,
                include=include,
                since=since,
                until=until,
                address_fields=address_fields,
                survey_resource_id=survey_resource_id,
                region=region,
                param_dic=param_dic,
            )
            cache_path = None if self.cache_dir is None else self._post_request_cache_path(url, body)
                
            # Check if it's already cached
            if self.cache_dir is not None and not self.overwrite_cache:
                if cache_path.exists():
                    if self.compress_cache:
                        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                            try:
                                payload_str = f.read()
                                return json.loads(payload_str)
                            except EOFError as e:
                                logger.error(f"Error loading compressed cache file {cache_path}.")
                                logger.error(f"Error: {e}")
                    else:
                        with open(cache_path, "r") as f:
                            return json.load(f)

            # Request data with retry loop for ChunkedEncodingError
            # Note: While urllib3 RetryRequest handles ChunkedEncodingError, this additional
            # retry loop exists to catch cases where the response is successfully received but
            # fails during reading. After exhausting retries, persistent ChunkedEncodingErrors
            # are treated as size errors to trigger gridding (the response may be too large).
            t1 = time.monotonic()
            response = None
            request_info = None

            for retry_attempt in range(MAX_RETRIES):
                try:
                    headers = {'Content-Type': 'application/json'}
                    response = session.post(url, json=body, headers=headers, timeout=session._timeout)
                    request_info = f"{url} with body {json.dumps(body)}"  # For error reporting

                    # Log geometry if urllib3 performed any retries (check response history)
                    if hasattr(response, 'history') and len(response.history) > 0:
                        aoi_geom = body.get('aoi', {}) if body else {}
                        num_retries = len(response.history)
                        status = response.status_code if hasattr(response, 'status_code') else 'unknown'
                        logger.info(f"Request required {num_retries} urllib3 retry(ies), final status {status}, AOI geometry: {json.dumps(aoi_geom)}")

                    break  # Success, exit retry loop
                except requests.exceptions.ReadTimeout as e:
                    # Treat read timeout exactly like a 504 Gateway Timeout from the server
                    # This will trigger gridding for large requests that timeout
                    logger.debug(f"Read timeout after {READ_TIMEOUT_SECONDS}s on attempt {retry_attempt + 1}/{MAX_RETRIES}, treating as 504")

                    # Create a mock 504 response object
                    class TimeoutAs504Response:
                        status_code = HTTPStatus.GATEWAY_TIMEOUT  # 504
                        text = f"Client-side read timeout after {READ_TIMEOUT_SECONDS} seconds - treating as 504"
                        ok = False
                        def json(self):
                            return {"error": "Read timeout treated as gateway timeout", "code": "READ_TIMEOUT_AS_504"}

                    response = TimeoutAs504Response()
                    request_info = f"{url} (timed out after {READ_TIMEOUT_SECONDS}s)"
                    break  # Exit retry loop and handle as 504 below
                except requests.exceptions.ChunkedEncodingError as e:
                    if retry_attempt < MAX_RETRIES - 1:
                        # Log debug message for retry attempts
                        logger.debug(f"ChunkedEncodingError on attempt {retry_attempt + 1}/{MAX_RETRIES}, retrying: {e}")
                        time.sleep(CHUNKED_ENCODING_RETRY_DELAY)  # Brief pause before retry
                        continue
                    else:
                        # Exhausted all retries, log error and fall back to size error
                        logger.error(f"ChunkedEncodingError persisted after {MAX_RETRIES} attempts, treating as size error to trigger gridding: {e}")
                        raise AIFeatureAPIRequestSizeError(None, self._clean_api_key(url))

            response_time_ms = (time.monotonic() - t1) * 1e3
            response_time_seconds = response_time_ms / 1000

            # Log response at debug level (can be enabled when diagnosing issues)
            sanitized_url = self._clean_api_key(url)
            logger.debug(f"Response from {sanitized_url} in {response_time_seconds:.1f}s (status: {response.status_code if response else 'unknown'})")

            # Log slow successful requests (failures go in errors CSV)
            if response.ok and response_time_seconds > SLOW_REQUEST_THRESHOLD_SECONDS:
                sanitized_url = self._clean_api_key(url)
                # Add geometry as GeoJSON if available in body
                geom_info = ""
                if body and "aoi" in body:
                    geom_info = f" | Geometry: {json.dumps(body['aoi'])}"
                logger.info(f"Slow request: {response_time_seconds:.1f}s response time for {sanitized_url}{geom_info}")

            if response.ok:
                if result_type == self.API_TYPE_ROLLUPS:
                    data = response.text
                elif result_type == self.API_TYPE_FEATURES:
                    try:
                        data = response.json()
                    except Exception as e:
                        # Treat JSON parsing errors as size errors to trigger gridding
                        logger.debug(f"JSON parsing error - treat as size error to try again with a gridded approach: {e}")
                        raise AIFeatureAPIRequestSizeError(response, self._clean_api_key(url))

                # Save to cache if configured
                if self.cache_dir is not None:
                    self._write_to_cache(cache_path, data)
                return data
            else:
                try:
                    status_code = response.status_code
                except requests.exceptions.ChunkedEncodingError:
                    status_code = ""
                try:
                    text = response.text
                except requests.exceptions.ChunkedEncodingError:
                    text = ""

                # Clean up cache if we're overwriting
                if self.overwrite_cache and cache_path:
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass

                if status_code in AIFeatureAPIRequestSizeError.status_codes:
                    logger.debug(f"Raising AIFeatureAPIRequestSizeError from status code {status_code=}")
                    raise AIFeatureAPIRequestSizeError(response, self._clean_api_key(request_info))
                elif status_code == HTTPStatus.BAD_REQUEST:
                    try:
                        error_data = json.loads(text)
                        error_code = error_data.get("code", "")
                        if error_code in AIFeatureAPIRequestSizeError.status_codes:
                            logger.debug(f"Raising AIFeatureAPIRequestSizeError from secondary status code {status_code=}")
                            raise AIFeatureAPIRequestSizeError(response, self._clean_api_key(request_info))
                    except (json.JSONDecodeError, KeyError):
                        pass

                    # Check for other errors
                    self._handle_response_errors(response, request_info)
                else:
                    # Handle other HTTP errors
                    self._handle_response_errors(response, request_info)

    @staticmethod
    def link_to_date(link: str) -> str:
        """
        Parse the date from a Map Browser link.
        """
        date = link.split("/")[-1]
        return f"{date[:4]}-{date[4:6]}-{date[6:8]}"

    @staticmethod
    def add_location_marker_to_link(link: str) -> str:
        """
        Check whether the link contains the location marker flag, and add it if not present.
        """
        location_marker_string = "?locationMarker"
        if location_marker_string in link:
            return link
        else:
            return link + location_marker_string




    @classmethod
    def payload_gdf(cls, payload: dict, aoi_id: Optional[str] = None, parcel_mode: Optional[bool] = False) -> Tuple[gpd.GeoDataFrame, dict]:
        """
        Create a GeoDataFrame from a feature API response dictionary.

        Args:
            payload: API response dictionary
            aoi_id: Optional ID for the AOI to add to the data
            parcel_mode: If True, filter out features where belongsToParcel is False

        Returns:
            Features GeoDataFrame
            Metadata dictionary
        """

        # Create metadata
        metadata = {
            "system_version": payload["systemVersion"],
            "link": cls.add_location_marker_to_link(payload["link"]),
            "date": cls.link_to_date(payload["link"]),
            "survey_id": payload["surveyId"],
            "survey_resource_id": payload["resourceId"],
            "perspective": payload["perspective"],
            "postcat": payload["postcat"],
        }

        columns = [
            "id",
            "classId",
            "description",
            "confidence",
            "fidelity",  # not on every class
            "parentId",
            "geometry",
            "clippedGeometry",  # Add clippedGeometry to columns list
            "areaSqm",
            "clippedAreaSqm",
            "unclippedAreaSqm",
            "areaSqft",
            "clippedAreaSqft",
            "unclippedAreaSqft",
            "attributes",
            "damage",  # Damage classification data (new structure for building lifecycle)
            "surveyDate",
            "meshDate",
            "belongsToParcel",  # New field from parcelMode API
        ]

        # Create features DataFrame
        if len(payload["features"]) == 0:
            df = pd.DataFrame([], columns=columns)
        else:
            # Filter features if parcel_mode is enabled
            features = payload["features"]
            if parcel_mode and len(features) > 0 and "belongsToParcel" in features[0]:
                features = [f for f in features if f.get("belongsToParcel", True)]

            # Check for and use clippedGeometry if present
            # Add a new flag for multiparcel features (those with clippedGeometry)
            for feature in features:
                # Default is False - not a multiparcel feature
                feature["multiparcel_feature"] = False

                # If clippedGeometry exists, use it and mark as multiparcel feature
                if "clippedGeometry" in feature and feature["clippedGeometry"]:
                    feature["geometry"] = feature["clippedGeometry"]
                    feature["multiparcel_feature"] = True

            df = pd.DataFrame(features)
            for col_name in set(columns).difference(set(df.columns)):
                df[col_name] = None

        for col in FeatureApi.FLOAT_COLS:
            if (col in df):
                df[col] = df[col].astype("float")

        df = df.rename(columns={"id": "feature_id"})
        df.columns = [stringcase.snakecase(c) for c in df.columns]

        # Add AOI ID if specified
        if aoi_id is not None:
            try:
                df[AOI_ID_COLUMN_NAME] = [aoi_id] * len(df)
                df = df.set_index(AOI_ID_COLUMN_NAME)
            except Exception as e:
                logger.error(
                    f"Problem setting aoi_id to {AOI_ID_COLUMN_NAME} as {aoi_id=} (dataframe has {len(df)} rows)."
                )
                raise ValueError
            metadata[AOI_ID_COLUMN_NAME] = aoi_id

        # Cast to GeoDataFrame - now we drop clipped_geometry since we've already used it
        if "geometry" in df.columns:
            gdf = gpd.GeoDataFrame(df.assign(geometry=df.geometry.apply(shape)))
            # Drop the clipped_geometry column if it exists
            if "clipped_geometry" in gdf.columns:
                gdf = gdf.drop(columns=["clipped_geometry"])
            gdf = gdf.set_crs(cls.SOURCE_CRS)
        else:
            gdf = df

        return gdf, metadata

    @classmethod
    def payload_rollup_df(cls, payload: dict, aoi_id: Optional[str] = None) -> Tuple[gpd.GeoDataFrame, dict]:
        """
        Create a dataframe from a rollup API response dictionary.

        Args:
            payload: API response dictionary
            aoi_id: Optional ID for the AOI to add to the data

        Returns:
            Features GeoDataFrame
            Metadata dictionary
        """

        # Create metadata
        payload_io = StringIO(payload)
        df = pd.read_csv(payload_io, header=[0, 1])  # Accounts for first header row as uuids, second as descriptions
        metadata = {
            "system_version": df.filter(regex=ROLLUP_SYSTEM_VERSION_ID).iloc[0, 0],
            "link": "",  # TODO: Once link is returned in payloads, add in here.
            "date": df.filter(regex=ROLLUP_SURVEY_DATE_ID).iloc[0, 0],
        }

        # Add AOI ID if specified
        if aoi_id is not None:
            try:
                df[AOI_ID_COLUMN_NAME] = aoi_id
                df = df.set_index(AOI_ID_COLUMN_NAME)
            except Exception as e:
                logger.error(
                    f"Problem setting aoi_id in col {AOI_ID_COLUMN_NAME} as {aoi_id} (dataframe has {len(df)} rows)."
                )
                raise ValueError
            metadata[AOI_ID_COLUMN_NAME] = aoi_id
        return df, metadata

    def _attempt_gridding(
        self,
        geometry: Union[Polygon, MultiPolygon],
        region: str,
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        aoi_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        survey_resource_id: Optional[str] = None,
        aoi_grid_inexact: Optional[bool] = None,
        reason: Optional[str] = None,
    ) -> Tuple[Optional[gpd.GeoDataFrame], Optional[dict], Optional[dict]]:
        """
        Helper method to attempt gridding large AOIs and handle the common pattern of gridding, 
        combining results, and creating metadata.
        
        This method is called when an AOI is too large for a single API request, either
        proactively (when area > MAX_AOI_AREA_SQM_BEFORE_GRIDDING) or reactively
        (after receiving AIFeatureAPIRequestSizeError).
        
        Args:
            geometry: AOI geometry in EPSG:4326 (WGS84)
            region: Country/region code (e.g., 'us', 'au') used for CRS selection
            packs: List of AI packs to query (e.g., ['building', 'vegetation', 'damage'])
            classes: List of specific feature class IDs to include
            include: List of feature types to include in results
            aoi_id: Unique identifier for the AOI (used in output and caching)
            since: Start date for temporal filtering (ISO format: YYYY-MM-DD)
            until: End date for temporal filtering (ISO format: YYYY-MM-DD)
            survey_resource_id: Specific survey resource ID to query
            aoi_grid_inexact: If True, allows combining results from different survey dates
                             across grid cells. If None, uses instance default.
        
        Returns:
            Tuple containing:
            - features_gdf: GeoDataFrame with combined features from all grid cells, or None if error
            - metadata: Dictionary with survey metadata, or None if error
            - error: Dictionary with error details if gridding failed, None if successful
        """
        try:
            # Use the provided aoi_grid_inexact parameter, or fall back to the instance default
            allow_inexact_gridding = aoi_grid_inexact if aoi_grid_inexact is not None else self.aoi_grid_inexact
            features_gdf, metadata_df, errors_df = self.get_features_gdf_gridded(
                geometry=geometry,
                region=region,
                packs=packs,
                classes=classes,
                include=include,
                aoi_id=aoi_id,
                since=since,
                until=until,
                survey_resource_id=survey_resource_id,
                aoi_grid_inexact=allow_inexact_gridding,
                grid_size=self.grid_size,
                reason=reason
            )
            error = None  # Reset error if we got here without an exception

            # Recombine gridded features
            features_gdf = geometry_utils.combine_features_from_grid(features_gdf)

            # Create metadata
            metadata_df = metadata_df.drop_duplicates().iloc[0]
            metadata = {
                AOI_ID_COLUMN_NAME: metadata_df[AOI_ID_COLUMN_NAME],
                "system_version": metadata_df["system_version"],
                "link": metadata_df["link"],
                "date": metadata_df["date"],
                "survey_id": metadata_df["survey_id"],
                "survey_resource_id": metadata_df["survey_resource_id"],
                "perspective": metadata_df["perspective"],
                "postcat": metadata_df["postcat"],
            }

            # Return grid cell errors as 4th element (may include geometry for failed grid squares)
            return features_gdf, metadata, error, errors_df

        except (AIFeatureAPIError, AIFeatureAPIGridError) as e:
            # Catch acceptable errors - these are complete grid failures
            features_gdf = None
            metadata = None
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": e.status_code,
                "message": e.message,
                "text": e.text[:200] if e.text else "",
                "request": "Grid error",
                "failure_type": "grid",
            }
            return features_gdf, metadata, error, None
        except Exception as grid_error:
            logger.error(f"Gridding failed for AOI (id {aoi_id}): {grid_error}")
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": None,
                "message": f"Gridding failed: {str(grid_error)}",
                "text": str(grid_error)[:200],
                "request": "Gridding failed",
                "failure_type": "grid",
            }
            return None, None, error, None

    def get_features_gdf(
        self,
        geometry: Union[Polygon, MultiPolygon],
        region: str,
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        aoi_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        address_fields: Optional[Dict[str, str]] = None,
        survey_resource_id: Optional[str] = None,
        fail_hard_regrid: Optional[bool] = False,
        in_gridding_mode: bool = False,
        param_dic: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[gpd.GeoDataFrame], Optional[dict], Optional[dict]]:
        """
        Get feature data for an AOI. If a cache is configured, the cache will be checked before using the API.
        Data is returned as a GeoDataframe with response metadata and error information (if any occurred).

        Args:
            geometry: AOI in EPSG4326
            region: The country code, used as a key to AREA_CRS.
            packs: List of AI packs
            classes: List of classes
            aoi_id: ID of the AOI to add to the data
            since: Earliest date to pull data for
            until: Latest date to pull data for
            address_fields: dictionary with values for the address fields, if available, or else None
            survey_resource_id: Alternative query mechanism to retrieve precise survey's results from coverage.
            fail_hard_regrid: If set to true, don't try and grid on an AIFeatureAPIRequestSizeError,
                              This option is here because we can get stuck in an infinite loop of
                              get_features_gdf -> get_features_gdf_gridded -> get_features_gdf_bulk -> get_features_gdf
                              and we need to be able to stop at 2nd call to get_features_gdf if we get another
                              AIFeatureAPIRequestSizeError
            param_dic: Optional dictionary of custom parameters to add to the API request
        Returns:
            API response features GeoDataFrame, metadata dictionary, and an error dictionary
        """
        if geometry is None and address_fields is None:
            raise Exception(
                f"Internal Error: get_features_gdf was called with NEITHER a geometry NOR address fields specified. This should be impossible"
            )
        
        # Check if AOI is too large and should be gridded directly
        if geometry is not None and not fail_hard_regrid and not in_gridding_mode:
            # Validate region parameter
            if region not in AREA_CRS:
                raise ValueError(f"Invalid region '{region}'. Valid regions are: {list(AREA_CRS.keys())}")
            
            # Convert geometry to appropriate CRS for area calculation
            geometry_gdf = gpd.GeoSeries([geometry], crs=API_CRS)
            geometry_projected = geometry_gdf.to_crs(AREA_CRS[region])
            area_sqm = geometry_projected.area.iloc[0]
            
            if area_sqm > MAX_AOI_AREA_SQM_BEFORE_GRIDDING:
                logger.debug(f"AOI (id {aoi_id}) area ({area_sqm:.0f} sqm) exceeds client threshold ({MAX_AOI_AREA_SQM_BEFORE_GRIDDING} sqm). Forcing gridding...")
                features_gdf, metadata, error, grid_errors_df = self._attempt_gridding(
                    geometry=geometry,
                    region=region,
                    packs=packs,
                    classes=classes,
                    include=include,
                    aoi_id=aoi_id,
                    since=since,
                    until=until,
                    survey_resource_id=survey_resource_id,
                    aoi_grid_inexact=self.aoi_grid_inexact,
                    reason=f"proactive - area {area_sqm:.0f} sqm > {MAX_AOI_AREA_SQM_BEFORE_GRIDDING} sqm"
                )
                return features_gdf, metadata, error, grid_errors_df
        
        try:
            features_gdf, metadata, error, grid_errors_df = None, None, None, None
            payload = self.get_features(
                geometry=geometry,
                region=region,
                packs=packs,
                classes=classes,
                include=include,
                since=since,
                until=until,
                address_fields=address_fields,
                survey_resource_id=survey_resource_id,
                in_gridding_mode=in_gridding_mode,
                param_dic=param_dic
            )
            features_gdf, metadata = self.payload_gdf(payload, aoi_id, self.parcel_mode)
        except AIFeatureAPIRequestSizeError as e:
            features_gdf, metadata, error = None, None, None

            # If the query was too big, split it up into a grid, and recombine as though it was one query.
            # Do not get stuck in an infinite loop of re-gridding and timing out
            if fail_hard_regrid or geometry is None:
                # If we're in gridding mode and this is a 400 (polygon complexity) error, try simplification
                if fail_hard_regrid and e.status_code == HTTPStatus.BAD_REQUEST and geometry is not None:
                    logger.warning(f"Grid cell still too complex for aoi_id {aoi_id}, applying minimal geometry simplification")
                    try:
                        simplified_geometry = geometry.simplify(tolerance=0.000001)  # ~5-11cm accuracy in EPSG:4326
                        # Retry with simplified geometry
                        payload = self.feature_request_payload(
                            geojson=simplified_geometry.__geo_interface__,
                            region=region,
                            packs=packs,
                            classes=classes,
                            include=include,
                            since=since,
                            until=until,
                            survey_resource_id=survey_resource_id,
                            in_gridding_mode=in_gridding_mode
                        )
                        features_gdf, metadata = self.payload_gdf(payload, aoi_id, self.parcel_mode)
                        error = None
                        logger.info(f"Geometry simplification successful for aoi_id {aoi_id}")
                    except Exception as simplify_error:
                        logger.error(f"Geometry simplification failed for aoi_id {aoi_id}: {simplify_error}")
                        error = {
                            AOI_ID_COLUMN_NAME: aoi_id,
                            "status_code": e.status_code,
                            "message": e.message,
                            "text": e.text[:200] if e.text else "",  # Truncate long text
                            "request": "Size error - geometry simplification failed",
                            "failure_type": "standard",
                        }
                else:
                    logger.debug("Failing hard and NOT re-gridding....")
                    error = {
                        AOI_ID_COLUMN_NAME: aoi_id,
                        "status_code": e.status_code,
                        "message": e.message,
                        "text": e.text[:200] if e.text else "",  # Truncate long text
                        "request": "Size error - request too large",
                        "failure_type": "standard",
                    }
            else:
                # First request was too big, so grid it up, recombine, and return. Any problems and the whole AOI should return an error as usual.
                logger.debug(f"Found an over-sized AOI (id {aoi_id}). Trying gridding...")
                features_gdf, metadata, error, grid_errors_df = self._attempt_gridding(
                    geometry=geometry,
                    region=region,
                    packs=packs,
                    classes=classes,
                    include=include,
                    aoi_id=aoi_id,
                    since=since,
                    until=until,
                    survey_resource_id=survey_resource_id,
                    aoi_grid_inexact=self.aoi_grid_inexact,
                    reason=f"reactive - HTTP {e.status_code}: {e.message}"
                )
        except AIFeatureAPIError as e:
            # Catch acceptable errors
            features_gdf = None
            metadata = None
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": e.status_code,
                "message": e.message,
                "text": e.text,
                "request": e.request_string,
                "failure_type": "standard",
            }

        except requests.exceptions.RetryError as e:
            status_code = e.response.status_code if e.response else DUMMY_STATUS_CODE
            logger.warning(
                f"Retry Exception - gave up retrying on aoi_id: {aoi_id} near {geometry.representative_point()}. Status code: {status_code}"
            )
            if logger.level == logging.DEBUG:
                # Generate debug information including curl command
                url, body, _ = self._create_post_request(
                    base_url=self.FEATURES_URL,
                    geometry=geometry,
                    packs=packs,
                    classes=classes,
                    include=include,
                    since=since,
                    until=until,
                    survey_resource_id=survey_resource_id,
                )
                logger.debug(f"Failed request URL: {self._clean_api_key(url)}")
                logger.debug("To reproduce this request, use the following curl command:")
                logger.debug(generate_curl_command(url, body, method="POST", timeout=READ_TIMEOUT_SECONDS))
            features_gdf = None
            metadata = None
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": status_code,
                "message": "RETRY_ERROR",
                "text": "Retry Error",
                "request": f"Geometry with {len(str(geometry))} chars",
                "failure_type": "standard",
            }
        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout Exception on aoi_id: {aoi_id} near {geometry.representative_point()}")
            if logger.level == logging.DEBUG and geometry is not None:
                # Generate debug information including curl command
                url, body, _ = self._create_post_request(
                    base_url=self.FEATURES_URL,
                    geometry=geometry,
                    packs=packs,
                    classes=classes,
                    include=include,
                    since=since,
                    until=until,
                    survey_resource_id=survey_resource_id,
                )
                logger.debug(f"Timeout on request URL: {self._clean_api_key(url)}")
                logger.debug("To reproduce this request, use the following curl command:")
                logger.debug(generate_curl_command(url, body, method="POST", timeout=READ_TIMEOUT_SECONDS))
            features_gdf = None
            metadata = None
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": DUMMY_STATUS_CODE,
                "message": "TIMEOUT_ERROR",
                "text": str(e),
                "request": f"Geometry with {len(str(geometry))} chars",
                "failure_type": "standard",
            }

        # Round the confidence column to two decimal places (nearest percent)
        if features_gdf is not None and "confidence" in features_gdf.columns:
            features_gdf["confidence"] = features_gdf["confidence"].round(2)

        # Increment progress counter for completed request (uses batching)
        self._increment_progress()

        return features_gdf, metadata, error, grid_errors_df

    def get_features_gdf_gridded(
        self,
        geometry: Union[Polygon, MultiPolygon],
        region: str,
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        aoi_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        survey_resource_id: Optional[str] = None,
        aoi_grid_inexact: Optional[bool] = False,
        grid_size: Optional[float] = GRID_SIZE_DEGREES,
        reason: Optional[str] = None
    ) -> Tuple[Optional[gpd.GeoDataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Get feature data for an AOI. If a cache is configured, the cache will be checked before using the API.
        Data is returned as a GeoDataframe with response metadata and error information (if any occurred).

        Args:
            geometry: AOI in EPSG4326
            region: Country code
            packs: List of AI packs
            classes: List of classes
            aoi_id: ID of the AOI to add to the data
            since: Earliest date to pull data for
            until: Latest date to pull data for
            survey_resource_id: The survey resource ID for the vector tile resources.
            grid_size: The AOI is gridded in the native projection (constants.API_CRS) to save compute.

        Returns:
            API response features GeoDataFrame, metadata dictionary, and an error dictionary
        """
        # Acquire semaphore to limit concurrent gridding operations
        # This prevents too many file handles being opened simultaneously
        with self._gridding_semaphore:
            df_gridded = geometry_utils.split_geometry_into_grid(geometry=geometry, cell_size=grid_size)

            reason_str = f" (reason: {reason})" if reason else ""
            logger.debug(f"Gridding AOI {aoi_id}: split into {len(df_gridded)} grid cells{reason_str}")

            # Update progress counter: we're splitting 1 AOI into N grid cells, so add (N-1) to total
            if self.progress_counters is not None:
                num_grid_cells = len(df_gridded)
                with self.progress_counters['lock']:
                    self.progress_counters['total'] += (num_grid_cells - 1)
                logger.debug(f"Gridding AOI {aoi_id}: added {num_grid_cells - 1} requests to progress total (1 AOI -> {num_grid_cells} grid cells)")

            # Retrieve the features for every one of the cells in the gridded AOIs
            # Assign temp integer IDs to grid cells and set as index to ensure dtype consistency
            # when errors_df is created from iterrows() in get_features_gdf_bulk()
            aoi_id_tmp = range(len(df_gridded))
            df_gridded[AOI_ID_COLUMN_NAME] = aoi_id_tmp
            df_gridded = df_gridded.set_index(AOI_ID_COLUMN_NAME)

            try:
                # If we are already in a 'gridded' call, then when we call get_features_gdf_bulk
                # we need to pass in fail_hard_regrid=True so we don't get stuck in an endless loop
                features_gdf, metadata_df, errors_df = self.get_features_gdf_bulk(
                gdf=df_gridded,
                region=region,
                packs=packs,
                classes=classes,
                include=include,
                since_bulk=since,
                until_bulk=until,
                survey_resource_id_bulk=survey_resource_id,
                max_allowed_error_pct=100 - self.aoi_grid_min_pct,
                fail_hard_regrid=True,
                in_gridding_mode=True,
                )
            except AIFeatureAPIError as e:
                logger.debug(
                    f"Failed whole grid for aoi_id {aoi_id}. Errors exceeded max allowed (min valid {self.aoi_grid_min_pct}%) ({e.status_code})."
                )
                raise AIFeatureAPIGridError(e.status_code)
            if len(features_gdf) == 0:
                # Got no data back from any grid square in the AOI.
                raise AIFeatureAPIGridError(DUMMY_STATUS_CODE, message="No data returned from grid query for AOI.")
            elif len(features_gdf["survey_date"].unique()) > 1:
                if aoi_grid_inexact:
                    logger.info(
                        f"Multiple dates detected for aoi_id {aoi_id} - certain to contain duplicates on grid boundaries."
                    )
                else:
                    logger.warning(
                        f"Failed whole grid for aoi_id {aoi_id}. Multiple dates detected - certain to contain duplicates on grid boundaries."
                    )
                    raise AIFeatureAPIGridError(
                        DUMMY_STATUS_CODE, message="Multiple dates on non survey resource ID query."
                    )
            elif survey_resource_id is None:
                logger.debug(
                    f"AOI {aoi_id} gridded on a single date - possible but unlikely to include deduplication errors (if two overlapping surveys flown on same date)."
                )
                # TODO: We should change query to guarantee same survey id is used somehow.

            if len(errors_df) > 0:
                # Check if we have any non-404 errors (which indicate retriable failures)
                non_404_errors = errors_df[errors_df['status_code'] != 404]
                
                if len(non_404_errors) > 0:
                    # Log warning for non-404 errors that create holes in the grid
                    logger.warning(
                        f"Allowing partial grid results on aoi {aoi_id} with {len(features_gdf)} good results and {len(errors_df)} errors of types {errors_df.status_code.value_counts().to_json()}."
                    )
                    # Log a random sample of non-404 errors to avoid massive output
                    if len(non_404_errors) > 0:
                        sample_size = min(5, len(non_404_errors))
                        error_sample = non_404_errors[['status_code', 'message']].sample(n=sample_size, random_state=42)
                        logger.warning(f"Sample of {sample_size} non-404 errors: {error_sample.to_dict('records')}")
                else:
                    # Only 404s - use debug level as before
                    logger.debug(
                        f"Allowing partial grid results on aoi {aoi_id} with {len(features_gdf)} good results and {len(errors_df)} errors of types {errors_df.status_code.value_counts().to_json()}."
                    )

                # raise AIFeatureAPIGridError(errors_df.query("status_code != 200").status_code.mode())

            # Reset the correct aoi_id for the gridded result
            features_gdf[AOI_ID_COLUMN_NAME] = aoi_id
            metadata_df[AOI_ID_COLUMN_NAME] = aoi_id

            # Process errors_df to add grid cell geometry and mark as partial failures
            if len(errors_df) > 0:
                # errors_df has the temp grid cell IDs in a column, df_gridded has them as index
                # Merge on the aoi_id column (errors_df) with the index (df_gridded)
                errors_with_geom = errors_df.merge(
                    df_gridded[['geometry']],
                    left_on=AOI_ID_COLUMN_NAME,
                    right_index=True,
                    how='left'
                )
                # Update aoi_id to the actual AOI (not the temp grid cell ID)
                errors_with_geom[AOI_ID_COLUMN_NAME] = aoi_id
                # Mark these as grid failures (individual grid cells failed, but AOI has some data)
                errors_with_geom['failure_type'] = 'grid'
                errors_df = errors_with_geom

            return features_gdf, metadata_df, errors_df

    def get_features_gdf_bulk(
        self,
        gdf: gpd.GeoDataFrame,
        region: str,
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        since_bulk: Optional[str] = None,
        until_bulk: Optional[str] = None,
        survey_resource_id_bulk: Optional[str] = None,
        max_allowed_error_pct: Optional[int] = 100,
        fail_hard_regrid: Optional[bool] = False,
        in_gridding_mode: bool = False,
    ) -> Tuple[Optional[gpd.GeoDataFrame], Optional[pd.DataFrame], pd.DataFrame]:
        """
        Get features data for many AOIs.

        Args:
            gdf: GeoDataFrame with AOIs
            region: Country code
            packs: List of AI packs
            classes: List of classes
            since_bulk: Earliest date to pull data for, applied across all Query AOIs.
            until_bulk: Latest date to pull data for, applied across all Query AOIs.
            survey_resource_id_bulk: Impose a single survey resource ID from which to pull all responses.
            max_allowed_error_pct:  Raise an AIFeatureAPIError if we exceed this proportion of errors. Otherwise, create a dataframe of errors and return all good data available.
            fail_hard_regrid: should be False on an initial call, this just gets used internally to prevent us
                              getting stuck in an infinite loop of get_features_gdf -> get_features_gdf_gridded ->
                                                                   get_features_gdf_bulk -> get_features_gdf

        Returns:
            API responses as feature GeoDataFrames, metadata DataFrame, and an error DataFrame
        """
        try:
            max_allowed_error_count = round(len(gdf) * max_allowed_error_pct / 100)

            # are address fields present?
            has_address_fields = set(gdf.columns.tolist()).intersection(set(ADDRESS_FIELDS)) == set(ADDRESS_FIELDS)
            # is a geometry field present?
            has_geom = "geometry" in gdf.columns

            # Check if we're already in an executor context (gridding scenario)
            # This prevents nested thread pool creation
            if hasattr(self._thread_local, 'executor') and self._thread_local.executor is not None:
                # Reuse existing executor - we're in a gridding operation
                executor = self._thread_local.executor
                external_executor = True
                logger.debug(f"Reusing parent executor for gridding {len(gdf)} cells")
            else:
                # Create new executor - we're at the top level
                executor = concurrent.futures.ThreadPoolExecutor(self.threads)
                self._thread_local.executor = executor
                external_executor = False
                logger.debug(f"Created new executor with {self.threads} threads for {len(gdf)} AOIs")
            
            # Submit jobs to executor
            jobs = []
            for aoi_id, row in gdf.iterrows():
                # Overwrite blanket since/until dates with per request since/until if columns are present
                since = since_bulk
                if SINCE_COL_NAME in row:
                    if isinstance(row[SINCE_COL_NAME], str):
                        since = row[SINCE_COL_NAME]
                until = until_bulk
                if UNTIL_COL_NAME in row:
                    if isinstance(row[UNTIL_COL_NAME], str):
                        until = row[UNTIL_COL_NAME]
                survey_resource_id = survey_resource_id_bulk
                if SURVEY_RESOURCE_ID_COL_NAME in row:
                    if isinstance(row[SURVEY_RESOURCE_ID_COL_NAME], str):
                        survey_resource_id = row[SURVEY_RESOURCE_ID_COL_NAME]
                jobs.append(
                    executor.submit(
                        self.get_features_gdf,
                        geometry=row.geometry if has_geom else None,
                        region=region,
                        packs=packs,
                        classes=classes,
                        include=include,
                        aoi_id=aoi_id,
                        since=since,
                        until=until,
                        address_fields={f: row[f] for f in ADDRESS_FIELDS} if has_address_fields and not has_geom else None,
                        survey_resource_id=survey_resource_id,
                        fail_hard_regrid=fail_hard_regrid,
                        in_gridding_mode=in_gridding_mode,
                    )
                )
            
            data = []
            metadata = []
            errors = []
            grid_errors = []  # Accumulate grid cell errors (DataFrames with geometry)
            # Use as_completed to process results as they finish, preventing blocking on slow requests
            for job in concurrent.futures.as_completed(jobs):
                aoi_data, aoi_metadata, aoi_error, aoi_grid_errors = job.result()
                if aoi_data is not None:
                    if len(aoi_data) > 0:
                        data.append(aoi_data)
                if aoi_metadata is not None:
                    if len(aoi_metadata) > 0:
                        metadata.append(aoi_metadata)
                if aoi_error is not None:
                    if len(errors) > max_allowed_error_count:
                        cleanup_executor(executor)
                        logger.debug(
                            f"Exceeded maximum error count of {max_allowed_error_count} out of {len(gdf)} AOIs."
                        )
                        raise AIFeatureAPIError(aoi_error, aoi_error["request"])
                    else:
                        errors.append(aoi_error)
                # Collect grid cell errors (partial failures with geometry)
                if aoi_grid_errors is not None and len(aoi_grid_errors) > 0:
                    grid_errors.append(aoi_grid_errors)
        finally:
            # Only cleanup executor if we created it (not reusing from parent)
            if not external_executor:
                executor.shutdown(wait=True)
                self._thread_local.executor = None
                self.cleanup()  # Clean up sessions after bulk operation

        non_empty_data = [df for df in data if len(df) > 0]
        if len(non_empty_data) > 0:
            features_gdf = pd.concat(non_empty_data)
            # Ensure we have a proper GeoDataFrame with geometry column and CRS
            features_gdf = gpd.GeoDataFrame(features_gdf, geometry='geometry', crs=API_CRS)
        else:
            # Create empty GeoDataFrame with geometry column to avoid CRS assignment error
            features_gdf = gpd.GeoDataFrame(columns=['geometry'], crs=API_CRS)
        if len(data) == 0:
            features_gdf.index.name = AOI_ID_COLUMN_NAME
        
        metadata_df = pd.DataFrame(metadata).set_index(AOI_ID_COLUMN_NAME) if len(metadata) > 0 else pd.DataFrame([])
        if len(metadata) == 0:
            metadata_df.index.name = AOI_ID_COLUMN_NAME

        # Combine single-AOI errors (from dict) and grid cell errors (from DataFrames)
        errors_dfs = []
        if len(errors) > 0:
            errors_dfs.append(pd.DataFrame(errors).set_index(AOI_ID_COLUMN_NAME))
        if len(grid_errors) > 0:
            # Grid errors are already DataFrames with geometry - concat them
            grid_errors_combined = pd.concat(grid_errors, ignore_index=True)
            # Ensure aoi_id is set as index if present
            if AOI_ID_COLUMN_NAME in grid_errors_combined.columns:
                grid_errors_combined = grid_errors_combined.set_index(AOI_ID_COLUMN_NAME)
            errors_dfs.append(grid_errors_combined)

        if len(errors_dfs) > 0:
            errors_df = pd.concat(errors_dfs)
        else:
            errors_df = pd.DataFrame([])
            errors_df.index.name = AOI_ID_COLUMN_NAME
        return features_gdf, metadata_df, errors_df

    def get_rollup_df(
        self,
        geometry: Union[Polygon, MultiPolygon],
        region: str,
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        aoi_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        address_fields: Optional[Dict[str, str]] = None,
        survey_resource_id: Optional[str] = None,
    ) -> Tuple[Optional[gpd.GeoDataFrame], Optional[dict], Optional[dict]]:
        """
        Get rollup data for an AOI. If a cache is configured, the cache will be checked before using the API.
        Data is returned as a dataframe with response metadata and error information (if any occurred).

        Args:
            geometry: AOI in EPSG4326
            region: The country code, used as a key to AREA_CRS.
            packs: List of AI
            classes: List of classes
            aoi_id: ID of the AOI to add to the data
            since: Earliest date to pull data for
            until: Latest date to pull data for
            address_fields: dictionary with values for the address fields, if available, or else None
            survey_resource_id: Alternative query mechanism to retrieve precise survey's results from coverage.
            fail_hard_regrid: If set to true, don't try and grid on an AIFeatureAPIRequestSizeError,
                              This option is here because we can get stuck in an infinite loop of
                              get_features_gdf -> get_features_gdf_gridded -> get_features_gdf_bulk -> get_features_gdf
                              and we need to be able to stop at 2nd call to get_features_gdf if we get another
                              AIFeatureAPIRequestSizeError
        Returns:
            API response features GeoDataFrame, metadata dictionary, and an error dictionary
        """
        if geometry is None and address_fields is None:
            raise Exception(
                f"Internal Error: get_features_gdf was called with NEITHER a geometry NOR address fields specified. This should be impossible"
            )

        try:
            rollup_df, metadata, error = None, None, None
            payload = self.get_rollup(
                geometry, region, packs, classes, since, until, address_fields, survey_resource_id
            )
            rollup_df, metadata = self.payload_rollup_df(payload, aoi_id)
        except AIFeatureAPIError as e:
            # Catch acceptable errors
            rollup_df = None
            metadata = None
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": e.status_code,
                "message": e.message,
                "text": e.text,
                "request": e.request_string,
                "failure_type": "standard",
            }

        except requests.exceptions.RetryError as e:
            logger.debug(f"Retry Exception - gave up retrying on aoi_id: {aoi_id}")
            rollup_df = None
            metadata = None
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": DUMMY_STATUS_CODE,
                "message": "RETRY_ERROR",
                "failure_type": "standard",
                "text": str(e),
                "request": "No request info",
            }
        return rollup_df, metadata, error

    @classmethod
    def _single_to_multi_index(cls, columns: pd.Index):
        """
        Take the columns from a rollup dataframe (which had the id and description collapsed with a pipe), and separate
        them out again - first level as the id (empty for added columns with no GUID attached). Second level as the
        description.
        """
        out_cols = columns.str.split("|")
        out_cols = pd.MultiIndex.from_tuples([[""] + d if len(d) == 1 else d for d in out_cols])
        out_cols.names = ["id", "description"]
        return out_cols

    @classmethod
    def _multi_to_single_index(cls, columns: pd.MultiIndex):
        """
        Inverse of _single_to_multi_index: flatten the columns from multi_index to flat, by joining levels with a pipe
        character. For use in rollup headings which have "id" and a "description" as two separate levels (from the
        two header rows in the returned csv file).
        This operation is important when merging with single level dataframes.
        """
        out_cols = columns.map("|".join).str.strip("|")
        return out_cols

    def get_rollup_df_bulk(
        self,
        gdf: gpd.GeoDataFrame,
        region: str,
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        since_bulk: Optional[str] = None,
        until_bulk: Optional[str] = None,
        survey_resource_id_bulk: Optional[str] = None,
        max_allowed_error_pct: Optional[int] = 100,
    ) -> Tuple[Optional[gpd.GeoDataFrame], Optional[pd.DataFrame], pd.DataFrame]:
        """
        Get features data for many AOIs.

        Args:
            gdf: GeoDataFrame with AOIs
            region: Country code
            packs: List of AI packs
            classes: List of classes
            since_bulk: Earliest date to pull data for, applied across all Query AOIs.
            until_bulk: Latest date to pull data for, applied across all Query AOIs.
            survey_resource_id_bulk: Impose a single survey resource ID from which to pull all responses.
            max_allowed_error_pct:  Raise an AIFeatureAPIError if we exceed this proportion of errors. Otherwise, create a dataframe of errors and
            return all good data available.
            fail_hard_regrid: should be False on an initial call, this just gets used internally to prevent us
                              getting stuck in an infinite loop of get_features_gdf -> get_features_gdf_gridded ->
                                                                   get_features_gdf_bulk -> get_features_gdf

        Returns:
            API responses as rollup csv GeoDataFrames, metadata DataFrame, and an error DataFrame
        """
        max_allowed_error_count = round(len(gdf) * max_allowed_error_pct / 100)

        # are address fields present?
        has_address_fields = set(gdf.columns.tolist()).intersection(set(ADDRESS_FIELDS)) == set(ADDRESS_FIELDS)
        # is a geometry field present?
        has_geom = "geometry" in gdf.columns

        # Run in thread pool
        with concurrent.futures.ThreadPoolExecutor(self.threads) as executor:
            try:
                jobs = []
                for aoi_id, row in gdf.iterrows():
                    # Overwrite blanket since/until dates with per request since/until if columns are present
                    since = since_bulk
                    if SINCE_COL_NAME in row:
                        if isinstance(row[SINCE_COL_NAME], str):
                            since = row[SINCE_COL_NAME]
                    until = until_bulk
                    if UNTIL_COL_NAME in row:
                        if isinstance(row[UNTIL_COL_NAME], str):
                            until = row[UNTIL_COL_NAME]
                    survey_resource_id = survey_resource_id_bulk
                    if SURVEY_RESOURCE_ID_COL_NAME in row:
                        if isinstance(row[SURVEY_RESOURCE_ID_COL_NAME], str):
                            survey_resource_id = row[SURVEY_RESOURCE_ID_COL_NAME]

                    jobs.append(
                        executor.submit(
                            self.get_rollup_df,
                            row.geometry if has_geom else None,
                            region,
                            packs,
                            classes,
                            aoi_id,
                            since,
                            until,
                            {f: row[f] for f in ADDRESS_FIELDS} if has_address_fields and not has_geom else None,
                            survey_resource_id,
                        )
                    )
                data = []
                metadata = []
                errors = []
                # Use as_completed to process results as they finish, preventing blocking on slow requests
                for job in concurrent.futures.as_completed(jobs):
                    aoi_data, aoi_metadata, aoi_error = job.result()
                    if aoi_data is not None:
                        data.append(aoi_data)
                    if aoi_metadata is not None:
                        metadata.append(aoi_metadata)
                    if aoi_error is not None:
                        if len(errors) > max_allowed_error_count:
                            raise AIFeatureAPIError(aoi_error, aoi_error["request"])
                        else:
                            errors.append(aoi_error)
            finally:
                executor.shutdown(wait=True)  # Ensure cleanup
                self.cleanup()  # Clean up sessions
        # Combine results
        # RANT: there can be some... unpleasantness... with multipolygons, missing primary roofs and dtypes.
        # Presence can be boolean, or converted to float if some parts of a multipolygon AOI request are NaN.
        # This causes problems later with mixed data types, and e.g. writing chunks to parquet.
        # Using "convert_dtypes" fixes it by auto-converting the booleans which pandas decided should be floats due
        # to the NaNs, to an integer dtype, which then combines properly with the boolean dtype.
        rollup_df = pd.concat(data) if len(data) > 0 else None
        metadata_df = pd.DataFrame(metadata) if len(metadata) > 0 else None
        errors_df = pd.DataFrame(errors).set_index(AOI_ID_COLUMN_NAME) if len(errors) > 0 else pd.DataFrame([])
        if len(errors) == 0:
            errors_df.index.name = AOI_ID_COLUMN_NAME
        return rollup_df, metadata_df, errors_df
