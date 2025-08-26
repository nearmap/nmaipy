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
from http.client import RemoteDisconnected
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely.geometry
import stringcase
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from shapely.geometry import MultiPolygon, Polygon, shape, GeometryCollection
from urllib3.util.retry import Retry
import urllib3  # Add this with other imports
import ssl  # Add this with other imports

# Load environment variables from .env file
load_dotenv()

# Get API key, with fallback to empty string
API_KEY = os.getenv("API_KEY", "")

from nmaipy import log
from nmaipy.constants import (
    ADDRESS_FIELDS,
    AOI_EXCEEDS_MAX_SIZE,
    AOI_ID_COLUMN_NAME,
    API_CRS,
    AREA_CRS,
    CONNECTED_CLASS_IDS,
    LAT_LONG_CRS,
    MAX_AOI_AREA_SQM_BEFORE_GRIDDING,
    MAX_RETRIES,
    POLYGON_TOO_COMPLEX,
    ROLLUP_SURVEY_DATE_ID,
    ROLLUP_SYSTEM_VERSION_ID,
    SINCE_COL_NAME,
    SQUARED_METERS_TO_SQUARED_FEET,
    SURVEY_RESOURCE_ID_COL_NAME,
    UNTIL_COL_NAME,
    GRID_SIZE_DEGREES,
)

TIMEOUT_SECONDS = 120  # Max time to wait for a server response
READ_TIMEOUT_SECONDS = 1200  # Max time to wait for reading server response (10 minutes)
CHUNKED_ENCODING_RETRY_DELAY = 1.0  # Delay between ChunkedEncodingError retries
BACKOFF_FACTOR = 0.2  # Exponential backoff multiplier
DUMMY_STATUS_CODE = -1


logger = log.get_logger()


class RetryRequest(Retry):
    """
    Inherited retry request to limit back-off to 5 seconds.
    """

    BACKOFF_MIN = 0.2  # Minimum backoff time in seconds
    BACKOFF_MAX = 5    # Maximum backoff time in seconds

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

    def new_timeout(self, *args, **kwargs):
        """Override to enforce backoff time between 1-5 seconds"""
        timeout = super().new_timeout(*args, **kwargs)
        return min(max(timeout, self.BACKOFF_MIN), self.BACKOFF_MAX)  # Clamp between 1-5 seconds

    @classmethod
    def from_int(cls, retries, **kwargs):
        """Helper to create retry config with better defaults"""
        kwargs.setdefault('backoff_factor', BACKOFF_FACTOR)
        kwargs.setdefault('status_forcelist', [429, 500, 502, 503, 504])
        kwargs.setdefault('respect_retry_after_header', True)
        return super().from_int(retries, **kwargs)


class AIFeatureAPIError(Exception):
    """
    Error responses for logging from AI Feature API. Also include non rest API errors (use dummy status code and
    explicitly set messages).
    """

    def __init__(self, response, request_string, text="Query Not Attempted", message="Error with Query AOI"):
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
            except requests.exceptions.ChunkedEncodingError:
                self.message = "ChunkedEncodingError"
            except AttributeError:
                self.message = ""
        self.request_string = request_string
    
    def __str__(self):
        """Return a concise error message without the full response body"""
        return f"AIFeatureAPIError: {self.status_code} - {self.message}"


class AIFeatureAPIGridError(Exception):
    """
    Specific error to indicate that at least one of the requests comprising a gridded request has failed.
    """

    def __init__(self, status_code_error_mode, message=""):
        self.status_code = status_code_error_mode
        self.text = "Gridding and re-requesting failed on one or more grid cell queries."
        self.request_string = ""
        self.message = message
    
    def __str__(self):
        """Return a concise error message"""
        return f"AIFeatureAPIGridError: {self.status_code} - {self.message}"


class AIFeatureAPIRequestSizeError(AIFeatureAPIError):
    """
    Error indicating the size is, or might, be too large. Either through explicit size too large issues, or a timeout
    indicating that the server was unable to cope with the complexity of the geometries, which is usually fixed by
    querying a smaller AOI.
    """

    status_codes = (HTTPStatus.GATEWAY_TIMEOUT,)
    codes = (AOI_EXCEEDS_MAX_SIZE, POLYGON_TOO_COMPLEX)
    """
    Use to indicate when an AOI should be gridded and recombined, as it is too large for a request to handle (413, 504).
    """
    
    def __str__(self):
        """Return a concise error message without the full response body"""
        return f"AIFeatureAPIRequestSizeError: {self.status_code} - Request too large, should grid"


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


class FeatureApi:
    """
    Class to connect to the AI Feature API
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
        """
        # Initialize thread-safety attributes first
        self._sessions = []
        self._thread_local = threading.local()
        self._lock = threading.Lock()

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
        self._sessions = []
        self._thread_local = threading.local()
        self._lock = threading.Lock()  # Initialize lock here

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

        # Set timeouts (connect timeout, read timeout)
        session.timeout = (TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)
        
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
            response = session.get(request_string)
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

    @staticmethod
    def _polygon_to_coordstring(poly: Polygon) -> str:
        """
        Turn a shapely polygon into the format required by the API for a query polygon.
        """
        coords = poly.exterior.coords[:]
        flat_coords = np.array(coords).ravel()
        coordstring = ",".join(flat_coords.astype(str))
        return coordstring

    @staticmethod
    def _clip_features_to_polygon(
        feature_poly_series: gpd.GeoSeries, geometry: Union[Polygon, MultiPolygon], region: str
    ) -> gpd.GeoDataFrame:
        """
        Take polygon of a feature, and reclip it to a new background geometry. Return the clipped polygon,
        and suitably rounded area in sqm and sqft. Args: feature_poly: Polygon of a single feature. geometry: Polygon
        to be used as a clipping mask (e.g. Query AOI). region: country region.


        Returns: A dataframe with same structure and rows, but corrected values.

        """
        assert isinstance(feature_poly_series, gpd.GeoSeries)
        gdf_clip = gpd.GeoDataFrame(geometry=feature_poly_series.intersection(geometry), crs=feature_poly_series.crs)

        clipped_area_sqm = gdf_clip.to_crs(AREA_CRS[region]).area
        gdf_clip["clipped_area_sqft"] = (clipped_area_sqm * SQUARED_METERS_TO_SQUARED_FEET).round()
        gdf_clip["clipped_area_sqm"] = clipped_area_sqm.round(1)
        return gdf_clip

    @staticmethod
    def _make_latlon_path_for_cache(request_string: str):
        r = request_string.split("?")[-1]
        dic = dict([token.split("=") for token in r.split("&")])
        lon, lat = np.array(dic["polygon"].split(",")).astype("float").round().astype("int").astype("str")[:2]
        return lon, lat

    def _clean_api_key(self, request_string: str) -> str:
        """
        Remove the API key from a request string.
        """
        return request_string.replace(self.api_key, "APIKEYREMOVED")

    def _request_cache_path(self, request_string: str) -> Path:
        """
        Hash a request string to create a cache path.
        """
        request_string = self._clean_api_key(request_string)
        request_hash = hashlib.md5(request_string.encode()).hexdigest()
        lon, lat = self._make_latlon_path_for_cache(request_string)
        ext = "json.gz" if self.compress_cache else "json"
        return self.cache_dir / lon / lat / f"{request_hash}.{ext}"
        
    def _post_request_cache_path(self, url: str, body: dict) -> Path:
        """
        Hash a POST request URL and body to create a cache path.
        """
        # Clean API key from URL
        url = self._clean_api_key(url)

        # Convert body to a stable string representation and hash
        body_str = json.dumps(body, sort_keys=True)
        combined_str = url + body_str
        request_hash = hashlib.md5(combined_str.encode()).hexdigest()

        # Extract lon/lat from geometry for cache directory organization
        if "aoi" in body and body["aoi"].get("type") in ["Polygon", "MultiPolygon"]:
            # Get first coordinate from first polygon
            if body["aoi"]["type"] == "Polygon":
                coords = body["aoi"]["coordinates"][0][0]
            else:  # MultiPolygon
                coords = body["aoi"]["coordinates"][0][0][0]

            lon, lat = str(int(float(coords[0]))), str(int(float(coords[1])))
        else:
            # Fallback if no geometry or unexpected format
            lon, lat = "0", "0"

        ext = "json.gz" if self.compress_cache else "json"
        return self.cache_dir / lon / lat / f"{request_hash}.{ext}"

    def _request_error_message(self, request_string: str, response: requests.Response) -> str:
        """
        Create a descriptive error message without the API key.
        """
        return f"\n{self._clean_api_key(request_string)=}\n\n{response.status_code=}\n\n{response.text}\n\n"

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
        survey_resource_id: Optional[str] = None,
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
                survey_resource_id=survey_resource_id,
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
            t1 = time.monotonic()
            response = None
            request_info = None
            
            for retry_attempt in range(MAX_RETRIES):
                try:
                    headers = {'Content-Type': 'application/json'}
                    response = session.post(url, json=body, headers=headers)
                    request_info = f"{url} with body {json.dumps(body)}"  # For error reporting
                    break  # Success, exit retry loop
                except requests.exceptions.ChunkedEncodingError as e:
                    if retry_attempt < MAX_RETRIES - 1:
                        # Log debug message for retry attempts
                        logger.debug(f"ChunkedEncodingError on attempt {retry_attempt + 1}/{MAX_RETRIES}, retrying: {e}")
                        time.sleep(CHUNKED_ENCODING_RETRY_DELAY)  # Brief pause before retry
                        continue
                    else:
                        # Exhausted all retries, log error and fall back to size error
                        logger.error(f"ChunkedEncodingError persisted after {MAX_RETRIES} attempts, treating as size error to trigger gridding: {e}")
                        raise AIFeatureAPIRequestSizeError(None, url)

            response_time_ms = (time.monotonic() - t1) * 1e3
            logger.debug(f"{response_time_ms:.1f}ms response time")

            if response.ok:
                if result_type == self.API_TYPE_ROLLUPS:
                    data = response.text
                elif result_type == self.API_TYPE_FEATURES:
                    try:
                        data = response.json()
                    except Exception as e:
                        # Treat JSON parsing errors as size errors to trigger gridding
                        logger.debug(f"JSON parsing error - treat as size error to try again with a gridded approach: {e}")
                        raise AIFeatureAPIRequestSizeError(response, url)

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
                    raise AIFeatureAPIRequestSizeError(response, request_info)
                elif status_code == HTTPStatus.BAD_REQUEST:
                    try:
                        error_data = json.loads(text)
                        error_code = error_data.get("code", "")
                        if error_code in AIFeatureAPIRequestSizeError.codes:
                            logger.debug(f"Raising AIFeatureAPIRequestSizeError from secondary status code {status_code=}")
                            raise AIFeatureAPIRequestSizeError(response, request_info)
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

    @staticmethod
    def create_grid(df: gpd.GeoDataFrame, cell_size: float):
        """
        Create a GeodataFrame of grid squares, matching the extent of an input GeoDataFrame.
        """
        minx, miny, maxx, maxy = df.total_bounds
        w, h = (cell_size, cell_size)

        rows = int(np.ceil((maxy - miny) / h))
        cols = int(np.ceil((maxx - minx) / w))

        polygons = []
        for i in range(cols):
            for j in range(rows):
                polygons.append(
                    Polygon(
                        [
                            (minx + i * w, miny + (j + 1) * h),
                            (minx + (i + 1) * w, miny + (j + 1) * h),
                            (minx + (i + 1) * w, miny + j * h),
                            (minx + i * w, miny + j * h),
                        ]
                    )
                )
        df_grid = gpd.GeoDataFrame(geometry=polygons, crs=df.crs)
        return df_grid

    @staticmethod
    def split_geometry_into_grid(geometry: Union[Polygon, MultiPolygon], cell_size: float):
        """
        Take a geometry (implied CRS as API_CRS), and split it into a grid of given height/width cells (in degrees).

        Returns:
            Gridded GeoDataFrame
        """
        df = gpd.GeoDataFrame(geometry=[geometry], crs=API_CRS)
        df_gridded = FeatureApi.create_grid(df, cell_size)
        df_gridded = gpd.overlay(df, df_gridded, keep_geom_type=True)
        # explicit index_parts added to get rid of warning, and this was the default behaviour so I am
        # assuming this is the behaviour that is intended
        df_gridded = df_gridded.explode(index_parts=True)  # Break apart grid squares that are multipolygons.
        df_gridded = df_gridded.to_crs(API_CRS)
        return df_gridded

    @staticmethod
    def combine_features_gdf_from_grid(features_gdf: gpd.GeoDataFrame):
        """
        Where a grid of adjacent queries has been used to pull features, remove duplicates for discrete classes,
        and recombine geometries of connected classes (including reconciling areas correctly) where the feature id
        of the larger object is the same.

        param features_gdf: Output from FeatureAPI.payload_gdf
        :return:
        """
        
        # Handle empty or None input (when no features returned from any grid cell)
        if features_gdf is None or len(features_gdf) == 0:
            # Return empty GeoDataFrame with proper structure
            empty_gdf = gpd.GeoDataFrame(columns=['geometry'], crs=API_CRS)
            empty_gdf.index.name = AOI_ID_COLUMN_NAME
            return empty_gdf

        # Columns that don't require aggregation.
        agg_cols_first = [
            AOI_ID_COLUMN_NAME,
            "class_id",
            "description",
            "confidence",
            "parent_id",
            "unclipped_area_sqm",
            "unclipped_area_sqft",
            "attributes",
            "survey_date",
            "mesh_date",
            "fidelity",
        ]

        # Columns with clipped areas that should be summed when geometries are merged.
        agg_cols_sum = [
            "area_sqm",
            "area_sqft",
            "clipped_area_sqm",
            "clipped_area_sqft",
        ]

        # Filter columns to only those that exist
        existing_agg_cols_first = [col for col in agg_cols_first if col in features_gdf.columns]
        existing_agg_cols_sum = [col for col in agg_cols_sum if col in features_gdf.columns]
        
        features_gdf_dissolved = (
            features_gdf.drop_duplicates(
                ["feature_id", "geometry"]
            )  # First, drop duplicate geometries rather than dissolving them together.
            .filter(existing_agg_cols_first + ["geometry", "feature_id"], axis=1)
            .dissolve(
                by="feature_id", aggfunc="first"
            )  # Then dissolve any remaining features that represent a single feature_id that has been split.
            .reset_index()
            .set_index("feature_id")
        )

        features_gdf_summed = (
            features_gdf.filter(existing_agg_cols_sum + ["feature_id"], axis=1)
            .groupby("feature_id")
            .aggregate(dict([c, "sum"] for c in existing_agg_cols_sum))
        )

        # final output - same format, same set of feature_ids, but fewer rows due to dedup and merging.
        # Set the index back to being the AOI_ID column
        features_gdf_out = features_gdf_dissolved.join(features_gdf_summed).reset_index().set_index(AOI_ID_COLUMN_NAME)

        return features_gdf_out

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
                aoi_grid_inexact=allow_inexact_gridding
            )
            error = None  # Reset error if we got here without an exception

            # Recombine gridded features
            features_gdf = FeatureApi.combine_features_gdf_from_grid(features_gdf)

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
            
            return features_gdf, metadata, error

        except (AIFeatureAPIError, AIFeatureAPIGridError) as e:
            # Catch acceptable errors
            features_gdf = None
            metadata = None
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": e.status_code,
                "message": e.message,
                "text": e.text[:200] if e.text else "",
                "request": "Grid error",
            }
            return features_gdf, metadata, error
        except Exception as grid_error:
            logger.error(f"Gridding failed for AOI (id {aoi_id}): {grid_error}")
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": None,
                "message": f"Gridding failed: {str(grid_error)}",
                "text": str(grid_error)[:200],
                "request": "Gridding failed",
            }
            return None, None, error

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
                logger.info(f"AOI (id {aoi_id}) area ({area_sqm:.0f} sqm) exceeds client threshold ({MAX_AOI_AREA_SQM_BEFORE_GRIDDING} sqm). Forcing gridding...")
                return self._attempt_gridding(
                    geometry=geometry,
                    region=region,
                    packs=packs,
                    classes=classes,
                    include=include,
                    aoi_id=aoi_id,
                    since=since,
                    until=until,
                    survey_resource_id=survey_resource_id,
                    aoi_grid_inexact=self.aoi_grid_inexact
                )
        
        try:
            features_gdf, metadata, error = None, None, None
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
                in_gridding_mode=in_gridding_mode
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
                        }
                else:
                    logger.debug("Failing hard and NOT re-gridding....")
                    error = {
                        AOI_ID_COLUMN_NAME: aoi_id,
                        "status_code": e.status_code,
                        "message": e.message,
                        "text": e.text[:200] if e.text else "",  # Truncate long text
                        "request": "Size error - request too large",
                    }
            else:
                # First request was too big, so grid it up, recombine, and return. Any problems and the whole AOI should return an error as usual.
                logger.debug(f"Found an over-sized AOI (id {aoi_id}). Trying gridding...")
                features_gdf, metadata, error = self._attempt_gridding(
                    geometry=geometry,
                    region=region,
                    packs=packs,
                    classes=classes,
                    include=include,
                    aoi_id=aoi_id,
                    since=since,
                    until=until,
                    survey_resource_id=survey_resource_id,
                    aoi_grid_inexact=self.aoi_grid_inexact
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
            }

        except requests.exceptions.RetryError as e:
            status_code = e.response.status_code if e.response else DUMMY_STATUS_CODE
            logger.warning(
                f"Retry Exception - gave up retrying on aoi_id: {aoi_id} near {geometry.representative_point()}. Status code: {status_code}"
            )
            if logger.level == logging.DEBUG:
                request_string = self._create_post_request(
                    base_url=self.FEATURES_URL,
                    geometry=geometry,
                    packs=packs,
                    classes=classes,
                    since=since,
                    until=until,
                    address_fields=address_fields,
                    survey_resource_id=survey_resource_id,
                )[0]
                logger.debug(f"Probable original request string was: {self._clean_api_key(request_string)}")
            features_gdf = None
            metadata = None
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": status_code,
                "message": "RETRY_ERROR",
                "text": "Retry Error",
                "request": f"Geometry with {len(str(geometry))} chars",
            }
        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout Exception on aoi_id: {aoi_id} near {geometry.representative_point()}")
            features_gdf = None
            metadata = None
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": DUMMY_STATUS_CODE,
                "message": "TIMEOUT_ERROR",
                "text": str(e),
                "request": f"Geometry with {len(str(geometry))} chars",
            }

        # Round the confidence column to two decimal places (nearest percent)
        if features_gdf is not None and "confidence" in features_gdf.columns:
            features_gdf["confidence"] = features_gdf["confidence"].round(2)
        return features_gdf, metadata, error

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
        grid_size: Optional[float] = GRID_SIZE_DEGREES
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
            logger.info(f"Gridding AOI {aoi_id}: acquiring gridding slot")
            
            df_gridded = FeatureApi.split_geometry_into_grid(geometry=geometry, cell_size=grid_size)
            
            logger.info(f"Gridding AOI {aoi_id}: split into {len(df_gridded)} grid cells")

            # Retrieve the features for every one of the cells in the gridded AOIs
            aoi_id_tmp = range(len(df_gridded))
            df_gridded[AOI_ID_COLUMN_NAME] = aoi_id_tmp

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
                        address_fields={f: row[f] for f in ADDRESS_FIELDS} if has_address_fields else None,
                        survey_resource_id=survey_resource_id,
                        fail_hard_regrid=fail_hard_regrid,
                        in_gridding_mode=in_gridding_mode,
                    )
                )
            
            data = []
            metadata = []
            errors = []
            # Use as_completed to process results as they finish, preventing blocking on slow requests
            for job in concurrent.futures.as_completed(jobs):
                aoi_data, aoi_metadata, aoi_error = job.result()
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
            
        errors_df = pd.DataFrame(errors) if len(errors) > 0 else pd.DataFrame([])
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
            }

        except requests.exceptions.RetryError as e:
            logger.debug(f"Retry Exception - gave up retrying on aoi_id: {aoi_id}")
            rollup_df = None
            metadata = None
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": DUMMY_STATUS_CODE,
                "message": "RETRY_ERROR",
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
                            {f: row[f] for f in ADDRESS_FIELDS} if has_address_fields else None,
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
        errors_df = pd.DataFrame(errors)
        return rollup_df, metadata_df, errors_df
