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
    MAX_RETRIES,
    ROLLUP_SURVEY_DATE_ID,
    ROLLUP_SYSTEM_VERSION_ID,
    SINCE_COL_NAME,
    SQUARED_METERS_TO_SQUARED_FEET,
    SURVEY_RESOURCE_ID_COL_NAME,
    UNTIL_COL_NAME,
)

TIMEOUT_SECONDS = 120  # Max time to wait for a server response.
DUMMY_STATUS_CODE = -1


logger = log.get_logger()


class RetryRequest(Retry):
    """
    Inherited retry request to limit back-off to 5 seconds.
    """

    BACKOFF_MAX = 5  # Maximum backoff time in seconds

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
        """Override to set a minimum backoff time"""
        timeout = super().new_timeout(*args, **kwargs)
        return max(timeout, 1.0)  # At least 1 second between retries

    @classmethod
    def from_int(cls, retries, **kwargs):
        """Helper to create retry config with better defaults"""
        kwargs.setdefault('backoff_factor', 1.0)
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


class AIFeatureAPIGridError(Exception):
    """
    Specific error to indicate that at least one of the requests comprising a gridded request has failed.
    """

    def __init__(self, status_code_error_mode, message=""):
        self.status_code = status_code_error_mode
        self.text = "Gridding and re-requesting failed on one or more grid cell queries."
        self.request_string = ""
        self.message = message


class AIFeatureAPIRequestSizeError(AIFeatureAPIError):
    """
    Error indicating the size is, or might, be too large. Either through explicit size too large issues, or a timeout
    indicating that the server was unable to cope with the complexity of the geometries, which is usually fixed by
    querying a smaller AOI.
    """

    status_codes = (HTTPStatus.GATEWAY_TIMEOUT,)
    codes = (AOI_EXCEEDS_MAX_SIZE,)
    """
    Use to indicate when an AOI should be gridded and recombined, as it is too large for a request to handle (413, 504).
    """
    pass


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

    CHAR_LIMIT = 3800
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
        system_version_prefix: Optional[str] = None,
        system_version: Optional[str] = None,
        aoi_grid_min_pct: Optional[int] = 100,
        aoi_grid_inexact: Optional[bool] = False,
        parcelMode: Optional[bool] = False,
        use_post: Optional[bool] = True,
        maxretry: int = MAX_RETRIES,
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
            parcelMode: When set to True, uses the API's parcel mode which filters features based on parcel boundaries.
            use_post: When set to True (default), uses POST requests for better geometry handling.
            maxretry: Number of retries to attempt on a failed request
        """
        # Initialize thread-safety attributes first
        self._sessions = []
        self._thread_local = threading.local()
        self._lock = threading.Lock()

        if not bulk_mode:
            url_root = "api.nearmap.com/ai/features/v4"

        URL_ROOT = f"https://{url_root}"
        self.FEATURES_URL = URL_ROOT + "/features.json"
        self.FEATURES_DAMAGE_URL = URL_ROOT + "/internal/pipelines/foo_fighters/features.json"
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
        self.parcelMode = parcelMode
        self.use_post = use_post
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
    def _session_scope(self):
        """Thread-safe context manager for session lifecycle"""
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            retries = RetryRequest(
                total=self.maxretry,
                backoff_factor=0.2,
                status_forcelist=[
                    HTTPStatus.TOO_MANY_REQUESTS,
                    HTTPStatus.BAD_GATEWAY,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    HTTPStatus.INTERNAL_SERVER_ERROR, # 500
                ],
                allowed_methods=["GET", "POST"],
                raise_on_status=False,
                connect=self.maxretry,
                read=self.maxretry,
                redirect=self.maxretry,
            )
            adapter = HTTPAdapter(
                max_retries=retries,
                pool_maxsize=self.POOL_SIZE,  # Double the pool size
                pool_connections=self.POOL_SIZE,
                pool_block=True
            )
            session.mount("https://", adapter)

            # Set longer timeouts
            session.timeout = (30, 600)  # (connect timeout, read timeout)
            
            self._thread_local.session = session
            with self._lock:
                self._sessions.append(session)
        try:
            yield session
        finally:
            pass  # Don't close here - keep session alive for thread reuse

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
                raise RuntimeError(f"\n{request_string=}\n\n{response.status_code=}\n\n{response.text}\n\n")
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

    @classmethod
    def _geometry_to_coordstring(cls, geometry: Union[Polygon, MultiPolygon]) -> Tuple[str, bool]:
        """
        Turn a shapely polygon or multipolygon into a single coord string to be used in API requests.
        To meet the constraints on the API URL the following changes may be applied:
         - Multipolygons are simplified to single polygons by taking the convex hull
         - Polygons that have too many coordinates (resulting in strings that are too long) are
           simplified by taking the convex hull.
         - Polygons that have a convex hull with too many coordinates are simplified to a box.
        If the coord string return does not represent the polygon exactly, the exact flag is set to False.
        """
        convex_hull = None
        if isinstance(geometry, (GeometryCollection)):
            logger.debug(f"Geometry is a collection - extracting polygons. {geometry=}")
            # Extract all polygons and multipolygons from collection
            polygons = []
            for geom in geometry.geoms:
                if isinstance(geom, (Polygon, MultiPolygon)):
                    polygons.append(geom)
            if len(polygons) == 0:
                raise ValueError("No valid polygons found in GeometryCollection")
            # Combine into single multipolygon
            geometry = MultiPolygon(polygons)
            exact = False

        if isinstance(geometry, MultiPolygon):
            if len(geometry.geoms) == 1:
                g = geometry.geoms[0]

                if len(g.interiors) > 0:
                    logger.debug(f"Geometry has inner rings - approximating query with convex hull.")
                    coordstring = cls._polygon_to_coordstring(g.convex_hull)
                    exact = False
                else:
                    coordstring = cls._polygon_to_coordstring(g)
                    exact = True
            else:
                raise ValueError("Must not be called with multipolygon - separate parts must be iterated externally.")
        else:
            # Tests whether the polygon has inner rings/holes.
            if len(geometry.interiors) > 0:
                logger.debug(f"Geometry has inner rings - approximating query with convex hull.")
                convex_hull = geometry.convex_hull
                coordstring = cls._polygon_to_coordstring(convex_hull)
                exact = False
            else:
                coordstring = cls._polygon_to_coordstring(geometry)
                exact = True
        if len(coordstring) > cls.CHAR_LIMIT:
            logger.debug(f"Geometry exceeds character limit - approximating query with convex hull.")
            exact = False
            if convex_hull is None:
                convex_hull = geometry.convex_hull
            coordstring = cls._polygon_to_coordstring(convex_hull)
            if len(coordstring) > cls.CHAR_LIMIT:
                exact = False
                coordstring = cls._polygon_to_coordstring(geometry.envelope)

        return coordstring, exact

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
        temp_path = self.cache_dir / f"{str(uuid.uuid4())}.tmp"
        try:
            if self.compress_cache:
                temp_path = temp_path.parent / f"{temp_path.name}.gz"
                with gzip.open(temp_path, "w") as f:
                    payload_bytes = json.dumps(payload).encode("utf-8")
                    f.write(payload_bytes)
                    f.flush()
                    os.fsync(f.fileno())
            else:
                with open(temp_path, "w") as f:
                    json.dump(payload, f)
                    f.flush()
                    os.fsync(f.fileno())
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _create_request_string(
        self,
        base_url: str,
        geometry: Union[Polygon, MultiPolygon],
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        address_fields: Optional[Dict[str, str]] = None,
        survey_resource_id: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """
        Create a request string with given parameters for GET requests
        base_url: Need to choose one of: self.FEATURES_URL, self.ROLLUPS_CSV_URL
        
        Note: This method is maintained for backward compatibility and for cases where GET requests are still needed.
        For most cases, consider using _create_post_request instead, which handles complex geometries better.
        """
        urlbase = base_url
        if survey_resource_id is not None:
            urlbase = f"{self.FEATURES_SURVEY_RESOURCE_URL}/{survey_resource_id}/features.json"
        if geometry is not None:
            coordstring, exact = self._geometry_to_coordstring(geometry)
            request_string = f"{urlbase}?polygon={coordstring}&apikey={self.api_key}"
        else:
            exact = True  # we treat address-based as exact always
            address_params = "&".join([f"{s}={address_fields[s]}" for s in address_fields])
            request_string = f"{urlbase}?{address_params}&apikey={self.api_key}"

        # Add dates if given
        if ((since is not None) or (until is not None)) and (survey_resource_id is not None):
            logger.debug(
                f"Request made with survey_resource_id {survey_resource_id} and either since or until - ignoring dates."
            )
        elif (since is not None) or (until is not None):
            if since and not isinstance(since, str):
                raise ValueError("Since must be a string")
            if until and not isinstance(until, str):
                raise ValueError("Until must be a string")
            if since:
                request_string += f"&since={since}"
            if until:
                request_string += f"&until={until}"

        if self.alpha:
            request_string += "&alpha=true"
        if self.beta:
            request_string += "&beta=true"
        if self.prerelease:
            request_string += "&prerelease=true"
        if self.only3d:
            request_string += "&3dCoverage=true"
        if self.parcelMode:
            request_string += "&parcelMode=true"
        if self.system_version_prefix is not None:
            request_string += f"&systemVersionPrefix={self.system_version_prefix}"
        if self.system_version is not None:
            request_string += f"&systemVersion={self.system_version}"
        # Add packs if given
        if packs:
            if isinstance(packs, list):
                packs = ",".join(packs)
            request_string += f"&packs={packs}"
        if classes:
            if isinstance(classes, list):
                classes = ",".join(classes)
            request_string += f"&classes={classes}"
        return request_string, exact
        
    def _create_post_request(
        self,
        base_url: str,
        geometry: Union[Polygon, MultiPolygon],
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
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
        if self.parcelMode:
            url += "&parcelMode=true"
        if self.system_version_prefix is not None:
            url += f"&systemVersionPrefix={self.system_version_prefix}"
        if self.system_version is not None:
            url += f"&systemVersion={self.system_version}"
        if self.bulk_mode:
            url += "&bulk=true"

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

        # With POST requests, we always get the exact geometry processed
        exact = True

        return url, body, exact

    def get_features(
        self,
        geometry: Union[Polygon, MultiPolygon],
        region: str,
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        address_fields: Optional[Dict[str, str]] = None,
        survey_resource_id: Optional[str] = None,
        use_post: bool = True,  # Default to using POST for better geometry handling
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
            use_post: Whether to use POST request (better for complex geometries) or GET
        
        Returns:
            API response as a dictionary
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
            result_type=self.API_TYPE_FEATURES,
            use_post=use_post,
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
        use_post: bool = True,  # Default to using POST, but API might not support it yet
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
            use_post: Whether to use POST request (better for complex geometries) or GET
        
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
            use_post=use_post,
        )
        return data

    def _get_results(
        self,
        geometry: Union[Polygon, MultiPolygon],
        region: str,
        packs: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        address_fields: Optional[Dict[str, str]] = None,
        survey_resource_id: Optional[str] = None,
        result_type: str = API_TYPE_FEATURES,
        use_post: bool = True,  # Default to using POST for all new requests
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
            use_post: Whether to use POST request (recommended) or GET request

        Returns:
            API response as a Dictionary
        """
        with self._session_scope() as session:
            # Determine the base URL based on the result type and packs
            if result_type == self.API_TYPE_FEATURES:
                base_url = self.FEATURES_URL
                if packs is not None:
                    if "damage" in packs or "damage_non_postcat" in packs:
                        base_url = self.FEATURES_DAMAGE_URL
            elif result_type == self.API_TYPE_ROLLUPS:
                base_url = self.ROLLUPS_CSV_URL
                
            # For address-based queries, always use GET (POST doesn't support address fields)
            # For rollups, we might need to stick with GET until POST is fully supported in API
            use_get = (address_fields is not None) or (not use_post)
            
            if use_get:
                # Use the legacy GET request approach
                request_string, exact = self._create_request_string(
                    base_url=base_url,
                    geometry=geometry,
                    packs=packs,
                    classes=classes,
                    since=since,
                    until=until,
                    address_fields=address_fields,
                    survey_resource_id=survey_resource_id,
                )
                cache_path = None if self.cache_dir is None else self._request_cache_path(request_string)
                
                if not exact and result_type == self.API_TYPE_ROLLUPS:
                    raise AIFeatureAPIError(
                        response=None,
                        request_string=self._clean_api_key(request_string),
                        text="MultiPolygons and inexact polygons not supported by rollup endpoint.",
                        message="MultiPolygons and inexact polygons not supported by rollup endpoint.",
                    )
            else:
                # Use POST request with JSON body for better geometry handling
                url, body, exact = self._create_post_request(
                    base_url=base_url,
                    geometry=geometry,
                    packs=packs,
                    classes=classes,
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

            # Request data
            t1 = time.monotonic()
            try:
                if use_get:
                    response = session.get(request_string, timeout=TIMEOUT_SECONDS)
                    request_info = request_string  # For error reporting
                else:
                    headers = {'Content-Type': 'application/json'}
                    response = session.post(url, json=body, headers=headers, timeout=TIMEOUT_SECONDS)
                    request_info = f"{url} with body {json.dumps(body)}"  # For error reporting
            except requests.exceptions.ChunkedEncodingError:
                logger.info(f"ChunkedEncodingError for request")
                self._handle_response_errors(None, request_info if use_get else url)

            response_time_ms = (time.monotonic() - t1) * 1e3
            logger.debug(f"{response_time_ms:.1f}ms response time")

            if response.ok:
                if result_type == self.API_TYPE_ROLLUPS:
                    data = response.text
                elif result_type == self.API_TYPE_FEATURES:
                    try:
                        data = response.json()
                    except Exception as e:
                        logging.warning(f"Error parsing JSON response from API: {e}")
                        raise AIFeatureAPIError(response, request_info if use_get else url)
                        
                    # With POST requests, exact is always true, so we don't need this filtering
                    # If the AOI was altered for the API request (GET only), we need to filter features
                    if use_get and not exact:
                        # Filter out any features that are not within the geometry
                        data_features_geoms = gpd.GeoSeries(
                            [shape(f["geometry"]) for f in data["features"]], crs=API_CRS, name="geometry"
                        )
                        keep_inds = data_features_geoms[data_features_geoms.intersects(geometry)].index
                        data["features"] = [data["features"][i] for i in keep_inds]
                        if len(data["features"]) > 0:
                            gdf_unclipped = data_features_geoms[keep_inds]
                            gdf_unclipped.index = range(len(keep_inds))

                            gdf_clip = self._clip_features_to_polygon(gdf_unclipped, geometry, region)

                            for i, feature in enumerate(data["features"]):
                                data["features"][i]["clippedAreaSqm"] = gdf_clip.loc[i, "clipped_area_sqm"]
                                data["features"][i]["clippedAreaSqft"] = gdf_clip.loc[i, "clipped_area_sqft"]

                                if feature["classId"] in CONNECTED_CLASS_IDS:
                                    # Replace geometry, with geojson style mapped clipped geometry
                                    data["features"][i]["geometry"] = shapely.geometry.mapping(
                                        gdf_clip.loc[i, "geometry"]
                                    )
                                    data["features"][i]["areaSqm"] = gdf_clip.loc[i, "clipped_area_sqm"]
                                    data["features"][i]["areaSqft"] = gdf_clip.loc[i, "clipped_area_sqft"]

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
                    raise AIFeatureAPIRequestSizeError(response, request_info if use_get else url)
                elif status_code == HTTPStatus.BAD_REQUEST:
                    try:
                        error_data = json.loads(text)
                        error_code = error_data.get("code", "")
                        if error_code in AIFeatureAPIRequestSizeError.codes:
                            logger.debug(f"Raising AIFeatureAPIRequestSizeError from secondary status code {status_code=}")
                            raise AIFeatureAPIRequestSizeError(response, request_info if use_get else url)
                    except (json.JSONDecodeError, KeyError):
                        pass
                    
                    # Check for other errors
                    self._handle_response_errors(response, request_info if use_get else url)
                else:
                    # Handle other HTTP errors
                    self._handle_response_errors(response, request_info if use_get else url)

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

        features_gdf_dissolved = (
            features_gdf.drop_duplicates(
                ["feature_id", "geometry"]
            )  # First, drop duplicate geometries rather than dissolving them together.
            .filter(agg_cols_first + ["geometry", "feature_id"], axis=1)
            .dissolve(
                by="feature_id", aggfunc="first"
            )  # Then dissolve any remaining features that represent a single feature_id that has been split.
            .reset_index()
            .set_index("feature_id")
        )

        features_gdf_summed = (
            features_gdf.filter(agg_cols_sum + ["feature_id"], axis=1)
            .groupby("feature_id")
            .aggregate(dict([c, "sum"] for c in agg_cols_sum))
        )

        # final output - same format, same set of feature_ids, but fewer rows due to dedup and merging.
        # Set the index back to being the AOI_ID column
        features_gdf_out = features_gdf_dissolved.join(features_gdf_summed).reset_index().set_index(AOI_ID_COLUMN_NAME)

        return features_gdf_out

    @classmethod
    def trim_features_to_aoi(
        cls, gdf_features: gpd.GeoDataFrame, geometry: Union[Polygon, MultiPolygon], region: str
    ) -> gpd.GeoDataFrame:
        """
        Trim all features in dataframe by performing intersection with the correct query AOI. Fix attributes like
        clipped areas, and remove features that no longer intersect.

        gdf_features: The dataframe of features, as returned by  FeatureAPI.payload_gdf.
        geometry: The polygon for the masking Query AOI.
        region: Country code.
        :return: Filtered and clipped GeoDataFrame in same format as the input gdf_features.
        """
        # Remove all features that don't intersect at all.
        gdf_features = (
            gdf_features[gdf_features.intersects(geometry)]
            .drop_duplicates(subset=["feature_id"])
            .reset_index()
            .set_index("feature_id")
        )
        gdf_clip = cls._clip_features_to_polygon(gdf_features.geometry, geometry, region)

        # TODO: Don't know why this has to be done, but I get a ValueError about len of keys and values being identical if I don't ...
        geom_column = gdf_features.geometry

        for feature_id, f in gdf_features.iterrows():
            gdf_features.loc[feature_id, "clipped_area_sqm"] = gdf_clip.loc[feature_id, "clipped_area_sqm"]
            gdf_features.loc[feature_id, "clipped_area_sqft"] = gdf_clip.loc[feature_id, "clipped_area_sqft"]

            class_id = gdf_features.loc[feature_id, "class_id"]
            if class_id in CONNECTED_CLASS_IDS:
                # Replace geometry, with clipped geometry
                geom_column[feature_id] = gdf_clip.loc[feature_id, "geometry"]
                gdf_features.loc[feature_id, "area_sqm"] = gdf_clip.loc[feature_id, "clipped_area_sqm"]
                gdf_features.loc[feature_id, "area_sqft"] = gdf_clip.loc[feature_id, "clipped_area_sqft"]
        gdf_features["geometry"] = geom_column
        return gdf_features.reset_index().set_index(AOI_ID_COLUMN_NAME)

    @classmethod
    def payload_gdf(cls, payload: dict, aoi_id: Optional = None, parcelMode: Optional[bool] = False) -> Tuple[gpd.GeoDataFrame, dict]:
        """
        Create a GeoDataFrame from a feature API response dictionary.

        Args:
            payload: API response dictionary
            aoi_id: Optional ID for the AOI to add to the data
            parcelMode: If True, filter out features where belongsToParcel is False

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
            # Filter features if parcelMode is enabled
            features = payload["features"]
            if parcelMode and len(features) > 0 and "belongsToParcel" in features[0]:
                features = [f for f in features if f.get("belongsToParcel", True)]

            # Check for and use clippedGeometry if present
            # Add a new flag for multiparcel features (those with clippedGeometry)
            for feature in features:
                # Default is False - not a multiparcel feature
                feature["multiparcelFeature"] = False

                # If clippedGeometry exists, use it and mark as multiparcel feature
                if "clippedGeometry" in feature and feature["clippedGeometry"]:
                    feature["geometry"] = feature["clippedGeometry"]
                    feature["multiparcelFeature"] = True

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

    def get_features_gdf(
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
        fail_hard_regrid: Optional[bool] = False,
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
        try:
            if isinstance(geometry, MultiPolygon) and len(geometry.geoms) > 1:
                # A proper multi-polygon - run it as separate requests, then recombine.
                features_gdf, metadata, error = [], [], None
                for sub_geometry in geometry.geoms:
                    sub_payload = self.get_features(
                        sub_geometry, region, packs, classes, since, until, address_fields, survey_resource_id, self.use_post
                    )
                    sub_features_gdf, sub_metadata = self.payload_gdf(sub_payload, aoi_id, self.parcelMode)
                    features_gdf.append(sub_features_gdf)
                    metadata.append(sub_metadata)
                # Warning - using arbitrary int index means duplicate index.
                features_gdf = pd.concat(features_gdf) if len(features_gdf) > 0 else None

                # Check for repeat appearances of the same feature in the multipolygon
                if len(features_gdf.feature_id.unique()) < len(features_gdf):
                    features_gdf = self.trim_features_to_aoi(features_gdf, geometry, region)

                # Deduplicate metadata, picking from the first part of the multipolygon rather than attempting to merge
                metadata_df = pd.DataFrame(metadata).drop(columns=["link", AOI_ID_COLUMN_NAME])
                metadata_df = metadata_df.drop_duplicates()
                if len(metadata_df) > 1:
                    raise AIFeatureAPIError(
                        response=None,
                        request_string=None,
                        text="MultiPolygon Match Failure",
                        message="Mismatching dates or system versions",
                    )
                else:
                    metadata = metadata[0]
            else:
                features_gdf, metadata, error = None, None, None
                payload = self.get_features(
                    geometry, region, packs, classes, since, until, address_fields, survey_resource_id, self.use_post
                )
                features_gdf, metadata = self.payload_gdf(payload, aoi_id, self.parcelMode)
        except AIFeatureAPIRequestSizeError as e:
            features_gdf, metadata, error = None, None, None

            # If the query was too big, split it up into a grid, and recombine as though it was one query.
            # Do not get stuck in an infinite loop of re-gridding and timing out
            if fail_hard_regrid or geometry is None:
                logger.debug("Failing hard and NOT re-gridding....")
                error = {
                    AOI_ID_COLUMN_NAME: aoi_id,
                    "status_code": e.status_code,
                    "message": e.message,
                    "text": e.text,
                    "request": e.request_string,
                }
            else:
                # First request was too big, so grid it up, recombine, and return. Any problems and the whole AOI should return an error as usual.
                logger.debug(f"Found an over-sized AOI (id {aoi_id}). Trying gridding...")
                try:
                    features_gdf, metadata_df, errors_df = self.get_features_gdf_gridded(
                        geometry,
                        region,
                        packs,
                        classes,
                        aoi_id,
                        since,
                        until,
                        survey_resource_id,
                        aoi_grid_inexact=self.aoi_grid_inexact,
                    )
                    error = None  # Reset error if we got here without an exception

                    # Recombine gridded features
                    features_gdf = FeatureApi.combine_features_gdf_from_grid(features_gdf)

                    # Creat metadata
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

                except (AIFeatureAPIError, AIFeatureAPIGridError) as e:
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
                request_string = self._create_request_string(
                    base_url=self.FEATURES_URL,
                    geometry=geometry,
                    packs=packs,
                    classes=classes,
                    since=since,
                    until=until,
                    address_fields=address_fields,
                    survey_resource_id=survey_resource_id,
                )[0]
                logger.debug(f"Probable original request string was: {request_string}")
            features_gdf = None
            metadata = None
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": status_code,
                "message": "RETRY_ERROR",
                "text": "Retry Error",
                "request": str(geometry),
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
                "request": str(geometry),
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
        aoi_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        survey_resource_id: Optional[str] = None,
        aoi_grid_inexact: Optional[bool] = False,
        grid_size: Optional[float] = 0.005,  # Approx 500m at the equator
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
        df_gridded = FeatureApi.split_geometry_into_grid(geometry=geometry, cell_size=grid_size)
        # TODO: At this point, we should hit coverage to check that each of the grid squares falls within the AOI. That way we can fail fast before retrieving any payloads. Currently pulls every possible payload before failing.

        # Retrieve the features for every one of the cells in the gridded AOIs
        aoi_id_tmp = range(len(df_gridded))
        df_gridded[AOI_ID_COLUMN_NAME] = aoi_id_tmp

        try:
            # If we are already in a 'gridded' call, then when we call get_features_gdf_bulk
            # we need to pass in fail_hard_regrid=True so we don't get stuck in an endless loop
            features_gdf, metadata_df, errors_df = self.get_features_gdf_bulk(
                df_gridded,
                region=region,
                packs=packs,
                classes=classes,
                since_bulk=since,
                until_bulk=until,
                survey_resource_id_bulk=survey_resource_id,
                max_allowed_error_pct=100 - self.aoi_grid_min_pct,
                fail_hard_regrid=True,
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
        since_bulk: Optional[str] = None,
        until_bulk: Optional[str] = None,
        survey_resource_id_bulk: Optional[str] = None,
        max_allowed_error_pct: Optional[int] = 100,
        fail_hard_regrid: Optional[bool] = False,
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
        with concurrent.futures.ThreadPoolExecutor(self.threads) as executor:
            try:
                max_allowed_error_count = round(len(gdf) * max_allowed_error_pct / 100)

                # are address fields present?
                has_address_fields = set(gdf.columns.tolist()).intersection(set(ADDRESS_FIELDS)) == set(ADDRESS_FIELDS)
                # is a geometry field present?
                has_geom = "geometry" in gdf.columns

                # Run in thread pool
                with concurrent.futures.ThreadPoolExecutor(self.threads) as executor:
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
                                row.geometry if has_geom else None,
                                region,
                                packs,
                                classes,
                                aoi_id,
                                since,
                                until,
                                {f: row[f] for f in ADDRESS_FIELDS} if has_address_fields else None,
                                survey_resource_id,
                                fail_hard_regrid,
                            )
                        )
                    data = []
                    metadata = []
                    errors = []
                    for job in jobs:
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
                executor.shutdown(wait=True)  # Ensure executor shuts down
                self.cleanup()  # Clean up sessions after bulk operation

        features_gdf = pd.concat([df for df in data if len(df) > 0]) if len(data) > 0 else gpd.GeoDataFrame([])
        metadata_df = pd.DataFrame(metadata).set_index(AOI_ID_COLUMN_NAME) if len(metadata) > 0 else pd.DataFrame([])
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
            if isinstance(geometry, MultiPolygon) and len(geometry.geoms) > 1:
                # A proper multi-polygon - run it as separate requests, then recombine.
                rollup_df, metadata, error = [], [], None
                for sub_geometry in geometry.geoms:
                    sub_payload = self.get_rollup(
                        sub_geometry, region, packs, classes, since, until, address_fields, survey_resource_id
                    )
                    sub_rollup_df, sub_metadata = self.payload_rollup_df(sub_payload, aoi_id)
                    rollup_df.append(sub_rollup_df)
                    metadata.append(sub_metadata)
                rollup_df = pd.concat(rollup_df) if len(rollup_df) > 0 else None
                # Warning - using arbitrary int index means duplicate index.

                # Deduplicate metadata, picking from the first part of the multipolygon rather than attempting to merge
                metadata_df = pd.DataFrame(metadata).drop(columns=["link", AOI_ID_COLUMN_NAME])
                metadata_df = metadata_df.drop_duplicates()
                if len(metadata_df) > 1:
                    raise AIFeatureAPIError(
                        response=None,
                        request_string=None,
                        text="MultiPolygon Match Failure",
                        message="Mismatching dates or system versions",
                    )
                else:
                    metadata = metadata[0]
            else:
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
                "request": "",
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
                for job in jobs:
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
