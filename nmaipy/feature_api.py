import logging

import concurrent.futures
import hashlib
from http import HTTPStatus
import json
import os
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Tuple, Union
import uuid
from io import StringIO

import gzip
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from shapely.geometry import MultiPolygon, Polygon, shape
import shapely.geometry
import stringcase

from nmaipy import log
from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    LAT_LONG_CRS,
    SINCE_COL_NAME,
    UNTIL_COL_NAME,
    SURVEY_RESOURCE_ID_COL_NAME,
    MAX_RETRIES,
    AOI_EXCEEDS_MAX_SIZE,
    ADDRESS_FIELDS,
    SQUARED_METERS_TO_SQUARED_FEET,
    AREA_CRS,
    API_CRS,
    CONNECTED_CLASS_IDS,
    ROLLUP_SURVEY_DATE_ID,
    ROLLUP_SYSTEM_VERSION_ID,
)

logger = log.get_logger()


class RetryRequest(Retry):
    """
    Inherited retry request to limit back-off to 1 second.
    """

    BACKOFF_MAX = 1


class AIFeatureAPIError(Exception):
    """
    Error responses for logging from AI Feature API. Also include non rest API errors (use dummy status code and
    explicitly set messages).
    """

    DUMMY_STATUS_CODE = -1

    def __init__(self, response, request_string, text="Query Not Attempted", message="Error with Query AOI"):
        if response is None:
            self.status_code = self.DUMMY_STATUS_CODE
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


class FeatureApi:
    """
    Class to connect to the AI Feature API
    """

    FEATURES_URL = "https://api.nearmap.com/ai/features/v4/features.json"
    ROLLUPS_CSV_URL = "https://api.nearmap.com/ai/features/v4/rollups.csv"
    FEATURES_SURVEY_RESOURCE_URL = "https://api.nearmap.com/ai/features/v4/surveyresources"
    CLASSES_URL = "https://api.nearmap.com/ai/features/v4/classes.json"
    PACKS_URL = "https://api.nearmap.com/ai/features/v4/packs.json"
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
        api_key: Optional[str] = None,
        bulk_mode: Optional[bool] = True,
        cache_dir: Optional[Path] = None,
        overwrite_cache: Optional[bool] = False,
        compress_cache: Optional[bool] = False,
        workers: Optional[int] = 10,
        alpha: Optional[bool] = False,
        beta: Optional[bool] = False,
        maxretry: int = MAX_RETRIES,
    ):
        """
        Initialize FeatureApi class

        Args:
            api_key: Nearmap API key. If not defined the environment variable will be used
            cache_dir: Directory to use as a payload cache
            overwrite_cache: Set to overwrite values stored in the cache
            compress_cache: Whether to use gzip compression (.json.gz) or save raw json text (.json).
            workers: Number of threads to spawn for concurrent execution
        """
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
        self.overwrite_cache = overwrite_cache
        self.compress_cache = compress_cache
        self.workers = workers
        self.bulk_mode = bulk_mode
        self.alpha = alpha
        self.beta = beta
        self.maxretry = maxretry

    @property
    def _session(self) -> requests.Session:
        """
        Return a request session with retrying configured.
        """
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            retries = RetryRequest(
                total=self.maxretry,
                backoff_factor=0.05,
                status_forcelist=[
                    HTTPStatus.TOO_MANY_REQUESTS,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    HTTPStatus.BAD_GATEWAY,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                ],
            )
            session.mount(
                "https://",
                HTTPAdapter(max_retries=retries, pool_maxsize=self.POOL_SIZE, pool_connections=self.POOL_SIZE),
            )
            self._thread_local.session = session
            self._sessions.append(session)
        return session

    def _get_feature_api_results_as_data(self, base_url: str) -> Tuple[requests.Response, Dict]:
        """
        Return a result from one of the base URLS.
        Args:
            base_url: self.PACKS_URL,

        Returns:

        """
        request_string = f"{base_url}?apikey={self.api_key}"
        if self.alpha:
            request_string += "&alpha=true"
        if self.beta:
            request_string += "&beta=true"
        # Request data
        response = self._session.get(request_string)
        # Check for errors
        if not response.ok:
            # Fail hard for unexpected errors
            raise RuntimeError(f"\n{request_string=}\n\n{response.status_code=}\n\n{response.text}\n\n")
        data = response.json()
        return response, data

    def get_packs(self) -> Dict[str, List[str]]:
        """
        Get packs with class IDs
        """
        t1 = time.monotonic()
        response, data = self._get_feature_api_results_as_data(self.PACKS_URL)
        response_time_ms = (time.monotonic() - t1) * 1e3
        logger.debug(f"{response_time_ms:.1f}ms response time for packs.json")
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

        return df_classes

    @staticmethod
    def _polygon_to_coordstring(poly: Polygon) -> str:
        """
        Turn a shapely polygon into the format required by the API for a query polygon.
        """
        coords = poly.exterior.coords[:]
        flat_coords = np.array(coords).flatten()
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
                coordstring = cls._polygon_to_coordstring(geometry.convex_hull)
                exact = False
            else:
                coordstring = cls._polygon_to_coordstring(geometry)
                exact = True
        if len(coordstring) > cls.CHAR_LIMIT:
            logger.debug(f"Geometry exceeds character limit - approximating query with convex hull.")
            exact = False
            coordstring = cls._polygon_to_coordstring(geometry.convex_hull)
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

    def _request_cache_path(self, request_string: str) -> Path:
        """
        Hash a request string to create a cache path.
        """
        request_string = request_string.replace(self.api_key, "")
        request_hash = hashlib.md5(request_string.encode()).hexdigest()
        lon, lat = self._make_latlon_path_for_cache(request_string)
        ext = "json.gz" if self.compress_cache else "json"
        return self.cache_dir / lon / lat / f"{request_hash}.{ext}"

    def _request_error_message(self, request_string: str, response: requests.Response) -> str:
        """
        Create a descriptive error message without the API key.
        """
        return f"\n{request_string.replace(self.api_key, '...')=}\n\n{response.status_code=}\n\n{response.text}\n\n"

    def _handle_response_errors(self, response: requests.Response, request_string: str):
        """
        Handle errors returned from the feature API
        """
        clean_request_string = request_string.replace(self.api_key, "...")
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
        since: Optional[str] = None,
        until: Optional[str] = None,
        address_fields: Optional[Dict[str, str]] = None,
        survey_resource_id: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """
        Create a request string with given parameters
        base_url: Need to choose one of: self.FEATURES_URL, self.ROLLUPS_CSV_URL
        """
        urlbase = (
            base_url
            if survey_resource_id is None
            else f"{self.FEATURES_SURVEY_RESOURCE_URL}/{survey_resource_id}/features.json"
        )
        bulk_str = str(self.bulk_mode).lower()
        if geometry is not None:
            coordstring, exact = self._geometry_to_coordstring(geometry)
            request_string = f"{urlbase}?polygon={coordstring}&bulk={bulk_str}&apikey={self.api_key}"
        else:
            exact = True  # we treat address-based as exact always
            address_params = "&".join([f"{s}={address_fields[s]}" for s in address_fields])
            request_string = f"{urlbase}?{address_params}&bulk={bulk_str}&apikey={self.api_key}"

        # Add dates if given
        if ((since is not None) or (until is not None)) and (survey_resource_id is not None):
            raise ValueError("Invalid combination of since, until and survey_resource_id requested")
        elif (since is not None) or (until is not None):
            if since:
                request_string += f"&since={since}"
            if until:
                request_string += f"&until={until}"

        if self.alpha:
            request_string += "&alpha=true"
        if self.beta:
            request_string += "&beta=true"

        # Add packs if given
        if packs:
            if isinstance(packs, list):
                packs = ",".join(packs)
            request_string += f"&packs={packs}"

        return request_string, exact

    def get_features(
        self,
        geometry: Union[Polygon, MultiPolygon],
        region: str,
        packs: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        address_fields: Optional[Dict[str, str]] = None,
        survey_resource_id: Optional[str] = None,
    ):
        data = self._get_results(
            geometry=geometry,
            region=region,
            packs=packs,
            since=since,
            until=until,
            address_fields=address_fields,
            survey_resource_id=survey_resource_id,
            result_type=self.API_TYPE_FEATURES,
        )
        return data

    def get_rollup(
        self,
        geometry: Union[Polygon, MultiPolygon],
        region: str,
        packs: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        address_fields: Optional[Dict[str, str]] = None,
        survey_resource_id: Optional[str] = None,
    ):
        data = self._get_results(
            geometry=geometry,
            region=region,
            packs=packs,
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
        since: Optional[str] = None,
        until: Optional[str] = None,
        address_fields: Optional[Dict[str, str]] = None,
        survey_resource_id: Optional[str] = None,
        result_type: str = API_TYPE_FEATURES,
    ):
        """
        Get feature data for an AOI. If a cache is configured, the cache will be checked before using the API.

        Args: geometry: AOI in EPSG4326 region: Country code, used for recalculating areas. packs: List of AI packs
        since: Earliest date to pull data for until: Latest date to pull data for address_fields: Fields for an
        address based query (rather than query AOI based query). survey_resource_id: The ID of the survey resource id
        if an exact survey is requested for the pull. NB: This is NOT the survey ID from coverage - it is the id of
        the AI resource attached to that survey.

        Returns:
            API response as a Dictionary
        """

        # Create request string
        if result_type == self.API_TYPE_FEATURES:
            base_url = self.FEATURES_URL
        elif result_type == self.API_TYPE_ROLLUPS:
            base_url = self.ROLLUPS_CSV_URL
        request_string, exact = self._create_request_string(
            base_url=base_url,
            geometry=geometry,
            packs=packs,
            since=since,
            until=until,
            address_fields=address_fields,
            survey_resource_id=survey_resource_id,
        )
        if not exact and result_type == self.API_TYPE_ROLLUPS:
            raise AIFeatureAPIError(
                response=None,
                request_string=request_string.replace(self.api_key, "..."),
                text="MultiPolygons and inexact polygons not supported by rollup endpoint.",
                message="MultiPolygons and inexact polygons not supported by rollup endpoint.",
            )
        logger.debug(f"Requesting: {request_string.replace(self.api_key, '...')}")
        cache_path = None if self.cache_dir is None else self._request_cache_path(request_string)

        # Check if it's already cached
        if self.cache_dir is not None and not self.overwrite_cache:
            if cache_path.exists():
                logger.debug(f"Retrieving payload from cache")
                if self.compress_cache:
                    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                        try:
                            payload_str = f.read()
                            return json.loads(payload_str)
                        except EOFError as e:
                            logging.error(f"Error loading compressed cache file {cache_path}.")
                            logging.error(payload_str)
                else:
                    with open(cache_path, "r") as f:
                        return json.load(f)

        # Request data
        t1 = time.monotonic()
        try:
            response = self._session.get(request_string)
        except requests.exceptions.ChunkedEncodingError:
            self._handle_response_errors(None, request_string)

        response_time_ms = (time.monotonic() - t1) * 1e3

        if response.ok:
            logger.debug(f"{response_time_ms:.1f}ms response time for polygon with these packs: {packs}")

            if result_type == self.API_TYPE_ROLLUPS:
                data = response.text
            elif result_type == self.API_TYPE_FEATURES:
                data = response.json()
                # If the AOI was altered for the API request, we need to filter features in the response, and clip connected features
                if not exact:
                    # Filter out any features that are not a candidate (e.g. a polygon with a central hole).
                    data["features"] = [f for f in data["features"] if shape(f["geometry"]).intersects(geometry)]
                    if len(data["features"]) > 0:
                        gdf_unclipped = gpd.GeoSeries(pd.DataFrame(data["features"]).geometry.apply(shape), crs=API_CRS)

                        gdf_clip = self._clip_features_to_polygon(gdf_unclipped, geometry, region)

                        for i, feature in enumerate(data["features"]):
                            data["features"][i]["clippedAreaSqm"] = gdf_clip.loc[i, "clipped_area_sqm"]
                            data["features"][i]["clippedAreaSqft"] = gdf_clip.loc[i, "clipped_area_sqft"]

                            if feature["classId"] in CONNECTED_CLASS_IDS:
                                # Replace geometry, with geojson style mapped clipped geometry
                                data["features"][i]["geometry"] = shapely.geometry.mapping(gdf_clip.loc[i, "geometry"])
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

            logger.debug(f"{response_time_ms:.1f}ms failure response time {text} {status_code}")
            if self.overwrite_cache:
                # Explicitly clean up request by deleting cache file, perhaps the request worked previously. Not out of spite, but to prevent confusing cases in future.
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

            if status_code in AIFeatureAPIRequestSizeError.status_codes:
                logger.debug(f"Raising AIFeatureAPIRequestSizeError from status code {status_code=}")
                raise AIFeatureAPIRequestSizeError(response, request_string)
            elif status_code == HTTPStatus.BAD_REQUEST:
                error_code = json.loads(text)["code"]
                if error_code in AIFeatureAPIRequestSizeError.codes:
                    logger.debug(f"Raising AIFeatureAPIRequestSizeError from secondary status code {status_code=}")
                    raise AIFeatureAPIRequestSizeError(response, request_string)
                else:
                    # Check for errors
                    self._handle_response_errors(response, request_string)
            else:
                # Check for errors
                self._handle_response_errors(response, request_string)

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
        if not location_marker_string not in link:
            return link.append(location_marker_string)
        else:
            return link

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
            "aoi_id",
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
        features_gdf_out = features_gdf_dissolved.join(features_gdf_summed).reset_index()

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
        return gdf_features.reset_index()

    @classmethod
    def payload_gdf(cls, payload: dict, aoi_id: Optional[str] = None) -> Tuple[gpd.GeoDataFrame, dict]:
        """
        Create a GeoDataFrame from a feature API response dictionary.

        Args:
            payload: API response dictionary
            aoi_id: Optional ID for the AOI to add to the data

        Returns:
            Features GeoDataFrame
            Metadata dictionary
        """

        # Create metadata
        metadata = {
            "system_version": payload["systemVersion"],
            "link": cls.add_location_marker_to_link(payload["link"]),
            "date": cls.link_to_date(payload["link"]),
        }

        columns = [
            "id",
            "classId",
            "description",
            "confidence",
            "fidelity",  # not on every class
            "parentId",
            "geometry",
            "areaSqm",
            "clippedAreaSqm",
            "unclippedAreaSqm",
            "areaSqft",
            "clippedAreaSqft",
            "unclippedAreaSqft",
            "attributes",
            "surveyDate",
            "meshDate",
        ]

        # Create features DataFrame
        if len(payload["features"]) == 0:
            df = pd.DataFrame([], columns=columns)
        else:
            df = pd.DataFrame(payload["features"])
            for col_name in set(columns).difference(set(df.columns)):
                df[col_name] = None

        for col in FeatureApi.FLOAT_COLS:
            if col in df:
                df[col] = df[col].astype("float")

        df = df.rename(columns={"id": "feature_id"})
        df.columns = [stringcase.snakecase(c) for c in df.columns]

        # Add AOI ID if specified
        if aoi_id is not None:
            try:
                df[AOI_ID_COLUMN_NAME] = aoi_id
            except Exception as e:
                logger.error(
                    f"Problem setting aoi_id in col {AOI_ID_COLUMN_NAME} as {aoi_id} (dataframe has {len(df)} rows)."
                )
                raise ValueError
            metadata[AOI_ID_COLUMN_NAME] = aoi_id
        # Cast to GeoDataFrame
        if "geometry" in df.columns:
            gdf = gpd.GeoDataFrame(df.assign(geometry=df.geometry.apply(shape)))
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
                        sub_geometry, region, packs, since, until, address_fields, survey_resource_id
                    )
                    sub_features_gdf, sub_metadata = self.payload_gdf(sub_payload, aoi_id)
                    features_gdf.append(sub_features_gdf)
                    metadata.append(sub_metadata)
                # Warning - using arbitrary int index means duplicate index.
                features_gdf = pd.concat(features_gdf) if len(features_gdf) > 0 else None

                # Check for repeat appearances of the same feature in the multipolygon
                if len(features_gdf.feature_id.unique()) < len(features_gdf):
                    features_gdf = self.trim_features_to_aoi(features_gdf, geometry, region)

                # Deduplicate metadata, picking from the first part of the multipolygon rather than attempting to merge
                metadata_df = pd.DataFrame(metadata).drop(columns=["link", "aoi_id"])
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
                payload = self.get_features(geometry, region, packs, since, until, address_fields, survey_resource_id)
                features_gdf, metadata = self.payload_gdf(payload, aoi_id)
        except AIFeatureAPIRequestSizeError as e:
            features_gdf, metadata, error = None, None, None

            # If the query was too big, split it up into a grid, and recombine as though it was one query.
            logging.debug(f"{fail_hard_regrid=}")
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
                        geometry, region, packs, aoi_id, since, until, survey_resource_id
                    )
                    if len(errors_df) == 0:
                        error = None
                    else:
                        raise Exception("This shouldn't happen")

                    # Recombine gridded features
                    features_gdf = FeatureApi.combine_features_gdf_from_grid(features_gdf)

                    # Creat metadata
                    metadata_df = metadata_df.drop_duplicates().iloc[0]

                    metadata = {
                        "aoi_id": metadata_df["aoi_id"],
                        "system_version": metadata_df["system_version"],
                        "link": metadata_df["link"],
                        "date": metadata_df["date"],
                    }
                    logger.debug(
                        f"Recombined grid - Metadata: {metadata}, Unique {AOI_ID_COLUMN_NAME} with features: {features_gdf[AOI_ID_COLUMN_NAME].unique()}, Error: {error} "
                    )

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
            logger.debug(f"Retry Exception - gave up retrying on aoi_id: {aoi_id}")
            features_gdf = None
            metadata = None
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": -1,
                "message": "RETRY_ERROR",
                "text": str(e),
                "request": "",
            }
        return features_gdf, metadata, error

    def get_features_gdf_gridded(
        self,
        geometry: Union[Polygon, MultiPolygon],
        region: str,
        packs: Optional[List[str]] = None,
        aoi_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        survey_resource_id: Optional[str] = None,
        grid_size: Optional[float] = 0.005,  # Approx 500m at the equator
    ) -> Tuple[Optional[gpd.GeoDataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Get feature data for an AOI. If a cache is configured, the cache will be checked before using the API.
        Data is returned as a GeoDataframe with response metadata and error information (if any occurred).

        Args:
            geometry: AOI in EPSG4326
            region: Country code
            packs: List of AI packs
            aoi_id: ID of the AOI to add to the data
            since: Earliest date to pull data for
            until: Latest date to pull data for
            survey_resource_id: The survey resource ID for the vector tile resources.
            grid_size: The AOI is gridded in the native projection (constants.API_CRS) to save compute.

        Returns:
            API response features GeoDataFrame, metadata dictionary, and an error dictionary
        """
        logger.debug(f"Gridding AOI into {grid_size} squares.")
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
                since_bulk=since,
                until_bulk=until,
                survey_resource_id_bulk=survey_resource_id,
                instant_fail_batch=True,
                fail_hard_regrid=True,
            )
        except AIFeatureAPIError as e:
            logger.debug(f"Failed whole grid for aoi_id {aoi_id}. Single error")
            logger.debug(f"Exception is {e}")
            raise AIFeatureAPIGridError(e.status_code)
        if len(features_gdf["survey_date"].unique()) > 1:
            logger.warning(
                f"Failed whole grid for aoi_id {aoi_id}. Multiple dates detected - certain to contain duplicates on grid boundaries."
            )
            raise AIFeatureAPIGridError(-1, message="Multiple dates on non survey resource ID query.")
        elif survey_resource_id is None:
            logger.debug(
                f"AOI {aoi_id} gridded on a single date - possible but unlikely to include deduplication errors (if two overlapping surveys flown on same date)."
            )
            # TODO: We should change query to guarantee same survey id is used somehow.

        if len(errors_df) > 0:
            raise AIFeatureAPIGridError(errors_df.query("status_code != 200").status_code.mode())
        else:
            logger.debug(f"Successfully gridded results for AOI ID: {aoi_id}, survey_resource_id: {survey_resource_id}")

            # Reset the correct aoi_id for the gridded result
            features_gdf[AOI_ID_COLUMN_NAME] = aoi_id
            metadata_df[AOI_ID_COLUMN_NAME] = aoi_id
            return features_gdf, metadata_df, errors_df

    def get_features_gdf_bulk(
        self,
        gdf: gpd.GeoDataFrame,
        region: str,
        packs: Optional[List[str]] = None,
        since_bulk: Optional[str] = None,
        until_bulk: Optional[str] = None,
        survey_resource_id_bulk: Optional[str] = None,
        instant_fail_batch: Optional[bool] = False,
        fail_hard_regrid: Optional[bool] = False,
    ) -> Tuple[Optional[gpd.GeoDataFrame], Optional[pd.DataFrame], pd.DataFrame]:
        """
        Get features data for many AOIs.

        Args:
            gdf: GeoDataFrame with AOIs
            region: Country code
            packs: List of AI packs
            since_bulk: Earliest date to pull data for, applied across all Query AOIs.
            until_bulk: Latest date to pull data for, applied across all Query AOIs.
            survey_resource_id_bulk: Impose a single survey resource ID from which to pull all responses.
            instant_fail_batch:  If true, raise an AIFeatureAPIError, otherwise create a dataframe of errors and
            return all good data available.
            fail_hard_regrid: should be False on an initial call, this just gets used internally to prevent us
                              getting stuck in an infinite loop of get_features_gdf -> get_features_gdf_gridded ->
                                                                   get_features_gdf_bulk -> get_features_gdf

        Returns:
            API responses as feature GeoDataFrames, metadata DataFrame, and an error DataFrame
        """
        if AOI_ID_COLUMN_NAME not in gdf.columns:
            raise KeyError(f"No ID column {AOI_ID_COLUMN_NAME} in dataframe, {gdf.columns=}")
        elif AOI_ID_COLUMN_NAME in gdf.columns[gdf.columns.duplicated()]:
            raise KeyError(f"Duplicate ID columns {AOI_ID_COLUMN_NAME} in dataframe, {gdf.columns=}")

        # are address fields present?
        has_address_fields = set(gdf.columns.tolist()).intersection(set(ADDRESS_FIELDS)) == set(ADDRESS_FIELDS)
        # is a geometry field present?
        has_geom = "geometry" in gdf.columns

        # Run in thread pool
        with concurrent.futures.ThreadPoolExecutor(self.workers) as executor:
            jobs = []
            for _, row in gdf.iterrows():
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
                        row[AOI_ID_COLUMN_NAME],
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
            for i, job in enumerate(jobs):
                aoi_data, aoi_metadata, aoi_error = job.result()
                if aoi_data is not None:
                    data.append(aoi_data)
                if aoi_metadata is not None:
                    metadata.append(aoi_metadata)
                if aoi_error is not None:
                    if instant_fail_batch:
                        executor.shutdown(wait=True, cancel_futures=True)  # Needed to prevent memory leak.
                        raise AIFeatureAPIError(aoi_error, aoi_error["request"])
                    else:
                        errors.append(aoi_error)
        # Combine results. reset_index() here because the index of the combined dataframes
        # is just the row number in the chunk submitted to each worker, and so we get an
        # index in the final dataframe that is not unique, and also not very useful.
        features_gdf = pd.concat(data).reset_index(drop=True) if len(data) > 0 else pd.DataFrame([])
        metadata_df = pd.DataFrame(metadata).reset_index(drop=True) if len(metadata) > 0 else pd.DataFrame([])
        errors_df = pd.DataFrame(errors).reset_index(drop=True) if len(errors) > 0 else pd.DataFrame([])

        return features_gdf, metadata_df, errors_df

    def get_rollup_df(
        self,
        geometry: Union[Polygon, MultiPolygon],
        region: str,
        packs: Optional[List[str]] = None,
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
            packs: List of AI packs
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
                        sub_geometry, region, packs, since, until, address_fields, survey_resource_id
                    )
                    sub_rollup_df, sub_metadata = self.payload_rollup_df(sub_payload, aoi_id)
                    rollup_df.append(sub_rollup_df)
                    metadata.append(sub_metadata)
                rollup_df = pd.concat(rollup_df) if len(rollup_df) > 0 else None
                # Warning - using arbitrary int index means duplicate index.

                # Deduplicate metadata, picking from the first part of the multipolygon rather than attempting to merge
                metadata_df = pd.DataFrame(metadata).drop(columns=["link", "aoi_id"])
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
                payload = self.get_rollup(geometry, region, packs, since, until, address_fields, survey_resource_id)
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
                "status_code": -1,
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
        since_bulk: Optional[str] = None,
        until_bulk: Optional[str] = None,
        survey_resource_id_bulk: Optional[str] = None,
        instant_fail_batch: Optional[bool] = False,
        fail_hard_regrid: Optional[bool] = False,
    ) -> Tuple[Optional[gpd.GeoDataFrame], Optional[pd.DataFrame], pd.DataFrame]:
        """
        Get features data for many AOIs.

        Args:
            gdf: GeoDataFrame with AOIs
            region: Country code
            packs: List of AI packs
            since_bulk: Earliest date to pull data for, applied across all Query AOIs.
            until_bulk: Latest date to pull data for, applied across all Query AOIs.
            survey_resource_id_bulk: Impose a single survey resource ID from which to pull all responses.
            instant_fail_batch:  If true, raise an AIFeatureAPIError, otherwise create a dataframe of errors and
            return all good data available.
            fail_hard_regrid: should be False on an initial call, this just gets used internally to prevent us
                              getting stuck in an infinite loop of get_features_gdf -> get_features_gdf_gridded ->
                                                                   get_features_gdf_bulk -> get_features_gdf

        Returns:
            API responses as rollup csv GeoDataFrames, metadata DataFrame, and an error DataFrame
        """
        if AOI_ID_COLUMN_NAME not in gdf.columns:
            raise KeyError(f"No ID column {AOI_ID_COLUMN_NAME} in dataframe, {gdf.columns=}")
        elif AOI_ID_COLUMN_NAME in gdf.columns[gdf.columns.duplicated()]:
            raise KeyError(f"Duplicate ID columns {AOI_ID_COLUMN_NAME} in dataframe, {gdf.columns=}")

        # are address fields present?
        has_address_fields = set(gdf.columns.tolist()).intersection(set(ADDRESS_FIELDS)) == set(ADDRESS_FIELDS)
        # is a geometry field present?
        has_geom = "geometry" in gdf.columns

        # Run in thread pool
        with concurrent.futures.ThreadPoolExecutor(self.workers) as executor:
            jobs = []
            for _, row in gdf.iterrows():
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
                        row[AOI_ID_COLUMN_NAME],
                        since,
                        until,
                        {f: row[f] for f in ADDRESS_FIELDS} if has_address_fields else None,
                        survey_resource_id,
                    )
                )
            data = []
            metadata = []
            errors = []
            for i, job in enumerate(jobs):
                aoi_data, aoi_metadata, aoi_error = job.result()
                if aoi_data is not None:
                    data.append(aoi_data)
                if aoi_metadata is not None:
                    metadata.append(aoi_metadata)
                if aoi_error is not None:
                    if instant_fail_batch:
                        raise AIFeatureAPIError(aoi_error, aoi_error["request"])
                    else:
                        errors.append(aoi_error)
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
