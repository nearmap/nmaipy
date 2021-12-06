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

import gzip
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from shapely.geometry import MultiPolygon, Polygon, shape
import stringcase

from nearmap_ai import log
from nearmap_ai.constants import AOI_ID_COLUMN_NAME, LAT_LONG_CRS, SINCE_COL_NAME, UNTIL_COL_NAME

logger = log.get_logger()


class RetryRequest(Retry):
    BACKOFF_MAX = 0.6


class AIFeatureAPIError(Exception):
    def __init__(self, response, request_string):
        self.status_code = response.status_code
        self.text = response.text
        self.request_string = request_string
        try:
            err_body = response.json()
            self.message = err_body["message"] if "message" in err_body else err_body.get("error", "")
        except json.JSONDecodeError:
            self.message = ""


class FeatureApi:
    FEATURES_URL = "https://api.nearmap.com/ai/features/v4/features.json"
    CLASSES_URL = "https://api.nearmap.com/ai/features/v4/classes.json"
    PACKS_URL = "https://api.nearmap.com/ai/features/v4/packs.json"
    CHAR_LIMIT = 3800
    SOURCE_CRS = LAT_LONG_CRS

    def __init__(
        self,
        api_key: Optional[str] = None,
        bulk_mode: Optional[bool] = True,
        cache_dir: Optional[Path] = None,
        overwrite_cache: Optional[bool] = False,
        compress_cache: Optional[bool] = False,
        workers: Optional[int] = 10,
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
        self._sessions = []
        self._thread_local = threading.local()
        self.overwrite_cache = overwrite_cache
        self.compress_cache = compress_cache
        self.workers = workers
        self.bulk_mode = bulk_mode

    @property
    def _session(self) -> requests.Session:
        """
        Return a request session with retrying configured.
        """
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            retries = RetryRequest(
                total=100,
                backoff_factor=0.05,
                status_forcelist=[
                    HTTPStatus.TOO_MANY_REQUESTS,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    HTTPStatus.BAD_GATEWAY,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                ],
            )
            session.mount("https://", HTTPAdapter(max_retries=retries))
            self._thread_local.session = session
            self._sessions.append(session)
        return session

    def get_packs(self) -> Dict[str, List[str]]:
        """
        Get packs with class IDs
        """
        # Create request string
        request_string = f"{self.PACKS_URL}?apikey={self.api_key}"
        # Request data
        t1 = time.monotonic()
        response = self._session.get(request_string)
        response_time_ms = (time.monotonic() - t1) * 1e3
        logger.debug(f"{response_time_ms:.1f}ms response time for packs.json")
        # Check for errors
        if not response.ok:
            # Fail hard for unexpected errors
            raise RuntimeError(f"\n{request_string=}\n\n{response.status_code=}\n\n{response.text}\n\n")
        data = response.json()
        return {p["code"]: [c["id"] for c in p["featureClasses"]] for p in data["packs"]}

    def get_feature_classes(self, packs: List[str] = None) -> pd.DataFrame:
        """
        Get the feature class IDs and descriptions as a dataframe.

        Args:
            packs: If defined, classes will be filtered to the set of packs
        """
        # Create request string
        request_string = f"{self.CLASSES_URL}?apikey={self.api_key}"

        # Request data
        t1 = time.monotonic()
        response = self._session.get(request_string)
        response_time_ms = (time.monotonic() - t1) * 1e3
        logger.debug(f"{response_time_ms:.1f}ms response time for classes.json")
        # Check for errors
        if not response.ok:
            # Fail hard for unexpected errors
            raise RuntimeError(f"\n{request_string=}\n\n{response.status_code=}\n\n{response.text}\n\n")
        data = response.json()
        df_classes = pd.DataFrame(data["classes"]).set_index("id")

        # Filter classes to packs
        if packs:
            pack_classes = self.get_packs()
            if diff := set(packs) - set(pack_classes.keys()):
                raise ValueError(f"Unknown packs: {diff}")
            all_classes = set([class_id for p in packs for class_id in pack_classes[p]])
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

    @classmethod
    def _geometry_to_coordstring(cls, geometry: Union[Polygon, MultiPolygon]) -> Tuple[str, bool]:
        """
        Turn a shapely polygon or multipolygon into a single coord sting to be used in API requests.
        To meet the constraints on the API URL the following changes may be applied:
         - Multipolygons are simplified to single polygons by taking the convex hull
         - Polygons that have too many coordinates (resulting in strings that are too long) are
           simplified by taking the convex hull.
         - Polygons that have a convex hull with too many coordinates are simplified to a box.
        If the coord string return does not represent the polygon exactly, the exact flag is set to False.
        """

        if isinstance(geometry, MultiPolygon):
            if len(geometry) == 1:
                coordstring = cls._polygon_to_coordstring(geometry[0])
                exact = True
            else:
                logger.warning(f"Geometry is a multipolygon - approximating. Length: {len(geometry)}")
                logger.warning(geometry)
                coordstring = cls._polygon_to_coordstring(geometry.convex_hull)
                exact = False
        else:
            coordstring = cls._polygon_to_coordstring(geometry)
            exact = True

        if len(coordstring) > cls.CHAR_LIMIT:
            exact = False
            coordstring = cls._polygon_to_coordstring(geometry.convex_hull)
        if len(coordstring) > cls.CHAR_LIMIT:
            exact = False
            coordstring = cls._polygon_to_coordstring(geometry.envelope)
        return coordstring, exact

    def _make_latlon_path_for_cache(self, request_string):
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
        if self.compress_cache:
            return self.cache_dir / lon / lat / f"{request_hash}.json.gz"
        else:
            return self.cache_dir / lon / lat / f"{request_hash}.json"

    def _request_error_message(self, request_string: str, response: requests.Response) -> str:
        """
        Create a descriptive error message without the API key.
        """
        return f"\n{request_string.replace(self.api_key, '...')=}\n\n{response.status_code=}\n\n{response.text}\n\n"

    def _handle_response_errors(self, response: requests.Response, request_string: str):
        """
        Handle errors returned from the feature API
        """
        if not response.ok:
            clean_request_string = request_string.replace(self.api_key, "...")
            raise AIFeatureAPIError(response, clean_request_string)

    def _write_to_cache(self, path, payload):
        """
        Write a payload to the cache. To make the write atomic, data is first written to a temp file and then renamed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.cache_dir / f"{str(uuid.uuid4())}.tmp"
        if self.compress_cache:
            temp_path = Path(str(temp_path) + ".gz")
            with gzip.open(temp_path, "w") as f:
                payload_bytes = json.dumps(payload).encode("utf-8")
                f.write(payload_bytes)
                f.flush()
                os.fsync(f.fileno())
                temp_path.replace(path)
        else:
            with open(temp_path, "w") as f:
                json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
            temp_path.replace(path)

    def _create_request_string(
        self,
        geometry: Union[Polygon, MultiPolygon],
        packs: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """
        Create a request string with given parameters
        """
        coordstring, exact = self._geometry_to_coordstring(geometry)
        bulk_str = str(self.bulk_mode).lower()
        request_string = f"{self.FEATURES_URL}?polygon={coordstring}&bulk={bulk_str}&apikey={self.api_key}"

        # Add dates if given
        if since:
            request_string += f"&since={since}"
        if until:
            request_string += f"&until={until}"
        # Add packs if given
        if packs:
            packs = ",".join(packs)
            request_string += f"&packs={packs}"
        return request_string, exact

    def get_features(
        self,
        geometry: Union[Polygon, MultiPolygon],
        packs: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ):
        """
        Get feature data for a AOI. If a cache is configured, the cache will be checked before using the API.

        Args:
            geometry: AOI in EPSG4326
            packs: List of AI packs
            since: Earliest date to pull data for
            until: Latest date to pull data for

        Returns:
            API response as a Dictionary
        """
        # Create request string
        request_string, exact = self._create_request_string(geometry, packs, since, until)

        # Check if it's already cached
        if self.cache_dir is not None and not self.overwrite_cache:
            cache_path = self._request_cache_path(request_string)
            if cache_path.exists():
                logger.debug(f"Retrieving payload from cache")
                if self.compress_cache:
                    with gzip.open(cache_path, 'r') as f:
                        return json.loads(f.read().decode('utf-8'))
                else:
                    with open(cache_path, "r") as f:
                        return json.load(f)

        # Request data
        t1 = time.monotonic()
        response = self._session.get(request_string)
        response_time_ms = (time.monotonic() - t1) * 1e3
        if response.ok:
            logger.debug(f"{response_time_ms:.1f}ms response time for polygon with these packs: {packs}")
        else:
            logger.debug(f"{response_time_ms:.1f}ms failure response time {response.text}")
        # Check for errors
        self._handle_response_errors(response, request_string)
        # Parse results
        try:
            data = response.json()
        except json.JSONDecodeError:
            logger.error(response.text)
            raise json.JSONDEcodeError

        # If the AOI was altered for the API request, we need to filter features in the response
        if not exact:
            data["features"] = [f for f in data["features"] if shape(f["geometry"]).intersects(geometry)]

        # Save to cache if configured
        if self.cache_dir is not None:
            self._write_to_cache(self._request_cache_path(request_string), data)
        return data

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
        Check whether the link contains the locationmarker flag, and add it if not present..
        """
        location_marker_string = "?locationMarker"
        if not location_marker_string not in link:
            return link.append(location_marker_string)
        else:
            return link

    @classmethod
    def payload_gdf(cls, payload: dict, aoi_id: Optional[str] = None) -> Tuple[gpd.GeoDataFrame, dict]:
        """
        Create a GeoDataFrame from a API response dictionary.

        Args:
            payload: API response dictionary
            aoi_id: Optional ID for the AOI to add to the data

        Returns:
            Features GeoDataFrame
            Metadata dictionary
        """

        # Creat metadata
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
            for colname in set(columns).difference(set(df.columns)):
                df[colname] = None

        df = df.rename(columns={"id": "feature_id"})
        df.columns = [stringcase.snakecase(c) for c in df.columns]

        # Add AOI ID if specified
        if aoi_id is not None:
            df[AOI_ID_COLUMN_NAME] = aoi_id
            metadata[AOI_ID_COLUMN_NAME] = aoi_id
        # Cast to GeoDataFrame
        gdf = gpd.GeoDataFrame(df.drop("geometry", axis=1), geometry=df.geometry.apply(shape))
        gdf = gdf.set_crs(cls.SOURCE_CRS)
        return gdf, metadata

    def get_features_gdf(
        self,
        geometry: Union[Polygon, MultiPolygon],
        packs: Optional[List[str]] = None,
        aoi_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> Tuple[Optional[gpd.GeoDataFrame], Optional[dict], Optional[dict]]:
        """
        Get feature data for a AOI. If a cache is configured, the cache will be checked before using the API.
        Data is returned as a GeoDataframe with response metadata and error information (if any occurred).

        Args:
            geometry: AOI in EPSG4326
            packs: List of AI packs
            aoi_id: ID of the AOI to add to the data
            since: Earliest date to pull data for
            until: Latest date to pull data for

        Returns:
            API response features GeoDataFrame, metadata dictionary, and a error dictionary
        """
        try:
            payload = self.get_features(geometry, packs, since, until)
            features_gdf, metadata = self.payload_gdf(payload, aoi_id)
            error = None
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
            logger.error(f"Retry Exception - gave up retrying on aoi_id: {aoi_id}")
            features_gdf = None
            metadata = None
            error = {
                AOI_ID_COLUMN_NAME: aoi_id,
                "status_code": "RETRY_ERROR",
                "message": "",
                "text": str(e),
                "request": "",
            }

        return features_gdf, metadata, error

    def get_features_gdf_bulk(
        self,
        gdf: gpd.GeoDataFrame,
        packs: Optional[List[str]] = None,
        since_bulk: Optional[str] = None,
        until_bulk: Optional[str] = None,
    ) -> Tuple[Optional[gpd.GeoDataFrame], Optional[pd.DataFrame], pd.DataFrame]:
        """
        Get features data for many AOIs.

        Args:
            gdf: GeoDataFrame with AOIs
            packs: List of AI packs
            since_bulk: Earliest date to pull data for, applied across all Query AOIs.
            until_bulk: Latest date to pull data for, applied across all Query AOIs.

        Returns:
            API responses as feature GeoDataFrames, metadata DataFrame, and a error DataFrame
        """
        if AOI_ID_COLUMN_NAME not in gdf.columns:
            raise KeyError(f"No 'aoi_id' column in dataframe, {gdf.columns=}")

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

                jobs.append(
                    executor.submit(
                        self.get_features_gdf,
                        row.geometry,
                        packs,
                        row[AOI_ID_COLUMN_NAME],
                        since,
                        until,
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
                    errors.append(aoi_error)
        # Combine results
        features_gdf = pd.concat(data) if len(data) > 0 else None
        metadata_df = pd.DataFrame(metadata) if len(metadata) > 0 else None
        errors_df = pd.DataFrame(errors)
        return features_gdf, metadata_df, errors_df
