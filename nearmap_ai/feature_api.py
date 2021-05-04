import concurrent.futures
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import stringcase
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from shapely.geometry import MultiPolygon, Polygon, shape
from tqdm import tqdm


class AoiNotFound(Exception):
    pass


class AoiExceedsMaxSize(Exception):
    pass


class FeatureApi:
    FEATURES_URL = "https://api.nearmap.com/ai/features/v4/features.json"
    CLASSES_URL = "https://api.nearmap.com/ai/features/v4/classes.json"
    CHAR_LIMIT = 3800
    SOURCE_CRS = "EPSG:4326"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        overwrite_cache: Optional[bool] = False,
        workers: Optional[int] = 10,
    ):
        """
        Initialize FeatureApi class

        Args:
            api_key: Nearmap API key. If not defined the environment variable will be used
            cache_dir: Directory to use as a payload cache
            overwrite_cache: Set to overwrite values stored in the cache
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
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = self._get_session()
        self.overwrite_cache = overwrite_cache
        self.workers = workers

    @staticmethod
    def _get_session() -> requests.Session:
        """
        Return a request session with retrying configured.
        """
        session = requests.Session()
        retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def get_feature_class_ids(self) -> pd.DataFrame:
        """
        Get the feature class IDs and descriptions as a dataframe.
        """
        # Create request string
        request_string = f"{self.CLASSES_URL}?apikey={self.api_key}"

        # Request data
        t1 = time.monotonic()
        response = self._session.get(request_string)
        response_time_ms = (time.monotonic() - t1) * 1e3
        logging.info(f"{response_time_ms:.1f}ms response time for classes.json")
        # Check for errors
        if not response.ok:
            # Fail hard for unexpected errors
            raise RuntimeError(f"\n{request_string=}\n\n{response.status_code=}\n\n{response.text}\n\n")
        data = response.json()
        df_classes = pd.DataFrame(data["classes"]).set_index("id")
        return df_classes

    @staticmethod
    def _polygon_to_coordstring(poly: Polygon) -> str:
        """
        Turn a shapely polygon into the format required by the API for a query polygon.
        """
        coords = poly.boundary.coords[:]
        flat_coords = np.array(coords).flatten()
        coordstring = ",".join(flat_coords.astype(str))
        return coordstring

    def _geometry_to_coordstring(self, geometry: Union[Polygon, MultiPolygon]) -> Tuple[str, bool]:
        """
        Turn a shapely polygon or multipolygon into a single coord sting to be used in API requests.
        To meet the contraints on the API URL the following changes may be applied:
         - Multipolygons are simplified to single polygons by taking the convex hull
         - Polygons that have too many coordinates (resulting in strings that are too long) are
           simplified by taking the convex hull.
         - Polygons that have a convex hull with too many coordinates are simplified to a box.
        If the coord string return does not represent the polygon exactly, the exact flag is set to False.
        """

        if isinstance(geometry, MultiPolygon):
            coordstring = self._polygon_to_coordstring(geometry.convex_hull)
            exact = False
        else:
            coordstring = self._polygon_to_coordstring(geometry)
            exact = True
        if len(coordstring) > self.CHAR_LIMIT:
            exact = False
            coordstring = self._polygon_to_coordstring(geometry.convex_hull)
        if len(coordstring) > self.CHAR_LIMIT:
            exact = False
            coordstring = self._polygon_to_coordstring(geometry.envelope)
        return coordstring, exact

    def _request_cache_path(self, request_string: str) -> str:
        """
        Hash a request string to create a cache path.
        """
        request_hash = hashlib.md5(request_string.encode()).hexdigest()
        return self.cache_dir / f"{request_hash}.json"

    def _request_error_message(self, request_string: str, response: requests.Response) -> str:
        """
        Create a descriptive error message without the API key.
        """
        return f"\n{request_string.replace(self.api_key, '...')=}\n\n{response.status_code=}\n\n{response.text}\n\n"

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
        coordstring, exact = self._geometry_to_coordstring(geometry)
        request_string = f"{self.FEATURES_URL}?polygon={coordstring}&apikey={self.api_key}"

        # Add dates if given
        if since:
            request_string += f"&since={since}"
        if until:
            request_string += f"&until={until}"
        # Add packs if given
        if packs:
            packs = ",".join(packs)
            request_string += f"&packs={packs}"

        # Check if it's already cached
        cache_path = self._request_cache_path(request_string)
        if self.cache_dir and not self.overwrite_cache:
            if cache_path.exists():
                logging.info(f"Retrieving payload from cache")
                with open(cache_path, "r") as f:
                    return json.load(f)

        # Request data
        t1 = time.monotonic()
        response = self._session.get(request_string)
        response_time_ms = (time.monotonic() - t1) * 1e3
        logging.info(f"{response_time_ms:.1f}ms response time for polygon with these packs: {packs}")
        # Check for errors
        if response.status_code == 404:
            raise AoiNotFound(f"AOI not found: {self._request_error_message(request_string, response)}")
        elif response.status_code == 400 and response.json()["code"] == "AOI_EXCEEDS_MAX_SIZE":
            raise AoiExceedsMaxSize(f"AOI too large: {self._request_error_message(request_string, response)}")
        elif not response.ok:
            # Fail hard for unexpected errors
            raise RuntimeError(self._request_error_message(request_string, response))
        data = response.json()

        # If the AOI was altered for the API request, we need to filter features in the response
        if not exact:
            data["features"] = [f for f in data["features"] if shape(f["geometry"]).intersects(geometry)]

        # Save to cache if configured
        if self.cache_dir:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
        return data

    @staticmethod
    def link_to_date(link: str) -> str:
        """
        Parse the date from a Map Browser link.
        """
        date = link.split("/")[-1]
        return f"{date[:4]}-{date[4:6]}-{date[6:8]}"

    @staticmethod
    def payload_gdf(payload: dict, aoi_id: Optional[str] = None) -> Tuple[gpd.GeoDataFrame, dict]:
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
            "link": payload["link"],
            "date": FeatureApi.link_to_date(payload["link"]),
        }

        # Create features DataFrame
        if len(payload["features"]) == 0:
            df = pd.DataFrame(
                [],
                columns=[
                    "id",
                    "classId",
                    "description",
                    "confidence",
                    "parentId",
                    "geometry",
                    "areaSqm",
                    "areaSqft",
                    "attributes",
                    "surveyDate",
                    "meshDate",
                ],
            )
        else:
            df = pd.DataFrame(payload["features"])
        df = df.rename(columns={"id": "feature_id"})
        df.columns = [stringcase.snakecase(c) for c in df.columns]

        # Add AOI ID if specified
        if aoi_id:
            df["aoi_id"] = aoi_id
            metadata["aoi_id"] = aoi_id
        # Cast to GeoDataFrame
        gdf = gpd.GeoDataFrame(df.drop("geometry", axis=1), geometry=df.geometry.apply(shape))
        gdf = gdf.set_crs(FeatureApi.SOURCE_CRS)
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
        except (AoiNotFound, AoiExceedsMaxSize) as e:
            # Catch acceptable errors
            return None, None, {"aoi_id": aoi_id, "error": str(e)}
        features_gdf, metadata = self.payload_gdf(payload, aoi_id)
        return features_gdf, metadata, None

    def get_features_gdf_bulk(
        self,
        gdf: gpd.GeoDataFrame,
        packs: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get features data for many AOIs.

        Args:
            gdf: GeoDataFrame with AOIs
            packs: List of AI packs
            since: Earliest date to pull data for
            until: Latest date to pull data for

        Returns:
            API responses as feature GeoDataFrames, metadata DataFrame, and a error DataFrame
        """
        if "aoi_id" not in gdf.columns:
            raise KeyError(f"No 'aoi_id' column in dataframe, {gdf.columns=}")

        # Run in thread pool
        with concurrent.futures.ThreadPoolExecutor(self.workers) as executor:
            jobs = []
            for _, row in gdf.iterrows():
                jobs.append(executor.submit(self.get_features_gdf, row.geometry, packs, row.aoi_id, since, until))
            data = []
            metadata = []
            errors = []
            for job in tqdm(jobs):
                aoi_data, aoi_metadata, aoi_error = job.result()
                if aoi_data is not None:
                    data.append(aoi_data)
                if aoi_metadata is not None:
                    metadata.append(aoi_metadata)
                if aoi_error is not None:
                    errors.append(aoi_error)
        # Combine results
        features_gdf = pd.concat(data)
        metadata_df = pd.DataFrame(metadata)
        errors_df = pd.DataFrame(errors, columns=["aoi_id", "error"])
        return features_gdf, metadata_df, errors_df
