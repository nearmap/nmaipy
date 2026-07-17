"""
Client for the Nearmap Roof Age API.

The Roof Age API provides predicted roof installation dates based on analysis of
imagery, AI data, building permits, and climate information. This API is currently
available for US properties only.

Example usage:
    ```python
    from nmaipy.roof_age_api import RoofAgeApi
    import geopandas as gpd
    from shapely.geometry import Polygon

    # Initialize client
    client = RoofAgeApi(api_key="your_api_key")

    # Query by AOI
    aoi = Polygon([...])  # Your area of interest
    roofs_gdf = client.get_roof_age_by_aoi(aoi, aoi_id="parcel_123")

    # Query by address
    address = {
        "streetAddress": "123 Main St",
        "city": "Austin",
        "state": "TX",
        "zip": "78701"
    }
    roofs_gdf = client.get_roof_age_by_address(address, aoi_id="property_456")
    ```
"""

import concurrent.futures
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import stringcase
from shapely.geometry import Polygon, shape

from nmaipy import log, storage
from nmaipy.api_common import BaseApiClient, RoofAgeAPIError
from nmaipy.constants import (
    ADDRESS_FIELDS,
    AOI_ID_COLUMN_NAME,
    API_CRS,
    FEATURE_CLASS_DESCRIPTIONS,
    ROOF_AGE_AREA_FIELD,
    ROOF_AGE_ASSESSOR_DATA_DETAILS_FIELD,
    ROOF_AGE_DEFAULT_PAGE_LIMIT,
    ROOF_AGE_DEFAULT_RESOURCE_ID,
    ROOF_AGE_HILBERT_ID_FIELD,
    ROOF_AGE_MAPBROWSER_URL_FIELD,
    ROOF_AGE_MODEL_VERSION_FIELD,
    ROOF_AGE_NEXT_CURSOR_FIELD,
    ROOF_AGE_RELEVANT_PERMITS_DETAILS_FIELD,
    ROOF_AGE_RESOURCE_ID_FIELD,
    ROOF_AGE_RESOURCE_PATH,
    ROOF_AGE_TIMELINE_FIELD,
    ROOF_AGE_URL_ROOT,
    ROOF_INSTANCE_CLASS_ID,
    SINCE_COL_NAME,
    UNTIL_COL_NAME,
)

logger = log.get_logger()


class RoofAgeApi(BaseApiClient):
    """
    Client for the Nearmap Roof Age API.

    Provides methods to query roof installation dates by AOI or address.
    Inherits session management, caching, and retry logic from BaseApiClient.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        overwrite_cache: Optional[bool] = False,
        compress_cache: Optional[bool] = False,
        threads: Optional[int] = 10,
        url_root: Optional[str] = None,
        country: str = "us",
        progress_counters: Optional[dict] = None,
        bulk_mode: bool = True,
        resource_id: str = ROOF_AGE_DEFAULT_RESOURCE_ID,
        until_as_of_date: Optional[str] = None,
        since_as_of_date: Optional[str] = None,
        bearer_token: Optional[str] = None,
    ):
        """
        Initialize Roof Age API client.

        Args:
            api_key: Nearmap API key. If not provided, reads from API_KEY environment variable
            cache_dir: Directory to use as a payload cache
            overwrite_cache: Whether to overwrite cached values
            compress_cache: Whether to use gzip compression for cache files
            threads: Number of threads for concurrent execution
            url_root: Override the default API root URL (for testing)
            country: Country code for address queries (e.g., 'us', 'au')
            progress_counters: Optional dict with 'total', 'completed', and 'lock' for tracking progress across processes
            bulk_mode: When True, uses bulk API mode with higher rate limits (1000 rps vs 20 rps)
            resource_id: Roof Age dataset resource id (e.g. ``"latest"`` for the API team's pointer to the
                current default, or a UUID for a specific dataset). Resource ids identify *datasets*, not
                model versions — the per-row ``model_version`` field is the source of truth for which model
                produced a given record.
            until_as_of_date: Optional ``YYYY-MM-DD`` upper-bound cutoff. When set, the API returns roof state
                as of that date.
            since_as_of_date: Optional ``YYYY-MM-DD`` lower-bound cutoff. When set, the API restricts the
                response to roof state from that date onwards.
            bearer_token: Short-lived Nearmap identity JWT used instead of an API key
                (see BaseApiClient for lifetime caveats)

            Note: cutoff parameters are not supported on the A.0 dataset. Callers are expected to validate
            the combination upstream — sending a cutoff against A.0 returns HTTP 500.
        """
        super().__init__(
            api_key=api_key,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            compress_cache=compress_cache,
            threads=threads,
            bearer_token=bearer_token,
        )

        # Store progress counters for cross-process progress tracking
        self.progress_counters = progress_counters

        # Store country for address queries
        self.country = country.upper()

        # Store bulk mode setting
        self.bulk_mode = bulk_mode

        # Store dataset selection
        self.resource_id = resource_id
        self.until_as_of_date = until_as_of_date
        self.since_as_of_date = since_as_of_date

        # Configure API endpoints
        if url_root is None:
            url_root = ROOF_AGE_URL_ROOT
        self.base_url = f"https://{url_root}/{ROOF_AGE_RESOURCE_PATH}/{resource_id}"

        logger.debug(
            f"Initialized RoofAgeApi with base_url: {self._clean_api_key(self.base_url)}, "
            f"resource_id: {resource_id}, until_as_of_date: {until_as_of_date}, "
            f"since_as_of_date: {since_as_of_date}, bulk_mode: {bulk_mode}"
        )

    def _qualifier_subdir(self, until_as_of_date: Optional[str], since_as_of_date: Optional[str]) -> str:
        """Build the cache sub-directory that scopes a request by dataset and cutoffs.

        Layout: ``roofage/<resource_id>/<until_or_none>/<since_or_none>``. Pulling these out of the
        cache *key* and into the *path* makes the cache human-inspectable (``ls cache/roofage/A.1/...``)
        and removes the previous ``__``-separator parsing that this client used to do.
        """
        return storage.join_path(
            "roofage",
            self.resource_id,
            until_as_of_date or "none",
            since_as_of_date or "none",
        )

    def _get_cache_path(
        self,
        cache_key: str,
        *,
        until_as_of_date: Optional[str] = None,
        since_as_of_date: Optional[str] = None,
    ) -> str:
        """
        Get the cache file path for a given key.

        Cache layout:
        ``<cache_dir>/roofage/<resource_id>/<until_or_none>/<since_or_none>/...``
            - address keys: ``.../<country>/<state>/<city>/<zip>/<hash>.json``
            - AOI keys:     ``.../<wkt-hash>.json``

        Args:
            cache_key: Unique identifier for the cached item.
            until_as_of_date: Effective upper-bound cutoff for this call.
            since_as_of_date: Effective lower-bound cutoff for this call.

        Returns:
            Path to the cache file.
        """
        if self.cache_dir is None:
            raise ValueError("Cache directory not configured")

        extension = ".json.gz" if self.compress_cache else ".json"
        qualifier_dir = storage.join_path(
            self.cache_dir,
            self._qualifier_subdir(until_as_of_date, since_as_of_date),
        )

        if cache_key.startswith("roofage_address_"):
            # Address keys carry a slash-separated <country>/<state>/<city>/<zip>/<street> tail.
            address_path = cache_key[len("roofage_address_") :]
            parts = address_path.split("/")
            if len(parts) >= 4:
                country, state, city, zipcode = parts[0], parts[1], parts[2], parts[3]
                key_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
                cache_subdir = storage.join_path(
                    qualifier_dir,
                    self._sanitize_path_component(country),
                    self._sanitize_path_component(state),
                    self._sanitize_path_component(city),
                    self._sanitize_path_component(zipcode),
                )
                storage.ensure_directory(cache_subdir)
                return storage.join_path(cache_subdir, f"{key_hash}{extension}")

        # AOI queries (and any fallback): hash-based flat layout under the qualifier dir.
        storage.ensure_directory(qualifier_dir)
        key_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        return storage.join_path(qualifier_dir, f"{key_hash}{extension}")

    def _load_from_cache(
        self,
        cache_key: str,
        *,
        until_as_of_date: Optional[str] = None,
        since_as_of_date: Optional[str] = None,
    ) -> Optional[Dict]:
        """Override of BaseApiClient._load_from_cache that routes through the qualifier-aware path."""
        if self.cache_dir is None or self.overwrite_cache:
            return None

        cache_path = self._get_cache_path(
            cache_key, until_as_of_date=until_as_of_date, since_as_of_date=since_as_of_date
        )
        if not storage.file_exists(cache_path):
            return None

        try:
            return storage.read_json(cache_path, compressed=self.compress_cache)
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None

    def _save_to_cache(
        self,
        cache_key: str,
        data: Dict,
        *,
        until_as_of_date: Optional[str] = None,
        since_as_of_date: Optional[str] = None,
    ) -> None:
        """Override of BaseApiClient._save_to_cache that routes through the qualifier-aware path."""
        if self.cache_dir is None:
            return

        cache_path = self._get_cache_path(
            cache_key, until_as_of_date=until_as_of_date, since_as_of_date=since_as_of_date
        )
        try:
            storage.write_json(cache_path, data, compressed=self.compress_cache)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def _build_request_payload(
        self,
        aoi: Optional[Polygon] = None,
        address: Optional[Dict[str, str]] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        until_as_of_date: Optional[str] = None,
        since_as_of_date: Optional[str] = None,
    ) -> Dict:
        """
        Build the request payload for the Roof Age API.

        Args:
            aoi: Polygon geometry (mutually exclusive with address)
            address: Address dict with keys: streetAddress, city, state, zip
            cursor: Pagination cursor from previous response (omit for first request)
            limit: Maximum number of features per page (default: API default of 1000)
            until_as_of_date: Optional ``YYYY-MM-DD`` upper-bound cutoff. When None, no cutoff is sent.
            since_as_of_date: Optional ``YYYY-MM-DD`` lower-bound cutoff. When None, no cutoff is sent.

        Returns:
            Request payload dict

        Raises:
            ValueError: If neither or both aoi and address are provided
        """
        if aoi is not None and address is not None:
            raise ValueError("Cannot specify both aoi and address")
        if aoi is None and address is None:
            raise ValueError("Must specify either aoi or address")

        payload = {}
        if aoi is not None:
            # Convert Shapely Polygon to GeoJSON
            geojson = json.loads(gpd.GeoSeries([aoi], crs=API_CRS).to_json())
            # Extract just the geometry (not the full FeatureCollection)
            payload["aoi"] = geojson["features"][0]["geometry"]
        else:
            # Validate address fields
            for field in ADDRESS_FIELDS:
                if field not in address:
                    raise ValueError(f"Address missing required field: {field}")
            # Add country field (required by API)
            address_with_country = {**address, "country": self.country}
            payload["address"] = address_with_country

        # Add pagination parameters if specified
        if cursor is not None:
            payload["cursor"] = cursor
        if limit is not None:
            payload["limit"] = limit

        if until_as_of_date is not None:
            payload["untilAsOfDate"] = until_as_of_date
        if since_as_of_date is not None:
            payload["sinceAsOfDate"] = since_as_of_date

        return payload

    def _build_cache_key(
        self,
        aoi: Optional[Polygon] = None,
        address: Optional[Dict[str, str]] = None,
    ) -> str:
        """Build a cache key identifying the request's geometry / address shape.

        Dataset (``resource_id``) and cutoff dates (``until_as_of_date``, ``since_as_of_date``) are
        no longer embedded here — they're encoded in the cache *path* via :meth:`_get_cache_path`.
        Two requests for the same address against different datasets/cutoffs share a key but resolve
        to different files.
        """
        if aoi is not None:
            return f"roofage_aoi_{aoi.wkt}"
        # Address keys keep a slash-separated <country>/<state>/<city>/<zip>/<street> tail so
        # _get_cache_path can build a nested directory layout from them.
        address_path = (
            f"{self.country}/"
            f"{address.get('state', '_')}/"
            f"{address.get('city', '_')}/"
            f"{address.get('zip', '_')}/"
            f"{address.get('streetAddress', '_')}"
        )
        return f"roofage_address_{address_path}"

    def _parse_response(self, response_data: Dict, aoi_id: str) -> gpd.GeoDataFrame:
        """
        Parse Roof Age API response into a GeoDataFrame.

        Args:
            response_data: JSON response from API
            aoi_id: AOI identifier to add to each feature

        Returns:
            GeoDataFrame with roof features and properties
        """
        if response_data.get("type") != "FeatureCollection":
            logger.warning(f"Unexpected response type: {response_data.get('type')}")

        features = response_data.get("features", [])
        if not features:
            logger.debug(f"No roof features found for {aoi_id}")
            # Return empty GeoDataFrame with expected columns (snake_case)
            return gpd.GeoDataFrame(
                columns=[
                    AOI_ID_COLUMN_NAME,
                    "class_id",
                    "description",
                    "geometry",
                    "installation_date",
                    "trust_score",
                    "area_sqm",
                ],
                crs=API_CRS,
            )

        # Parse features into records
        # Note: The API returns both "roof" and "parcel" kind features
        # "parcel" features are property boundaries with attached roof age info when no roof instance exists
        # We keep both, but "parcel" features should be excluded during parent/child matching with Feature API roofs
        records = []
        geometries = []
        for feature in features:
            props = feature.get("properties", {})

            # Extract geometry
            geom = shape(feature["geometry"])
            geometries.append(geom)
            props[AOI_ID_COLUMN_NAME] = aoi_id

            # Add class_id and description for unified feature model
            # This allows roof instances to be treated like any other feature class
            # TODO: This is a temporary measure only for Roof Age API responses
            props["class_id"] = ROOF_INSTANCE_CLASS_ID
            props["description"] = FEATURE_CLASS_DESCRIPTIONS.get(ROOF_INSTANCE_CLASS_ID, "Roof Instance")

            # Serialize nested list/dict fields to JSON strings for CSV/Parquet
            # compatibility. timeline, relevantPermitsDetails and
            # assessorDataDetails arrive as lists of dicts; left raw they reach
            # the rollup as NumPy object arrays whose str() is single-quoted and
            # newline-separated (invalid JSON, breaks line-based ETL). Doing it
            # here — before snakecase conversion and before parcel_rollup maps
            # these columns — guarantees identical clean JSON on every
            # downstream path (rollup, per-class files, features.parquet).
            for nested_field in (
                ROOF_AGE_TIMELINE_FIELD,
                ROOF_AGE_RELEVANT_PERMITS_DETAILS_FIELD,
                ROOF_AGE_ASSESSOR_DATA_DETAILS_FIELD,
            ):
                if isinstance(props.get(nested_field), (list, dict)):
                    props[nested_field] = json.dumps(props[nested_field])

            # Append ?locationMarker to mapBrowserUrl if not present
            if ROOF_AGE_MAPBROWSER_URL_FIELD in props:
                url = props[ROOF_AGE_MAPBROWSER_URL_FIELD]
                if url and not url.endswith("?locationMarker"):
                    props[ROOF_AGE_MAPBROWSER_URL_FIELD] = f"{url}?locationMarker"

            records.append(props)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=API_CRS)

        # Map Roof Age API 'area' field to area_sqm for consistency
        # Note: Roof instances don't have clipped/unclipped distinction - they have a single 'area'
        if ROOF_AGE_AREA_FIELD in gdf.columns:
            gdf["area_sqm"] = gdf[ROOF_AGE_AREA_FIELD]

        # Use hilbertId as feature_id if available
        if ROOF_AGE_HILBERT_ID_FIELD in gdf.columns:
            gdf["feature_id"] = gdf[ROOF_AGE_HILBERT_ID_FIELD]
        else:
            gdf["feature_id"] = [f"roof_instance_{i}" for i in range(len(gdf))]

        # Add metadata from top level
        if ROOF_AGE_RESOURCE_ID_FIELD in response_data:
            gdf[ROOF_AGE_RESOURCE_ID_FIELD] = response_data[ROOF_AGE_RESOURCE_ID_FIELD]
        if ROOF_AGE_MODEL_VERSION_FIELD in response_data:
            gdf[ROOF_AGE_MODEL_VERSION_FIELD] = response_data[ROOF_AGE_MODEL_VERSION_FIELD]

        # Convert camelCase API columns to snake_case, matching feature_api.py pattern
        gdf.columns = [stringcase.snakecase(c) for c in gdf.columns]

        # Drop redundant/internal columns (originals already mapped above)
        gdf = gdf.drop(columns=[c for c in ["timeline", "hilbert_id", "area"] if c in gdf.columns])

        return gdf

    def _fetch_all_pages(
        self,
        url: str,
        params: Dict,
        aoi: Optional[Polygon] = None,
        address: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
        until_as_of_date: Optional[str] = None,
        since_as_of_date: Optional[str] = None,
    ) -> Dict:
        """
        Fetch all pages of results from the Roof Age API, handling pagination.

        Makes repeated requests until no nextCursor is returned, accumulating
        all features into a single merged response.

        Args:
            url: API endpoint URL
            params: Query parameters (including API key)
            aoi: Polygon geometry (mutually exclusive with address)
            address: Address dict
            limit: Maximum features per page
            until_as_of_date: Optional ``YYYY-MM-DD`` upper-bound cutoff sent on every page request.
            since_as_of_date: Optional ``YYYY-MM-DD`` lower-bound cutoff sent on every page request.

        Returns:
            Merged response dict with all features from all pages
        """
        all_features = []
        cursor = None
        page_count = 0
        first_page_metadata = None

        while True:
            page_count += 1
            payload = self._build_request_payload(
                aoi=aoi,
                address=address,
                cursor=cursor,
                limit=limit,
                until_as_of_date=until_as_of_date,
                since_as_of_date=since_as_of_date,
            )

            with self._session_scope() as session:
                t1 = time.monotonic()
                response = session.post(url, json=payload, params=params, timeout=session._timeout)
                response_time_ms = (time.monotonic() - t1) * 1e3

                # Record per-request latency for statistics
                self._latencies.append(response_time_ms)

                # Debug log
                logger.debug(f"Roof Age API response in {response_time_ms:.1f}ms (page {page_count})")

                if not response.ok:
                    error_msg = f"Failed to get roof age data (page {page_count})"
                    raise RoofAgeAPIError(response, self._clean_api_key(url), message=error_msg)

                response_data = response.json()

            # Accumulate features
            features = response_data.get("features", [])
            all_features.extend(features)

            # Capture all top-level metadata from first page (except features and nextCursor)
            if first_page_metadata is None:
                first_page_metadata = {
                    k: v for k, v in response_data.items() if k not in ("features", ROOF_AGE_NEXT_CURSOR_FIELD)
                }

            # Check for next page
            next_cursor = response_data.get(ROOF_AGE_NEXT_CURSOR_FIELD)
            if next_cursor:
                logger.debug(
                    f"Fetching page {page_count + 1} (got {len(features)} features, total so far: {len(all_features)})"
                )
                cursor = next_cursor
            else:
                # No more pages
                if page_count > 1:
                    logger.debug(f"Pagination complete: {page_count} pages, {len(all_features)} total features")
                break

        # Build merged response: start with first page metadata, then set merged features
        merged_response = first_page_metadata.copy() if first_page_metadata else {}
        merged_response["features"] = all_features

        return merged_response

    def get_roof_age_by_aoi(
        self,
        aoi: Polygon,
        aoi_id: str,
        file_format: str = "json",
        limit: Optional[int] = None,
        until_as_of_date: Optional[str] = None,
        since_as_of_date: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """
        Get roof age data for an area of interest.

        Automatically handles pagination if more than `limit` features exist.

        Args:
            aoi: Polygon defining the area of interest (in EPSG:4326)
            aoi_id: Unique identifier for this AOI
            file_format: Response format ("json" or "geojson")
            limit: Maximum features per page (default: API default of 1000)
            until_as_of_date: Per-call ``YYYY-MM-DD`` upper-bound cutoff override. Falls back to
                ``self.until_as_of_date`` if not given.
            since_as_of_date: Per-call ``YYYY-MM-DD`` lower-bound cutoff override. Falls back to
                ``self.since_as_of_date`` if not given.

        Returns:
            GeoDataFrame with roof features, installation dates, and metadata

        Raises:
            RoofAgeAPIError: If the API request fails
        """
        effective_until = until_as_of_date if until_as_of_date is not None else self.until_as_of_date
        effective_since = since_as_of_date if since_as_of_date is not None else self.since_as_of_date
        cache_key = self._build_cache_key(aoi=aoi)

        # Check cache first (caches the full merged result)
        cached_response = self._load_from_cache(
            cache_key, until_as_of_date=effective_until, since_as_of_date=effective_since
        )
        if cached_response is not None:
            self._cache_hits += 1
            logger.debug(f"Using cached roof age data for {aoi_id}")
            return self._parse_response(cached_response, aoi_id)

        # Fetch all pages (this is a cache miss)
        self._cache_misses += 1
        url = f"{self.base_url}.{file_format}"
        # In bearer mode the apikey param is omitted (auth is the session header).
        params = {} if self.bearer_token else {"apikey": self.api_key}
        if self.bulk_mode:
            params["bulk"] = "true"

        logger.debug(f"Requesting roof age data for {aoi_id}")
        response_data = self._fetch_all_pages(
            url=url,
            params=params,
            aoi=aoi,
            limit=limit,
            until_as_of_date=effective_until,
            since_as_of_date=effective_since,
        )

        # Cache the merged response
        self._save_to_cache(
            cache_key, response_data, until_as_of_date=effective_until, since_as_of_date=effective_since
        )

        return self._parse_response(response_data, aoi_id)

    def get_roof_age_by_address(
        self,
        address: Dict[str, str],
        aoi_id: str,
        file_format: str = "json",
        limit: Optional[int] = None,
        until_as_of_date: Optional[str] = None,
        since_as_of_date: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """
        Get roof age data for a property address.

        Automatically handles pagination if more than `limit` features exist.

        Args:
            address: Dict with keys: streetAddress, city, state, zip
                     Example: {"streetAddress": "123 Main St", "city": "Austin",
                              "state": "TX", "zip": "78701"}
            aoi_id: Unique identifier for this property
            file_format: Response format ("json" or "geojson")
            limit: Maximum features per page (default: API default of 1000)
            until_as_of_date: Per-call ``YYYY-MM-DD`` upper-bound cutoff override. Falls back to
                ``self.until_as_of_date`` if not given.
            since_as_of_date: Per-call ``YYYY-MM-DD`` lower-bound cutoff override. Falls back to
                ``self.since_as_of_date`` if not given.

        Returns:
            GeoDataFrame with roof features, installation dates, and metadata

        Raises:
            RoofAgeAPIError: If the API request fails
        """
        effective_until = until_as_of_date if until_as_of_date is not None else self.until_as_of_date
        effective_since = since_as_of_date if since_as_of_date is not None else self.since_as_of_date
        cache_key = self._build_cache_key(address=address)

        # Check cache first (caches the full merged result)
        cached_response = self._load_from_cache(
            cache_key, until_as_of_date=effective_until, since_as_of_date=effective_since
        )
        if cached_response is not None:
            self._cache_hits += 1
            logger.debug(f"Using cached roof age data for {aoi_id}")
            return self._parse_response(cached_response, aoi_id)

        # Fetch all pages (this is a cache miss)
        self._cache_misses += 1
        url = f"{self.base_url}.{file_format}"
        # In bearer mode the apikey param is omitted (auth is the session header).
        params = {} if self.bearer_token else {"apikey": self.api_key}
        if self.bulk_mode:
            params["bulk"] = "true"

        logger.debug(f"Requesting roof age data for {aoi_id} at {address.get('streetAddress', 'unknown')}")
        response_data = self._fetch_all_pages(
            url=url,
            params=params,
            address=address,
            limit=limit,
            until_as_of_date=effective_until,
            since_as_of_date=effective_since,
        )

        # Cache the merged response
        self._save_to_cache(
            cache_key, response_data, until_as_of_date=effective_until, since_as_of_date=effective_since
        )

        return self._parse_response(response_data, aoi_id)

    def get_roof_age_bulk(
        self,
        aoi_gdf: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get roof age data for multiple AOIs in parallel.

        Supports both geometry-based and address-based queries. Automatically detects
        the input mode based on available columns.

        Args:
            aoi_gdf: GeoDataFrame with AOIs to query (must have aoi_id index)
                     For geometry mode: must have 'geometry' column
                     For address mode: must have streetAddress, city, state, zip columns

        Returns:
            Tuple of (roofs_gdf, metadata_df, errors_df):
                - roofs_gdf: GeoDataFrame with all roof features
                - metadata_df: DataFrame with query metadata (resource IDs, etc.)
                - errors_df: DataFrame with failed queries

        Raises:
            ValueError: If required columns are missing from aoi_gdf
        """
        if not isinstance(aoi_gdf.index, pd.Index) or aoi_gdf.index.name != AOI_ID_COLUMN_NAME:
            raise ValueError(f"aoi_gdf must have '{AOI_ID_COLUMN_NAME}' as index")

        # Detect input mode - check for address fields and geometry
        has_address_fields = set(aoi_gdf.columns.tolist()).intersection(set(ADDRESS_FIELDS)) == set(ADDRESS_FIELDS)
        has_geom = "geometry" in aoi_gdf.columns

        if not has_geom and not has_address_fields:
            raise ValueError(
                "aoi_gdf must have either a 'geometry' column (for AOI-based queries) "
                "or address columns (streetAddress, city, state, zip) for address-based queries"
            )

        mode = "address" if has_address_fields and not has_geom else "geometry"
        logger.debug(f"Getting roof age data for {len(aoi_gdf)} AOIs using {self.threads} threads ({mode} mode)")

        roofs_list = []
        metadata_list = []
        errors_list = []

        def process_aoi(row):
            """Process a single AOI row (handles both geometry and address modes)"""
            aoi_id = row.name

            # Per-AOI 'until' / 'since' columns override the bulk cutoffs when present and non-empty.
            # Mirrors Feature API behaviour in feature_api.py. Callers must supply YYYY-MM-DD strings —
            # the AOI-load step in the exporter validates dtype, so a non-string here would indicate a
            # bypass of that validation rather than legitimate input.
            until_for_row = self.until_as_of_date
            if UNTIL_COL_NAME in row and isinstance(row[UNTIL_COL_NAME], str) and row[UNTIL_COL_NAME]:
                until_for_row = row[UNTIL_COL_NAME]

            since_for_row = self.since_as_of_date
            if SINCE_COL_NAME in row and isinstance(row[SINCE_COL_NAME], str) and row[SINCE_COL_NAME]:
                since_for_row = row[SINCE_COL_NAME]

            try:
                if has_geom:
                    # Geometry-based query
                    roofs_gdf = self.get_roof_age_by_aoi(
                        row.geometry,
                        aoi_id,
                        until_as_of_date=until_for_row,
                        since_as_of_date=since_for_row,
                    )
                else:
                    # Address-based query - convert all values to strings for JSON serialization
                    address = {f: str(row[f]) for f in ADDRESS_FIELDS}
                    roofs_gdf = self.get_roof_age_by_address(
                        address,
                        aoi_id,
                        until_as_of_date=until_for_row,
                        since_as_of_date=since_for_row,
                    )

                # Extract metadata
                metadata = {
                    AOI_ID_COLUMN_NAME: aoi_id,
                }
                # Only extract resourceId if we have roofs (DataFrame may be empty after filtering)
                if len(roofs_gdf) > 0 and "resource_id" in roofs_gdf.columns:
                    metadata["resource_id"] = roofs_gdf["resource_id"].iloc[0]

                return ("success", roofs_gdf, metadata, None)
            except Exception as e:
                error_info = {
                    AOI_ID_COLUMN_NAME: aoi_id,
                    "status_code": getattr(e, "status_code", -1),
                    "message": str(e),
                }
                return ("error", None, None, error_info)

        # Process in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = [executor.submit(process_aoi, row) for _, row in aoi_gdf.iterrows()]

            for future in concurrent.futures.as_completed(futures):
                status, roofs_gdf, metadata, error_info = future.result()

                if status == "success":
                    if roofs_gdf is not None and len(roofs_gdf) > 0:
                        roofs_list.append(roofs_gdf)
                    if metadata is not None:
                        metadata_list.append(metadata)
                else:
                    errors_list.append(error_info)

                # Increment progress counter for completed request
                self._increment_progress()

        # Combine results
        if roofs_list:
            roofs_gdf = gpd.GeoDataFrame(pd.concat(roofs_list, ignore_index=False), crs=API_CRS)
        else:
            roofs_gdf = gpd.GeoDataFrame(columns=[AOI_ID_COLUMN_NAME, "geometry"], crs=API_CRS)

        metadata_df = pd.DataFrame(metadata_list)
        if len(metadata_df) > 0:
            metadata_df = metadata_df.set_index(AOI_ID_COLUMN_NAME)

        errors_df = pd.DataFrame(errors_list)
        if len(errors_df) > 0:
            errors_df = errors_df.set_index(AOI_ID_COLUMN_NAME)

        # Calculate error rate for logging
        error_pct = (len(errors_df) / len(aoi_gdf)) * 100 if len(aoi_gdf) > 0 else 0

        logger.debug(
            f"Roof age bulk query complete: {len(roofs_gdf)} roofs found, "
            f"{len(errors_df)} errors ({error_pct:.1f}%)"
        )

        return roofs_gdf, metadata_df, errors_df
