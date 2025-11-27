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
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, shape

from nmaipy import log
from nmaipy.api_common import BaseApiClient, RoofAgeAPIError
from nmaipy.constants import (
    ADDRESS_FIELDS,
    AOI_ID_COLUMN_NAME,
    API_CRS,
    FEATURE_CLASS_DESCRIPTIONS,
    ROOF_AGE_AFTER_INSTALLATION_CAPTURE_DATE_FIELD,
    ROOF_AGE_AREA_FIELD,
    ROOF_AGE_EVIDENCE_TYPE_DESC_FIELD,
    ROOF_AGE_EVIDENCE_TYPE_FIELD,
    ROOF_AGE_HILBERT_ID_FIELD,
    ROOF_AGE_INSTALLATION_DATE_FIELD,
    ROOF_AGE_MAX_CAPTURE_DATE_FIELD,
    ROOF_AGE_MIN_CAPTURE_DATE_FIELD,
    ROOF_AGE_NUM_CAPTURES_FIELD,
    ROOF_AGE_RESOURCE_ENDPOINT,
    ROOF_AGE_RESOURCE_ID_FIELD,
    ROOF_AGE_TIMELINE_FIELD,
    ROOF_AGE_TRUST_SCORE_FIELD,
    ROOF_AGE_UNTIL_DATE_FIELD,
    ROOF_AGE_URL_ROOT,
    ROOF_INSTANCE_CLASS_ID,
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
        """
        super().__init__(
            api_key=api_key,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            compress_cache=compress_cache,
            threads=threads,
        )

        # Configure API endpoints
        if url_root is None:
            url_root = ROOF_AGE_URL_ROOT
        self.base_url = f"https://{url_root}/{ROOF_AGE_RESOURCE_ENDPOINT}"

        logger.debug(f"Initialized RoofAgeApi with base_url: {self._clean_api_key(self.base_url)}")

    def _build_request_payload(
        self,
        aoi: Optional[Polygon] = None,
        address: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Build the request payload for the Roof Age API.

        Args:
            aoi: Polygon geometry (mutually exclusive with address)
            address: Address dict with keys: streetAddress, city, state, zip

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
            payload["address"] = address

        return payload

    def _build_cache_key(
        self,
        aoi: Optional[Polygon] = None,
        address: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Build a cache key for a roof age request.

        Args:
            aoi: Polygon geometry
            address: Address dict

        Returns:
            Cache key string
        """
        if aoi is not None:
            # Use WKT representation for cache key
            return f"roofage_aoi_{aoi.wkt}"
        else:
            # Build key from address fields
            addr_str = "_".join([str(address[f]) for f in ADDRESS_FIELDS])
            return f"roofage_address_{addr_str}"

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
            # Return empty GeoDataFrame with expected columns
            return gpd.GeoDataFrame(
                columns=[AOI_ID_COLUMN_NAME, "class_id", "description", "geometry"] + [
                    ROOF_AGE_INSTALLATION_DATE_FIELD,
                    ROOF_AGE_TRUST_SCORE_FIELD,
                    ROOF_AGE_AREA_FIELD,
                ],
                crs=API_CRS
            )

        # Parse features into records
        records = []
        geometries = []
        for feature in features:
            # Extract geometry
            geom = shape(feature["geometry"])
            geometries.append(geom)

            # Extract properties and add aoi_id
            props = feature.get("properties", {})
            props[AOI_ID_COLUMN_NAME] = aoi_id

            # Add class_id and description for unified feature model
            # This allows roof instances to be treated like any other feature class
            # TODO: This is a temporary measure only for Roof Age API responses
            props["class_id"] = ROOF_INSTANCE_CLASS_ID
            props["description"] = FEATURE_CLASS_DESCRIPTIONS.get(ROOF_INSTANCE_CLASS_ID, "Roof Instance")

            # Serialize timeline to JSON string if present (for compatibility with CSV/Parquet)
            if ROOF_AGE_TIMELINE_FIELD in props:
                props[ROOF_AGE_TIMELINE_FIELD] = json.dumps(props[ROOF_AGE_TIMELINE_FIELD])

            records.append(props)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=API_CRS)

        # Add metadata from top level
        if ROOF_AGE_RESOURCE_ID_FIELD in response_data:
            gdf[ROOF_AGE_RESOURCE_ID_FIELD] = response_data[ROOF_AGE_RESOURCE_ID_FIELD]

        return gdf

    def get_roof_age_by_aoi(
        self,
        aoi: Polygon,
        aoi_id: str,
        file_format: str = "json"
    ) -> gpd.GeoDataFrame:
        """
        Get roof age data for an area of interest.

        Args:
            aoi: Polygon defining the area of interest (in EPSG:4326)
            aoi_id: Unique identifier for this AOI
            file_format: Response format ("json" or "geojson")

        Returns:
            GeoDataFrame with roof features, installation dates, and metadata

        Raises:
            RoofAgeAPIError: If the API request fails
        """
        # Build request
        payload = self._build_request_payload(aoi=aoi)
        cache_key = self._build_cache_key(aoi=aoi)

        # Check cache first
        cached_response = self._load_from_cache(cache_key)
        if cached_response is not None:
            logger.debug(f"Using cached roof age data for {aoi_id}")
            return self._parse_response(cached_response, aoi_id)

        # Make API request
        url = f"{self.base_url}.{file_format}"
        params = {"apikey": self.api_key}

        logger.debug(f"Requesting roof age data for {aoi_id}")
        with self._session_scope() as session:
            response = session.post(url, json=payload, params=params, timeout=session._timeout)

            if not response.ok:
                error_msg = f"Failed to get roof age data for {aoi_id}"
                raise RoofAgeAPIError(response, self._clean_api_key(url), message=error_msg)

            response_data = response.json()

        # Cache the response
        self._save_to_cache(cache_key, response_data)

        return self._parse_response(response_data, aoi_id)

    def get_roof_age_by_address(
        self,
        address: Dict[str, str],
        aoi_id: str,
        file_format: str = "json"
    ) -> gpd.GeoDataFrame:
        """
        Get roof age data for a property address.

        Args:
            address: Dict with keys: streetAddress, city, state, zip
                     Example: {"streetAddress": "123 Main St", "city": "Austin",
                              "state": "TX", "zip": "78701"}
            aoi_id: Unique identifier for this property
            file_format: Response format ("json" or "geojson")

        Returns:
            GeoDataFrame with roof features, installation dates, and metadata

        Raises:
            RoofAgeAPIError: If the API request fails
        """
        # Build request
        payload = self._build_request_payload(address=address)
        cache_key = self._build_cache_key(address=address)

        # Check cache first
        cached_response = self._load_from_cache(cache_key)
        if cached_response is not None:
            logger.debug(f"Using cached roof age data for {aoi_id}")
            return self._parse_response(cached_response, aoi_id)

        # Make API request
        url = f"{self.base_url}.{file_format}"
        params = {"apikey": self.api_key}

        logger.debug(f"Requesting roof age data for {aoi_id} at {address.get('streetAddress', 'unknown')}")
        with self._session_scope() as session:
            response = session.post(url, json=payload, params=params, timeout=session._timeout)

            if not response.ok:
                error_msg = f"Failed to get roof age data for {aoi_id}"
                raise RoofAgeAPIError(response, self._clean_api_key(url), message=error_msg)

            response_data = response.json()

        # Cache the response
        self._save_to_cache(cache_key, response_data)

        return self._parse_response(response_data, aoi_id)

    def get_roof_age_bulk(
        self,
        aoi_gdf: gpd.GeoDataFrame,
        max_allowed_error_pct: float = 10.0
    ) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get roof age data for multiple AOIs in parallel.

        Supports both geometry-based and address-based queries. Automatically detects
        the input mode based on available columns.

        Args:
            aoi_gdf: GeoDataFrame with AOIs to query (must have aoi_id index)
                     For geometry mode: must have 'geometry' column
                     For address mode: must have streetAddress, city, state, zip columns
            max_allowed_error_pct: Maximum percentage of AOIs that can fail before raising error

        Returns:
            Tuple of (roofs_gdf, metadata_df, errors_df):
                - roofs_gdf: GeoDataFrame with all roof features
                - metadata_df: DataFrame with query metadata (resource IDs, etc.)
                - errors_df: DataFrame with failed queries

        Raises:
            ValueError: If error rate exceeds max_allowed_error_pct or required columns are missing
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
        logger.info(f"Getting roof age data for {len(aoi_gdf)} AOIs using {self.threads} threads ({mode} mode)")

        roofs_list = []
        metadata_list = []
        errors_list = []

        def process_aoi(row):
            """Process a single AOI row (handles both geometry and address modes)"""
            aoi_id = row.name

            try:
                if has_geom:
                    # Geometry-based query
                    roofs_gdf = self.get_roof_age_by_aoi(row.geometry, aoi_id)
                else:
                    # Address-based query
                    address = {f: row[f] for f in ADDRESS_FIELDS}
                    roofs_gdf = self.get_roof_age_by_address(address, aoi_id)

                # Extract metadata
                metadata = {
                    AOI_ID_COLUMN_NAME: aoi_id,
                }
                if ROOF_AGE_RESOURCE_ID_FIELD in roofs_gdf.columns:
                    metadata[ROOF_AGE_RESOURCE_ID_FIELD] = roofs_gdf[ROOF_AGE_RESOURCE_ID_FIELD].iloc[0]

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

        # Check error rate
        error_pct = (len(errors_df) / len(aoi_gdf)) * 100 if len(aoi_gdf) > 0 else 0
        if error_pct > max_allowed_error_pct:
            raise ValueError(
                f"Error rate {error_pct:.1f}% exceeds maximum allowed {max_allowed_error_pct}%. "
                f"{len(errors_df)} of {len(aoi_gdf)} requests failed."
            )

        logger.info(
            f"Roof age bulk query complete: {len(roofs_gdf)} roofs found, "
            f"{len(errors_df)} errors ({error_pct:.1f}%)"
        )

        return roofs_gdf, metadata_df, errors_df
