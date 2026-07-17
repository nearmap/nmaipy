"""
Client for the Nearmap Damage Conflation API (ai/damage/v2).

The Damage Conflation API provides event-scoped, building-level damage ratings for
a catastrophe event (e.g. a hurricane or wildfire). Where the AI Feature API returns
per-survey damage classifications in near real-time, the conflation API fuses every
capture across an event into a single authoritative rating per building — the
highest-confidence assessment available across the event lifecycle, even when later
captures are occluded by smoke, cloud, or debris.

Querying is per-AOI (mirrors the Roof Age API): POST an area of interest (or address)
to ``/events/{event_id}/latest.geojson`` and the API returns a GeoJSON FeatureCollection
of every building intersecting the AOI, each with ``damage.event`` (post-event) and
``damage.preEvent`` (baseline) ratings. Large AOIs (mesh block → full event boundary)
are handled by pagination (``cursor``/``nextCursor``), not gridding — this client
extends ``BaseApiClient``, not ``GriddedApiClient``.

Example usage:
    ```python
    from nmaipy.damage_conflation_api import DamageConflationApi
    from shapely.geometry import Polygon

    api = DamageConflationApi(event_id="2f510853-5d55-50f4-9102-2c02de08190e")
    gdf = api.get_damage_by_aoi(Polygon([...]), aoi_id="parcel_123")
    ```

Contract confirmed against SwaggerHub ``nearmap/ai-conflation-damage/2.0.0`` and the
internal pyaiutils wrapper.
"""

import concurrent.futures
import hashlib
import time
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, mapping, shape

from nmaipy import log, storage
from nmaipy.api_common import BaseApiClient, DamageConflationAPIError
from nmaipy.constants import (
    ADDRESS_FIELDS,
    AOI_ID_COLUMN_NAME,
    API_CRS,
    DAMAGE_CONFLATION_DEFAULT_PAGE_LIMIT,
    DAMAGE_CONFLATION_EVENTS_PATH,
    DAMAGE_CONFLATION_MAX_PAGES,
    DAMAGE_CONFLATION_NEXT_CURSOR_FIELD,
    DAMAGE_CONFLATION_URL_ROOT,
)
from nmaipy.feature_attributes import flatten_conflated_damage_attributes

logger = log.get_logger()

# Top-level response metadata copied onto every parsed row. (API key, snake column.)
_RESPONSE_METADATA_FIELDS = (
    ("resourceId", "resource_id"),
    ("eventUuid", "event_uuid"),
    ("eventName", "event_name"),
    ("modelVersion", "model_version"),
    ("presentationVersion", "presentation_version"),
    ("geomatchedAddress", "geomatched_address"),
)

# Columns of an empty (no-feature) parse result, so concat/rollup stay well-formed.
_EMPTY_COLUMNS = [AOI_ID_COLUMN_NAME, "feature_id", "geometry", "damage_event_rating", "area_sqm"]

# Columns that are string/date-valued but can be entirely absent from a given response
# (e.g. damage_event_latest_capture_date is only ever populated on the preEvent block;
# geomatched_address is address-mode only) — cast explicitly to a nullable string dtype
# so an all-None column doesn't get inferred as pyarrow's untyped `null`, which some
# schema-strict parquet readers (Athena, older Spark/DuckDB) reject when unifying chunks.
_NULLABLE_STRING_COLUMNS = (
    "damage_event_latest_capture_date",
    "damage_pre_event_latest_capture_date",
    "geomatched_address",
)


class DamageConflationApi(BaseApiClient):
    """
    Client for the Nearmap Damage Conflation API.

    Provides methods to query event-scoped conflated damage by AOI or address.
    Inherits session management, caching, retry, and progress tracking from
    BaseApiClient. Scoped to a single ``event_id``.
    """

    def __init__(
        self,
        event_id: str,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        overwrite_cache: Optional[bool] = False,
        compress_cache: Optional[bool] = False,
        threads: Optional[int] = 10,
        url_root: Optional[str] = None,
        country: str = "us",
        progress_counters: Optional[dict] = None,
        bearer_token: Optional[str] = None,
    ):
        """
        Initialize Damage Conflation API client.

        Args:
            event_id: UUID of the catastrophe event to query (required). This is the
                only scoping axis — there is no resource/version/cutoff axis (cf. Roof
                Age). The ``latest`` conflation for the event is always returned.
            api_key: Nearmap API key. If not provided, reads from API_KEY env var.
            cache_dir: Directory to use as a payload cache.
            overwrite_cache: Whether to overwrite cached values.
            compress_cache: Whether to gzip cache files.
            threads: Number of threads for concurrent bulk execution.
            url_root: Override the default API root URL (for testing).
            country: Country code for address queries (e.g. 'us'). Also drives output
                area units downstream.
            progress_counters: Optional dict with 'total', 'completed', 'lock' for
                cross-process progress tracking.
            bearer_token: Short-lived Nearmap identity JWT used instead of an API key
                (see BaseApiClient for lifetime caveats)
        """
        super().__init__(
            api_key=api_key,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            compress_cache=compress_cache,
            threads=threads,
            bearer_token=bearer_token,
        )

        if not event_id:
            raise ValueError("event_id is required for the Damage Conflation API")

        self.event_id = event_id
        self.country = country.upper()
        self.progress_counters = progress_counters

        if url_root is None:
            url_root = DAMAGE_CONFLATION_URL_ROOT
        # POST {base_url}.geojson -> /ai/damage/v2/events/{event_id}/latest.geojson
        # base_url contains no secret: the API key is only ever sent as a `?apikey=`
        # query param at request time, never embedded here. So it is logged directly
        # rather than routed through _clean_api_key — doing so would feed an api-key
        # named sanitizer into the log, which CodeQL flags as clear-text logging of a
        # secret (a false positive, since there is no secret to clean).
        self.base_url = f"https://{url_root}/{DAMAGE_CONFLATION_EVENTS_PATH}/{event_id}/latest"

        logger.debug(f"Initialized DamageConflationApi with base_url: {self.base_url}, event_id: {event_id}")

    # ------------------------------------------------------------------ caching
    def _qualifier_subdir(self) -> str:
        """Cache sub-directory scoping requests by event: ``damageconflation/<event_id>``.

        Keeping the event in the *path* (not the cache key) keeps the cache
        human-inspectable and lets two events for the same AOI resolve to different
        files. Mirrors the Roof Age client's qualifier layout.
        """
        return storage.join_path("damageconflation", self.event_id)

    def _get_cache_path(self, cache_key: str) -> str:
        """
        Cache file path for a key, under ``<cache_dir>/damageconflation/<event_id>/``.

        - address keys: ``.../<country>/<state>/<city>/<zip>/<hash>.json``
        - AOI keys:     ``.../<wkt-hash>.json``
        """
        if self.cache_dir is None:
            raise ValueError("Cache directory not configured")

        extension = ".json.gz" if self.compress_cache else ".json"
        qualifier_dir = storage.join_path(self.cache_dir, self._qualifier_subdir())

        if cache_key.startswith("damageconflation_address_"):
            address_path = cache_key[len("damageconflation_address_") :]
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

        storage.ensure_directory(qualifier_dir)
        key_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        return storage.join_path(qualifier_dir, f"{key_hash}{extension}")

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load a cached response via the event-scoped path layout."""
        if self.cache_dir is None or self.overwrite_cache:
            return None
        cache_path = self._get_cache_path(cache_key)
        if not storage.file_exists(cache_path):
            return None
        try:
            return storage.read_json(cache_path, compressed=self.compress_cache)
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        """Save a response via the event-scoped path layout."""
        if self.cache_dir is None:
            return
        cache_path = self._get_cache_path(cache_key)
        try:
            storage.write_json(cache_path, data, compressed=self.compress_cache)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def _build_cache_key(self, aoi: Optional[Polygon] = None, address: Optional[Dict[str, str]] = None) -> str:
        """Cache key identifying the request geometry/address (event lives in the path)."""
        if aoi is not None:
            return f"damageconflation_aoi_{aoi.wkt}"
        address_path = (
            f"{self.country}/"
            f"{address.get('state', '_')}/"
            f"{address.get('city', '_')}/"
            f"{address.get('zip', '_')}/"
            f"{address.get('streetAddress', '_')}"
        )
        return f"damageconflation_address_{address_path}"

    # ------------------------------------------------------------------ requests
    def _build_request_payload(
        self,
        aoi: Optional[Polygon] = None,
        address: Optional[Dict[str, str]] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict:
        """
        Build the JSON request body for the Damage Conflation API.

        Body is ``{"aoi": <GeoJSON geometry>}`` XOR ``{"address": {...}}``, plus an
        optional ``cursor`` (pagination) and ``limit``. This is the single point that
        encodes the HTTP request contract.

        Raises:
            ValueError: If neither or both of aoi and address are provided.
        """
        if aoi is not None and address is not None:
            raise ValueError("Cannot specify both aoi and address")
        if aoi is None and address is None:
            raise ValueError("Must specify either aoi or address")

        payload: Dict = {}
        if aoi is not None:
            payload["aoi"] = mapping(aoi)
        else:
            for field in ADDRESS_FIELDS:
                if field not in address:
                    raise ValueError(f"Address missing required field: {field}")
            payload["address"] = {**address, "country": self.country}

        if cursor is not None:
            payload["cursor"] = cursor
        if limit is not None:
            payload["limit"] = limit

        return payload

    def _fetch_all_pages(
        self,
        url: str,
        params: Dict,
        aoi: Optional[Polygon] = None,
        address: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
    ) -> Dict:
        """
        Fetch all pages for a query, following ``nextCursor`` until exhausted.

        Returns a merged response dict (first-page top-level metadata + all features).
        Raises DamageConflationAPIError on HTTP error or an error body.
        """
        all_features: List[dict] = []
        cursor = None
        page_count = 0
        first_page_metadata = None

        while True:
            page_count += 1
            payload = self._build_request_payload(aoi=aoi, address=address, cursor=cursor, limit=limit)

            with self._session_scope() as session:
                t1 = time.monotonic()
                response = session.post(url, json=payload, params=params, timeout=session._timeout)
                self._latencies.append((time.monotonic() - t1) * 1e3)

                if not response.ok:
                    raise DamageConflationAPIError(
                        response,
                        self._clean_api_key(url),
                        message=f"Failed to get damage conflation data (page {page_count})",
                    )

                response_data = response.json()

            # Some platform errors arrive as HTTP 200 with an error body — surface them.
            if isinstance(response_data, dict) and "error" in response_data:
                raise DamageConflationAPIError(
                    None,
                    self._clean_api_key(url),
                    message=f"{response_data.get('error')} (code: {response_data.get('code', 'UNKNOWN')})",
                )

            all_features.extend(response_data.get("features", []))

            if first_page_metadata is None:
                first_page_metadata = {
                    k: v for k, v in response_data.items() if k not in ("features", DAMAGE_CONFLATION_NEXT_CURSOR_FIELD)
                }

            cursor = response_data.get(DAMAGE_CONFLATION_NEXT_CURSOR_FIELD)
            if cursor:
                if page_count >= DAMAGE_CONFLATION_MAX_PAGES:
                    raise DamageConflationAPIError(
                        None,
                        self._clean_api_key(url),
                        message=(
                            f"Pagination exceeded {DAMAGE_CONFLATION_MAX_PAGES} pages "
                            f"({len(all_features)} features so far) without exhausting nextCursor "
                            "— aborting rather than looping indefinitely"
                        ),
                    )
                logger.debug(f"Fetching page {page_count + 1} (total features so far: {len(all_features)})")
            else:
                if page_count > 1:
                    logger.debug(f"Pagination complete: {page_count} pages, {len(all_features)} features")
                break

        merged = first_page_metadata.copy() if first_page_metadata else {}
        merged["features"] = all_features
        return merged

    def _parse_response(self, response_data: Dict, aoi_id: str) -> gpd.GeoDataFrame:
        """
        Parse a (merged) Damage Conflation response into a GeoDataFrame.

        One row per building feature, with the flattened ``damage_*`` columns, the
        stable top-level ``id`` as ``feature_id``, the ``aoi_id`` tag, and the
        event-level metadata columns copied onto each row.
        """
        if response_data.get("type") != "FeatureCollection":
            logger.warning(f"Unexpected response type: {response_data.get('type')}")

        features = response_data.get("features", [])
        if not features:
            logger.debug(f"No damage features found for {aoi_id}")
            return gpd.GeoDataFrame(columns=_EMPTY_COLUMNS, geometry="geometry", crs=API_CRS)

        metadata = {col: response_data.get(api_key) for api_key, col in _RESPONSE_METADATA_FIELDS}

        records = []
        geometries = []
        for feature in features:
            geometry = feature.get("geometry")
            if geometry is None:
                raise DamageConflationAPIError(
                    None,
                    self._clean_api_key(self.base_url),
                    message=f"Feature missing 'geometry' for {aoi_id} (feature id: {feature.get('id')})",
                )
            geometries.append(shape(geometry))
            record = flatten_conflated_damage_attributes(feature.get("properties", {}))
            record[AOI_ID_COLUMN_NAME] = aoi_id
            # The top-level GeoJSON id is the stable, globally-unique building id
            # (cf. properties.hilbertId, a spatial index). Use it as feature_id.
            record["feature_id"] = feature.get("id")
            record.update(metadata)
            records.append(record)

        gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=API_CRS)
        for col in _NULLABLE_STRING_COLUMNS:
            if col in gdf.columns:
                gdf[col] = gdf[col].astype("string")
        return gdf

    # ------------------------------------------------------------------ queries
    def get_damage_by_aoi(
        self,
        aoi: Polygon,
        aoi_id: str,
        file_format: str = "geojson",
        limit: Optional[int] = DAMAGE_CONFLATION_DEFAULT_PAGE_LIMIT,
    ) -> gpd.GeoDataFrame:
        """
        Get conflated damage for an area of interest, handling pagination.

        Args:
            aoi: Polygon defining the area of interest (EPSG:4326).
            aoi_id: Unique identifier for this AOI.
            file_format: Response format suffix (default "geojson").
            limit: Max features per page (default 1000).

        Returns:
            GeoDataFrame with one row per building, flattened damage attributes.

        Raises:
            DamageConflationAPIError: If the API request fails.
        """
        cache_key = self._build_cache_key(aoi=aoi)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(f"Using cached damage data for {aoi_id}")
            return self._parse_response(cached, aoi_id)

        self._cache_misses += 1
        url = f"{self.base_url}.{file_format}"
        # In bearer mode the apikey param is omitted (auth is the session header).
        params = {} if self.bearer_token else {"apikey": self.api_key}
        response_data = self._fetch_all_pages(url=url, params=params, aoi=aoi, limit=limit)
        self._save_to_cache(cache_key, response_data)
        return self._parse_response(response_data, aoi_id)

    def get_damage_by_address(
        self,
        address: Dict[str, str],
        aoi_id: str,
        file_format: str = "geojson",
        limit: Optional[int] = DAMAGE_CONFLATION_DEFAULT_PAGE_LIMIT,
    ) -> gpd.GeoDataFrame:
        """
        Get conflated damage for a property address, handling pagination.

        Args:
            address: Dict with keys streetAddress, city, state, zip.
            aoi_id: Unique identifier for this property.
            file_format: Response format suffix (default "geojson").
            limit: Max features per page (default 1000).

        Returns:
            GeoDataFrame with one row per building, flattened damage attributes.
        """
        cache_key = self._build_cache_key(address=address)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(f"Using cached damage data for {aoi_id}")
            return self._parse_response(cached, aoi_id)

        self._cache_misses += 1
        url = f"{self.base_url}.{file_format}"
        # In bearer mode the apikey param is omitted (auth is the session header).
        params = {} if self.bearer_token else {"apikey": self.api_key}
        response_data = self._fetch_all_pages(url=url, params=params, address=address, limit=limit)
        self._save_to_cache(cache_key, response_data)
        return self._parse_response(response_data, aoi_id)

    def get_damage_bulk(
        self,
        aoi_gdf: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get conflated damage for multiple AOIs in parallel.

        Auto-detects geometry vs address mode from the available columns, mirroring
        ``RoofAgeApi.get_roof_age_bulk``.

        Args:
            aoi_gdf: GeoDataFrame indexed by aoi_id. Geometry mode needs a 'geometry'
                column; address mode needs streetAddress, city, state, zip columns.

        Returns:
            Tuple of (features_gdf, metadata_df, errors_df).
        """
        if not isinstance(aoi_gdf.index, pd.Index) or aoi_gdf.index.name != AOI_ID_COLUMN_NAME:
            raise ValueError(f"aoi_gdf must have '{AOI_ID_COLUMN_NAME}' as index")

        has_address_fields = set(aoi_gdf.columns.tolist()).intersection(set(ADDRESS_FIELDS)) == set(ADDRESS_FIELDS)
        has_geom = "geometry" in aoi_gdf.columns
        if not has_geom and not has_address_fields:
            raise ValueError(
                "aoi_gdf must have either a 'geometry' column (for AOI-based queries) "
                "or address columns (streetAddress, city, state, zip) for address-based queries"
            )

        mode = "address" if has_address_fields and not has_geom else "geometry"
        logger.debug(f"Getting damage data for {len(aoi_gdf)} AOIs using {self.threads} threads ({mode} mode)")

        features_list = []
        metadata_list = []
        errors_list = []

        def process_aoi(row):
            aoi_id = row.name
            try:
                if has_geom:
                    features_gdf = self.get_damage_by_aoi(row.geometry, aoi_id)
                else:
                    address = {f: str(row[f]) for f in ADDRESS_FIELDS}
                    features_gdf = self.get_damage_by_address(address, aoi_id)

                # One metadata row per successful query (even when zero buildings) so the
                # exporter can tell "queried, empty" from "errored". Carry event metadata
                # when present.
                metadata = {AOI_ID_COLUMN_NAME: aoi_id}
                if len(features_gdf) > 0:
                    for _, col in _RESPONSE_METADATA_FIELDS:
                        if col in features_gdf.columns:
                            metadata[col] = features_gdf[col].iloc[0]
                return ("success", features_gdf, metadata, None)
            except Exception as e:
                error_info = {
                    AOI_ID_COLUMN_NAME: aoi_id,
                    "status_code": getattr(e, "status_code", -1),
                    "message": str(e),
                }
                return ("error", None, None, error_info)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = [executor.submit(process_aoi, row) for _, row in aoi_gdf.iterrows()]
            for future in concurrent.futures.as_completed(futures):
                status, features_gdf, metadata, error_info = future.result()
                if status == "success":
                    if features_gdf is not None and len(features_gdf) > 0:
                        features_list.append(features_gdf)
                    if metadata is not None:
                        metadata_list.append(metadata)
                else:
                    errors_list.append(error_info)
                self._increment_progress()

        if features_list:
            features_gdf = gpd.GeoDataFrame(pd.concat(features_list, ignore_index=False), crs=API_CRS)
        else:
            features_gdf = gpd.GeoDataFrame(columns=_EMPTY_COLUMNS, geometry="geometry", crs=API_CRS)

        metadata_df = pd.DataFrame(metadata_list)
        if len(metadata_df) > 0:
            metadata_df = metadata_df.set_index(AOI_ID_COLUMN_NAME)

        errors_df = pd.DataFrame(errors_list)
        if len(errors_df) > 0:
            errors_df = errors_df.set_index(AOI_ID_COLUMN_NAME)

        error_pct = (len(errors_df) / len(aoi_gdf)) * 100 if len(aoi_gdf) > 0 else 0
        logger.debug(
            f"Damage conflation bulk query complete: {len(features_gdf)} buildings found, "
            f"{len(errors_df)} errors ({error_pct:.1f}%)"
        )

        return features_gdf, metadata_df, errors_df
