import argparse
import concurrent.futures
import gc
import json
import logging
import multiprocessing
import os
import sys
import traceback
import warnings
from enum import Enum
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
import shapely.geometry
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")

import atexit
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import psutil

from nmaipy import log, parcels
from nmaipy.__version__ import __version__
from nmaipy.base_exporter import BaseExporter
from nmaipy.constants import (
    ADDRESS_FIELDS,
    AOI_ID_COLUMN_NAME,
    API_CRS,
    BUILDING_STYLE_CLASS_IDS,
    DEFAULT_URL_ROOT,
    SINCE_COL_NAME,
    SURVEY_RESOURCE_ID_COL_NAME,
    UNTIL_COL_NAME,
)
from nmaipy.feature_api import FeatureApi

_process_feature_api = None


def _flatten_attribute_list(attr_list):
    """
    Flatten a list of attribute dictionaries into a single flat dictionary with dot notation.

    This function processes the 'attributes' field from Nearmap AI Feature API responses,
    which contains a list of attribute objects with nested structures. It flattens these
    into a single dictionary suitable for columnar storage in GeoParquet files.

    Args:
        attr_list: List of attribute dictionaries from the API response. Each dictionary
                  typically contains:
                  - 'description': Human-readable name (e.g., "Building 3d attributes")
                  - 'classId': UUID identifier for the attribute type
                  - 'internalClassId': Internal ID (skipped for security)
                  - Various data fields specific to the attribute type
                  - 'components': Optional list of sub-components (e.g., roof materials)

    Returns:
        dict: Flattened dictionary with dot-notation keys. For example:
              {
                  "Building 3d attributes.height": 8.5,
                  "Building 3d attributes.numStories.1": 0.8,
                  "Roof material.components": "[{...}]"  # JSON string
              }

    Notes:
        - 'internalClassId' fields are always skipped (internal use only)
        - 'description' fields are used as prefixes but not included as values
        - 'components' arrays are JSON-serialized for QGIS compatibility
        - Nested dictionaries are flattened with dot notation
        - Returns empty dict if attr_list is None or not a list
    """
    if not attr_list or not isinstance(attr_list, list):
        return {}

    flat_dict = {}
    for i, attr_obj in enumerate(attr_list):
        if not isinstance(attr_obj, dict):
            continue

        # Get the description to use as a prefix
        desc = attr_obj.get("description", f"attr_{i}")

        # Process each field in the attribute object
        for key, value in attr_obj.items():
            # Skip internal fields and redundant description
            if key in ["description", "internalClassId"]:
                continue

            # Special handling for components - serialize as JSON
            if key == "components" and isinstance(value, (list, dict)):
                # Clean components to remove internalClassId
                if isinstance(value, list):
                    cleaned_components = []
                    for comp in value:
                        if isinstance(comp, dict):
                            # Remove internalClassId from each component
                            cleaned_comp = {
                                k: v for k, v in comp.items() if k != "internalClassId"
                            }
                            cleaned_components.append(cleaned_comp)
                        else:
                            cleaned_components.append(comp)
                    flat_dict[f"{desc}.components"] = json.dumps(cleaned_components)
                else:
                    # If it's a dict (shouldn't be, but just in case)
                    flat_dict[f"{desc}.components"] = json.dumps(value)
            # Handle nested dictionaries
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_value is not None:
                        flat_dict[f"{desc}.{key}.{sub_key}"] = sub_value
            # Direct attributes
            elif value is not None:
                flat_dict[f"{desc}.{key}"] = value

    return flat_dict


def _flatten_damage(damage_obj):
    """
    Flatten a damage dictionary into a flat dictionary with dot notation.

    This function processes the 'damage' field from Nearmap AI Feature API responses
    for building lifecycle features. It flattens the nested damage structure into
    a single dictionary suitable for columnar storage in GeoParquet files.

    Args:
        damage_obj: Damage dictionary from the API response, containing:
                   - 'confidences': dict with 'raw', '3tier', '2tier' sub-dicts
                   - 'ratios': list of damage indicator ratios

    Returns:
        dict: Flattened dictionary with dot-notation keys. For example:
              {
                  "damage.confidences.raw.Undamaged": 0.967,
                  "damage.confidences.raw.Affected": 0.028,
                  "damage.confidences.2tier.MajorOrDestroyed": 0.001,
                  "damage.ratios.Exposed Underlayment": 0,
                  "damage.ratios.Missing Roof Tile or Shingle": 0.15,
              }

    Notes:
        - Ratio descriptions preserve spaces to match attribute naming convention
        - Returns empty dict if damage_obj is None or not a dict
    """
    if not damage_obj or not isinstance(damage_obj, dict):
        return {}

    flat_dict = {}

    # Flatten confidences
    confidences = damage_obj.get("confidences")
    if isinstance(confidences, dict):
        # Flatten raw confidences
        raw = confidences.get("raw")
        if isinstance(raw, dict):
            for class_name, confidence in raw.items():
                flat_dict[f"damage.confidences.raw.{class_name}"] = confidence

        # Flatten 3tier confidences
        tier3 = confidences.get("3tier")
        if isinstance(tier3, dict):
            for class_name, confidence in tier3.items():
                flat_dict[f"damage.confidences.3tier.{class_name}"] = confidence

        # Flatten 2tier confidences
        tier2 = confidences.get("2tier")
        if isinstance(tier2, dict):
            for class_name, confidence in tier2.items():
                flat_dict[f"damage.confidences.2tier.{class_name}"] = confidence

    # Flatten ratios
    ratios = damage_obj.get("ratios")
    if isinstance(ratios, list):
        for ratio_item in ratios:
            if isinstance(ratio_item, dict):
                description = ratio_item.get("description")
                ratio_value = ratio_item.get("ratioAbove50PctConf")
                if description is not None and ratio_value is not None:
                    # Keep spaces in description to match attribute naming convention
                    flat_dict[f"damage.ratios.{description}"] = ratio_value

    return flat_dict


def cleanup_process_feature_api():
    """
    Clean up the process-level FeatureApi instance
    """
    global _process_feature_api
    if _process_feature_api is not None:
        try:
            _process_feature_api.cleanup()
        except:
            pass
        _process_feature_api = None


class Endpoint(Enum):
    FEATURE = "feature"
    ROLLUP = "rollup"


CHUNK_SIZE = 500
PROCESSES = 4
THREADS = 10  # Reduced from 20 to prevent resource exhaustion

logger = log.get_logger()


def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(
        prog="nmaipy",
        description="Nearmap AI Python Library - Extract AI features from aerial imagery",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--aoi-file", help="Input AOI file path or S3 URL", type=str, required=True
    )
    parser.add_argument(
        "--output-dir", help="Directory to store results", type=str, required=True
    )
    parser.add_argument(
        "--packs",
        help="List of AI packs",
        type=str,
        nargs="+",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--classes",
        help="List of Feature Class IDs (UUIDs)",
        type=str,
        nargs="+",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--include",
        help="List of additional data to include (e.g. roofSpotlightIndex, roofConditionConfidenceStats)",
        type=str,
        nargs="+",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--primary-decision",
        help="Primary feature decision method: largest_intersection|nearest",
        type=str,
        required=False,
        default="largest_intersection",
    )
    parser.add_argument(
        "--aoi-grid-min-pct",
        help="The minimum threshold (0-100) for how much of a grid cell (proportion of squares) must get a successful result (not 404) to return. Default is strict full coverage (100) required.",
        type=int,
        required=False,
        default=100,
    )
    parser.add_argument(
        "--aoi-grid-inexact",
        help="Permit inexact merging of large AOIs that get gridded, end up getting grid squares from multiple dates, then merging. Deduplication will work poorly for things like buildings.",
        action="store_true",
    )
    parser.add_argument(
        "--processes",
        help="Number of processes",
        type=int,
        required=False,
        default=PROCESSES,
    )
    parser.add_argument(
        "--threads",
        help="Number of threads",
        type=int,
        required=False,
        default=THREADS,
    )
    parser.add_argument(
        "--chunk-size",
        help="Number of AOIs to process in a single temporarily stored chunk file. Smaller files increase parallelism.",
        type=int,
        required=False,
        default=CHUNK_SIZE,
    )
    parser.add_argument(
        "--include-parcel-geometry",
        help="If set, parcel geometries will be in the output",
        action="store_true",
    )
    parser.add_argument(
        "--no-parcel-mode",
        help="If set, disable the API's parcel mode (which filters features based on parcel boundaries)",
        action="store_true",
    )
    parser.add_argument(
        "--save-features",
        help="If set, save the raw vectors as a geoparquet file for loading in GIS tools. This can be quite time consuming.",
        action="store_true",
    )
    parser.add_argument(
        "--save-buildings",
        help="If set, save a building-level geoparquet file with one row per building feature and associated attributes.",
        action="store_true",
    )
    parser.add_argument(
        "--rollup-format",
        help="csv | parquet: Whether to store output as .csv or .parquet (defaults to csv)",
        type=str,
        required=False,
        default="csv",
    )
    parser.add_argument(
        "--cache-dir",
        help="Location to store cache.",
        required=False,
    )
    parser.add_argument(
        "--no-cache",
        help="If set, turn off cache.",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite-cache",
        help="If set, ignore the existing cache and overwrite files as they are downloaded.",
        action="store_true",
    )
    parser.add_argument(
        "--compress-cache",
        help="If set, use gzip compression on each json payload in the cache.",
        action="store_true",
    )
    parser.add_argument(
        "--country",
        help="Country code for area calculations (au, us, ca, nz)",
        required=True,
    )
    parser.add_argument(
        "--alpha",
        help="Include alpha layers",
        action="store_true",
    )
    parser.add_argument(
        "--beta",
        help="Include beta layers",
        action="store_true",
    )
    parser.add_argument(
        "--prerelease",
        help="Include prerelease system versions",
        action="store_true",
    )
    parser.add_argument(
        "--only3d",
        help="Restrict date based queries to 3D data only",
        action="store_true",
    )
    parser.add_argument(
        "--since",
        help="Bulk limit on date for responses (earliest inclusive date returned). Presence of 'since' column in data takes precedent.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--until",
        help="Bulk limit on date for responses (earliest inclusive date returned). Presence of 'until' column in data takes precedent.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--endpoint",
        help="Select which endpoint gets used for rollups - 'feature' (default) or 'rollup'",
        type=str,
        required=False,
        default="feature",
    )
    parser.add_argument(
        "--url-root",
        help="Overwrite the root URL with a custom one.",
        type=str,
        required=False,
        default=DEFAULT_URL_ROOT,
    )
    parser.add_argument(
        "--system-version-prefix",
        help="Restrict responses to a specific system version generation (e.g. gen6-).",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--system-version",
        help="Restrict responses to a specific system version (e.g. gen6-glowing_grove-1.0).",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--rapid",
        help="Enable rapid mode for damage classification (requires gen6 system version)",
        action="store_true",
    )
    parser.add_argument(
        "--order",
        help="Order for date-based requests: 'earliest' or 'latest' (default: latest)",
        type=str,
        choices=["earliest", "latest"],
        required=False,
        default=None,
    )
    parser.add_argument(
        "--exclude-tiles-with-occlusion",
        help="Exclude survey resources with occluded tiles",
        action="store_true",
    )
    parser.add_argument(
        "--api-key",
        help="API key to use (overrides API_KEY environment variable)",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--log-level",
        help="Log level (DEBUG, INFO, ...)",
        required=False,
        default="INFO",
        type=str,
    )
    return parser.parse_args()


def cleanup_process_resources():
    """Helper to ensure processes are cleaned up"""
    # Clean up the process-level FeatureApi instance
    cleanup_process_feature_api()

    gc.collect()
    # Force cleanup of any remaining ProcessPoolExecutor threads
    if hasattr(concurrent.futures.process, "_threads_wakeups"):
        concurrent.futures.process._threads_wakeups.clear()


def cleanup_thread_sessions(executor):
    """Helper to ensure thread sessions are properly closed"""
    if hasattr(executor, "_threads"):
        for thread in executor._threads:
            if hasattr(thread, "_local"):
                if hasattr(thread._local, "session"):
                    try:
                        thread._local.session.close()
                    except:
                        pass


class AOIExporter(BaseExporter):
    def __init__(
        self,
        aoi_file="default_aoi_file",
        output_dir="default_output_dir",
        packs=None,
        classes=None,
        include=None,
        primary_decision="largest_intersection",
        aoi_grid_min_pct=100,
        aoi_grid_inexact=False,
        processes=PROCESSES,
        threads=THREADS,
        chunk_size=CHUNK_SIZE,
        include_parcel_geometry=False,
        save_features=False,
        save_buildings=False,
        rollup_format="csv",
        cache_dir=None,
        no_cache=False,
        overwrite_cache=False,
        compress_cache=False,
        country="us",
        alpha=False,
        beta=False,
        prerelease=False,
        only3d=False,
        since=None,
        until=None,
        endpoint="feature",
        url_root=DEFAULT_URL_ROOT,
        system_version_prefix=None,
        system_version=None,
        log_level="INFO",
        api_key=None,  # Add API key parameter
        parcel_mode=True,  # Add parcel mode parameter with default True
        rapid=False,
        order=None,
        exclude_tiles_with_occlusion=False,
    ):
        # Initialize base exporter first
        super().__init__(
            output_dir=output_dir,
            processes=processes,
            chunk_size=chunk_size,
            log_level=log_level,
        )

        # Assign AOIExporter-specific parameters to instance variables
        self.aoi_file = aoi_file
        self.packs = packs
        self.classes = classes
        self.include = include
        self.primary_decision = primary_decision
        self.aoi_grid_min_pct = aoi_grid_min_pct
        self.aoi_grid_inexact = aoi_grid_inexact
        # Note: processes, chunk_size, log_level handled by BaseExporter
        self.threads = threads
        self.include_parcel_geometry = include_parcel_geometry
        self.save_features = save_features
        self.save_buildings = save_buildings
        self.rollup_format = rollup_format
        self.cache_dir = cache_dir
        self.no_cache = no_cache
        self.overwrite_cache = overwrite_cache
        self.compress_cache = compress_cache
        self.country = country
        self.alpha = alpha
        self.beta = beta
        self.prerelease = prerelease
        self.only3d = only3d
        self.since = since
        self.until = until
        self.endpoint = endpoint
        self.url_root = url_root
        self.system_version_prefix = system_version_prefix
        self.system_version = system_version
        self.api_key_param = api_key  # Store the API key parameter
        self.parcel_mode = parcel_mode  # Store the parcel mode parameter
        self.rapid = rapid
        self.order = order
        self.exclude_tiles_with_occlusion = exclude_tiles_with_occlusion

        # Note: logger already configured by BaseExporter

    def api_key(self) -> str:
        # Use provided API key if available, otherwise fall back to environment variable
        if hasattr(self, "api_key_param") and self.api_key_param is not None:
            return self.api_key_param
        return os.getenv("API_KEY")

    def _stream_and_convert_features(
        self, feature_paths: List[Path], outpath_features: Path
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Stream feature chunks directly to a geoparquet file.
        This approach avoids loading all chunks into memory simultaneously.

        Args:
            feature_paths: List of paths to feature chunk parquet files
            outpath_features: Output path for final geoparquet file

        Returns:
            None since we don't need the GeoDataFrame in memory
        """

        import pyarrow as pa
        import pyarrow.parquet as pq

        pqwriter = None
        reference_columns = None  # Store column order from first chunk
        reference_schema = None  # Store PyArrow schema from first chunk

        # Stream chunks directly to geoparquet
        for i, cp in enumerate(
            tqdm(
                feature_paths,
                desc="Streaming chunks",
                file=sys.stdout,
                position=0,
                leave=True,
            )
        ):
            try:
                df_feature_chunk = gpd.read_parquet(cp)
            except Exception as e:
                self.logger.error(f"Failed to read {cp}: {e}")
                continue

            if len(df_feature_chunk) > 0:
                # Store CRS and column schema from first chunk
                if reference_columns is None:
                    reference_columns = df_feature_chunk.columns

                # Validate and reorder columns for subsequent chunks
                current_columns = list(df_feature_chunk.columns)
                missing_cols = set(reference_columns) - set(current_columns)
                extra_cols = set(current_columns) - set(reference_columns)
                if missing_cols or extra_cols:
                    self.logger.warning(f"Chunk {i} schema mismatch detected:")
                    if missing_cols:
                        self.logger.warning(
                            f"  Missing columns: {sorted(missing_cols)}"
                        )
                        # Add missing columns with null values
                        for col in missing_cols:
                            df_feature_chunk[col] = None
                    if extra_cols:
                        self.logger.warning(f"  Extra columns: {sorted(extra_cols)}")

                # Ensure column order matches reference for all chunks
                if list(current_columns) != list(reference_columns):
                    self.logger.debug(
                        f"Chunk {i}: Column order differs from reference, reordering silently"
                    )
                    df_feature_chunk = df_feature_chunk[reference_columns]

                # Convert to regular pandas DataFrame for pyarrow, converting geometry to WKB
                geom_col = df_feature_chunk.geometry.to_wkb()
                df_feature_chunk = pd.DataFrame(df_feature_chunk)
                df_feature_chunk["geometry"] = geom_col

                # Convert to pyarrow table and stream
                table = pa.Table.from_pandas(df_feature_chunk, preserve_index=True)

                if pqwriter is None:
                    reference_schema = table.schema

                    # Create geoparquet metadata from the start
                    geo_metadata = {
                        "version": "1.0.0",
                        "primary_column": "geometry",
                        "columns": {
                            "geometry": {
                                "encoding": "WKB",
                                "geometry_types": [],
                                "crs": API_CRS,
                                "edges": "planar",
                                "orientation": "counterclockwise",
                                "bbox": None,
                            }
                        },
                    }

                    # Add geoparquet metadata to schema
                    schema_with_geo = reference_schema.with_metadata(
                        {b"geo": json.dumps(geo_metadata).encode("utf-8")}
                    )

                    pqwriter = pq.ParquetWriter(outpath_features, schema_with_geo)
                else:
                    # Cast to reference schema if needed
                    if table.schema != reference_schema:
                        try:
                            table = table.cast(reference_schema)
                        except Exception as e:
                            # Schema casting failed - likely due to null type columns
                            # Try to fix by creating an empty table with the reference schema
                            self.logger.warning(
                                f"Chunk {i}: Schema casting failed, attempting to create compatible table"
                            )
                            self.logger.debug(f"  Error: {e}")
                            self.logger.debug(f"  Reference schema: {reference_schema}")
                            self.logger.debug(f"  Current schema: {table.schema}")

                            # Create a new table with the reference schema structure but current data
                            # For columns that can't be cast (e.g., null -> string), create empty arrays
                            arrays = []
                            for field in reference_schema:
                                if field.name in table.column_names:
                                    col = table.column(field.name)
                                    # Try to cast this individual column
                                    try:
                                        arrays.append(col.cast(field.type))
                                    except pa.ArrowInvalid:
                                        # Can't cast (e.g., null -> string), create empty/null array with correct type
                                        self.logger.debug(
                                            f"    Creating null array for column '{field.name}' ({field.type})"
                                        )
                                        arrays.append(
                                            pa.nulls(len(table), type=field.type)
                                        )
                                else:
                                    # Column missing entirely, create null array with correct type
                                    self.logger.debug(
                                        f"    Creating null array for missing column '{field.name}' ({field.type})"
                                    )
                                    arrays.append(pa.nulls(len(table), type=field.type))

                            table = pa.Table.from_arrays(
                                arrays, schema=reference_schema
                            )
                            self.logger.debug(
                                f"  Successfully created compatible table with {len(table)} rows"
                            )
                pqwriter.write_table(table)

        # Close the writer
        if pqwriter is not None:
            pqwriter.close()

            # Log final status
            mem = psutil.virtual_memory()
            final_file_size_gb = outpath_features.stat().st_size / (1024**3)
            self.logger.debug(
                f"Successfully streamed to geoparquet without temporary files. "
                f"Memory: {(mem.total - mem.available)/1024**3:.2f}GB / {mem.total/1024**3:.2f}GB ({mem.percent:.1f}%). "
                f"Final file size: {final_file_size_gb:.2f}GB"
            )

            return None
        else:
            self.logger.warning("No feature data found to write")
            return None

    def get_chunk_output_file(self, chunk_id: str) -> Path:
        """
        Get the path to the main output file for a chunk.

        Args:
            chunk_id: Unique identifier for this chunk

        Returns:
            Path to the chunk's rollup file (used for cache checking)
        """
        return self.chunk_path / f"rollup_{chunk_id}.parquet"

    def process_chunk(
        self,
        chunk_id: str,
        aoi_gdf: gpd.GeoDataFrame,
        classes_df: pd.DataFrame = None,
        progress_counters: dict = None,
        **kwargs
    ):
        """
        Create a parcel rollup for a chunk of parcels.

        Args:
            chunk_id: Unique identifier for this chunk
            aoi_gdf: GeoDataFrame containing AOIs to process
            classes_df: DataFrame of feature classes
            progress_counters: Optional dict with 'total' and 'completed' multiprocessing.Value counters
            **kwargs: Additional parameters (unused, but required by base class)
        """
        # Configure logging for worker process
        BaseExporter.configure_worker_logging(self.log_level)
        logger = log.get_logger()

        feature_api = None
        try:
            if self.cache_dir is None and not self.no_cache:
                cache_dir = Path(self.output_dir)
            else:
                cache_dir = (
                    Path(self.cache_dir) if self.cache_dir else Path(self.output_dir)
                )

            if not self.no_cache:
                cache_path = cache_dir / "cache"
            else:
                cache_path = None

            outfile = self.chunk_path / f"rollup_{chunk_id}.parquet"
            outfile_features = self.chunk_path / f"features_{chunk_id}.parquet"
            outfile_errors = self.chunk_path / f"errors_{chunk_id}.parquet"
            if outfile.exists():
                return

            # Get additional parcel attributes from parcel geometry
            if isinstance(aoi_gdf, gpd.GeoDataFrame):
                rep_point = aoi_gdf.representative_point()
                aoi_gdf["query_aoi_lat"] = rep_point.y
                aoi_gdf["query_aoi_lon"] = rep_point.x

            # Get features
            feature_api = FeatureApi(
                api_key=self.api_key(),
                cache_dir=cache_path,
                overwrite_cache=self.overwrite_cache,
                compress_cache=self.compress_cache,
                threads=self.threads,
                alpha=self.alpha,
                beta=self.beta,
                prerelease=self.prerelease,
                only3d=self.only3d,
                url_root=self.url_root,
                system_version_prefix=self.system_version_prefix,
                system_version=self.system_version,
                aoi_grid_min_pct=self.aoi_grid_min_pct,
                aoi_grid_inexact=self.aoi_grid_inexact,
                parcel_mode=self.parcel_mode,
                progress_counters=progress_counters,
            )
            if self.endpoint == Endpoint.ROLLUP.value:
                self.logger.debug(
                    f"Chunk {chunk_id}: Getting rollups for {len(aoi_gdf)} AOIs ({self.endpoint=})"
                )
                rollup_df, metadata_df, errors_df = feature_api.get_rollup_df_bulk(
                    aoi_gdf,
                    region=self.country,
                    since_bulk=self.since,
                    until_bulk=self.until,
                    packs=self.packs,
                    classes=self.classes,
                    max_allowed_error_pct=100,
                )
                rollup_df.columns = FeatureApi._multi_to_single_index(rollup_df.columns)
                mem = psutil.virtual_memory()
                self.logger.debug(
                    f"Chunk {chunk_id} failed {len(errors_df)} of {len(aoi_gdf)} AOI requests. "
                    f"{len(rollup_df)} rollups returned on {len(rollup_df.index.unique())} unique {rollup_df.index.name}s. "
                    f"Memory: {(mem.total - mem.available)/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB ({mem.percent}%)"
                )
                if len(errors_df) > 0:
                    if "message" in errors_df:
                        error_counts = errors_df["message"].value_counts().to_dict()
                        self.logger.debug(
                            f"Found {len(errors_df)} errors by type: {error_counts}"
                        )
                    else:
                        self.logger.debug(f"Found {len(errors_df)} errors")
                if len(errors_df) == len(aoi_gdf):
                    errors_df.to_parquet(outfile_errors)
                    return
            elif self.endpoint == Endpoint.FEATURE.value:
                self.logger.debug(
                    f"Chunk {chunk_id}: Getting features for {len(aoi_gdf)} AOIs ({self.endpoint=})"
                )
                features_gdf, metadata_df, errors_df = (
                    feature_api.get_features_gdf_bulk(
                        aoi_gdf,
                        region=self.country,
                        since_bulk=self.since,
                        until_bulk=self.until,
                        packs=self.packs,
                        classes=self.classes,
                        include=self.include,
                        max_allowed_error_pct=100,
                    )
                )
                mem = psutil.virtual_memory()
                self.logger.debug(
                    f"Chunk {chunk_id} failed {len(errors_df)} of {len(aoi_gdf)} AOI requests. "
                    f"Memory: {(mem.total - mem.available)/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB ({mem.percent}%)"
                )
                if len(errors_df) > 0:
                    if "message" in errors_df:
                        error_counts = errors_df["message"].value_counts().to_dict()
                        self.logger.debug(
                            f"Found {len(errors_df)} errors by type: {error_counts}"
                        )
                    else:
                        self.logger.debug(f"Found {len(errors_df)} errors")
                if len(errors_df) == len(aoi_gdf):
                    errors_df.to_parquet(outfile_errors)
                    return

                # Create rollup
                rollup_df = parcels.parcel_rollup(
                    aoi_gdf,
                    features_gdf,
                    classes_df,
                    country=self.country,
                    primary_decision=self.primary_decision,
                )
            else:
                self.logger.error(f"Not a valid endpoint selection: {self.endpoint}")
                sys.exit(1)

            # Put it all together and save
            meta_data_columns = [
                "system_version",
                "link",
                "date",
                "survey_id",
                "survey_resource_id",
                "perspective",
                "postcat",
            ]
            # Validate columns
            for meta_data_column in meta_data_columns:
                if meta_data_column in aoi_gdf.columns:
                    metadata_df = metadata_df.rename(
                        columns={meta_data_column: f"nmaipy_{meta_data_column}"}
                    )
                    meta_data_columns.remove(meta_data_column)

            final_df = metadata_df.merge(rollup_df, on=AOI_ID_COLUMN_NAME).merge(
                aoi_gdf, on=AOI_ID_COLUMN_NAME
            )
            parcel_columns = [c for c in aoi_gdf.columns if c != "geometry"]
            columns = (
                parcel_columns
                + [c for c in meta_data_columns if c in final_df.columns]
                + [
                    c
                    for c in final_df.columns
                    if c not in parcel_columns + meta_data_columns + ["geometry"]
                ]
            )
            final_df = final_df[columns]
            if self.include_parcel_geometry:
                columns.append("geometry")
            columns = [c for c in columns if c in final_df.columns]
            date2str = lambda d: str(d).replace("-", "")
            make_link = (
                lambda d: f"https://apps.nearmap.com/maps/#/@{d.query_aoi_lat},{d.query_aoi_lon},21.00z,0d/V/{date2str(d.date)}?locationMarker"
            )
            if self.endpoint == Endpoint.ROLLUP.value:
                if (
                    "query_aoi_lat" in final_df.columns
                    and "query_aoi_lon" in final_df.columns
                ):
                    final_df["link"] = final_df.apply(make_link, axis=1)
                final_df = final_df.drop(columns=["system_version", "date"])
            self.logger.debug(
                f"Chunk {chunk_id}: Writing {len(final_df)} rows for rollups and {len(errors_df)} for errors."
            )
            try:
                errors_df.to_parquet(outfile_errors)
            except Exception as e:
                self.logger.error(
                    f"Chunk {chunk_id}: Failed writing errors_df ({len(errors_df)} rows) to {outfile_errors}."
                )
                self.logger.error(f"Error: {type(e).__name__}: {str(e)}")
            try:
                # Handle the geometry column separately to avoid conversion issues
                has_geometry = "geometry" in final_df.columns
                geometry_series = None

                if has_geometry:
                    # Store the geometry column separately
                    geometry_series = final_df["geometry"]
                    final_df = final_df.drop(columns=["geometry"])

                # Convert dtypes on the dataframe without geometry
                final_df = final_df.convert_dtypes()

                if has_geometry:
                    # Reattach the geometry column
                    final_df["geometry"] = geometry_series
                    # Create a proper GeoDataFrame
                    final_df = gpd.GeoDataFrame(
                        final_df, geometry="geometry", crs=API_CRS
                    )

                # Save with explicit schema version for better QGIS compatibility
                # Requires geopandas >= 1.1.0
                try:
                    final_df.to_parquet(outfile, schema_version="1.0.0")
                except (TypeError, ValueError) as e:
                    # Fallback for older geopandas or pyarrow versions
                    self.logger.debug(
                        f"Could not use schema_version parameter: {e}. Falling back to default."
                    )
                    final_df.to_parquet(outfile)
            except Exception as e:
                self.logger.error(
                    f"Chunk {chunk_id}: Failed writing final_df ({len(final_df)} rows) to {outfile}."
                )
                self.logger.error(
                    f"Error type: {type(e).__name__}, Error message: {str(e)}"
                )
            if self.save_features and (self.endpoint != Endpoint.ROLLUP.value):
                # Check for column name collisions between any two dataframes
                final_features_df = aoi_gdf.rename(
                    columns=dict(geometry="aoi_geometry")
                )

                metadata_cols = set(metadata_df.columns)
                features_cols = set(features_gdf.columns)
                aoi_cols = set(final_features_df.columns)
                metadata_features_overlap = metadata_cols & features_cols - {
                    AOI_ID_COLUMN_NAME
                }
                metadata_aoi_overlap = metadata_cols & aoi_cols - {AOI_ID_COLUMN_NAME}
                features_aoi_overlap = features_cols & aoi_cols - {AOI_ID_COLUMN_NAME}
                all_overlapping = (
                    metadata_features_overlap
                    | metadata_aoi_overlap
                    | features_aoi_overlap
                )
                if all_overlapping:
                    self.logger.warning(
                        f"Column name collisions detected. The following columns exist in multiple dataframes "
                        f"and may be duplicated with '_x' and '_y' suffixes: {sorted(all_overlapping)}"
                    )

                # First merge
                merged1 = metadata_df.merge(features_gdf, on=AOI_ID_COLUMN_NAME)

                # Second merge
                merged2 = merged1.merge(final_features_df, on=AOI_ID_COLUMN_NAME)

                # Check what geometry columns we have after the merge
                geom_cols = [
                    col for col in merged2.columns if "geometry" in col.lower()
                ]

                # Create GeoDataFrame with the appropriate geometry column
                if "geometry" in merged2.columns:
                    final_features_df = gpd.GeoDataFrame(merged2, crs=API_CRS)
                elif "geometry_y" in merged2.columns:
                    # Features geometry (from poles)
                    final_features_df = gpd.GeoDataFrame(
                        merged2, geometry="geometry_y", crs=API_CRS
                    )
                elif "geometry_x" in merged2.columns:
                    # AOI geometry
                    final_features_df = gpd.GeoDataFrame(
                        merged2, geometry="geometry_x", crs=API_CRS
                    )
                else:
                    error_msg = (
                        f"Chunk {chunk_id}: No valid geometry column found after merge. "
                        f"Expected 'geometry', 'geometry_x', or 'geometry_y'. "
                        f"Found columns: {geom_cols if geom_cols else 'none'}. "
                        f"All columns: {list(merged2.columns)[:20]}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                if "aoi_geometry" in final_features_df.columns:
                    final_features_df["aoi_geometry"] = (
                        final_features_df.aoi_geometry.to_wkt()
                    )

                # Apply flattening to attributes if present
                if "attributes" in final_features_df.columns:
                    # Apply flattening and create DataFrame - simpler approach that avoids index issues
                    flattened_attrs = (
                        final_features_df["attributes"]
                        .apply(_flatten_attribute_list)
                        .apply(pd.Series)
                    )
                    if not flattened_attrs.empty:
                        logger.debug(
                            f"Chunk {chunk_id}: Flattened {len(flattened_attrs.columns)} attribute columns from {final_features_df['attributes'].notna().sum()} features"
                        )
                        # Drop the attributes column
                        final_features_df = final_features_df.drop(
                            columns=["attributes"]
                        )
                        # Add the flattened columns
                        for col in flattened_attrs.columns:
                            if col not in final_features_df.columns:
                                final_features_df[col] = flattened_attrs[col]
                    else:
                        # No attributes to flatten, just drop the column
                        final_features_df = final_features_df.drop(
                            columns=["attributes"]
                        )

                # Apply flattening to damage if present
                if "damage" in final_features_df.columns:
                    # Apply flattening and create DataFrame
                    flattened_damage = (
                        final_features_df["damage"]
                        .apply(_flatten_damage)
                        .apply(pd.Series)
                    )
                    if not flattened_damage.empty:
                        logger.debug(
                            f"Chunk {chunk_id}: Flattened {len(flattened_damage.columns)} damage columns from {final_features_df['damage'].notna().sum()} features"
                        )
                        # Drop the damage column
                        final_features_df = final_features_df.drop(columns=["damage"])
                        # Add the flattened columns
                        for col in flattened_damage.columns:
                            if col not in final_features_df.columns:
                                final_features_df[col] = flattened_damage[col]
                    else:
                        # No damage to flatten, just drop the column
                        final_features_df = final_features_df.drop(columns=["damage"])
                if len(final_features_df) > 0:
                    try:
                        if (
                            not self.include_parcel_geometry
                            and "aoi_geometry" in final_features_df.columns
                        ):
                            final_features_df = final_features_df.drop(
                                columns=["aoi_geometry"]
                            )
                        final_features_df = final_features_df[
                            ~(
                                final_features_df.geometry.is_empty
                                | final_features_df.geometry.isna()
                            )
                        ]

                        # Convert dict-type include parameters to JSON strings to avoid Parquet serialization errors
                        # Include parameters like defensibleSpace, hurricaneScore, roofSpotlightIndex can be dicts
                        # and need to be serialized to JSON strings for Parquet compatibility
                        # Apply to all object-dtype columns (potential dict containers) and let the function
                        # handle each value type appropriately - more robust than sampling
                        def serialize_include_param(val):
                            if val is None or pd.isna(val):
                                return None
                            if isinstance(val, dict):
                                return json.dumps(val)
                            # Return other types as-is (strings, numbers, etc.)
                            return val

                        # Apply serialization to all object-dtype columns (where dicts would be stored)
                        # Skip geometry column which is handled separately by GeoPandas
                        object_columns = final_features_df.select_dtypes(
                            include=["object"]
                        ).columns
                        object_columns = [
                            col for col in object_columns if col != "geometry"
                        ]

                        for col in object_columns:
                            final_features_df[col] = final_features_df[col].apply(
                                serialize_include_param
                            )

                        # Ensure it's a proper GeoDataFrame before saving to parquet
                        if not isinstance(final_features_df, gpd.GeoDataFrame):
                            final_features_df = gpd.GeoDataFrame(
                                final_features_df, geometry="geometry", crs=API_CRS
                            )
                        else:
                            final_features_df = final_features_df.set_crs(
                                API_CRS, allow_override=True
                            )

                        # Save with explicit schema version for better QGIS compatibility
                        # Requires geopandas >= 1.1.0
                        try:
                            final_features_df.to_parquet(
                                outfile_features, schema_version="1.0.0"
                            )
                        except (TypeError, ValueError) as e:
                            # Fallback for older geopandas or pyarrow versions
                            self.logger.debug(
                                f"Could not use schema_version parameter: {e}. Falling back to default."
                            )
                            final_features_df.to_parquet(outfile_features)
                    except Exception as e:
                        self.logger.error(
                            f"Failed to save features parquet file for chunk_id {chunk_id}. Errors saved to {outfile_errors}. Rollup saved to {outfile}."
                        )
                        self.logger.error(
                            f"Error type: {type(e).__name__}, Error message: {str(e)}"
                        )
                        self.logger.error(e)
            self.logger.debug(f"Finished saving chunk {chunk_id}")

        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_id}: {e}")
            raise
        finally:
            # Clean up feature API to close network connections
            if "feature_api" in locals():
                try:
                    feature_api.cleanup()
                    del feature_api
                except:
                    pass

            # Clear GeoPandas/Shapely/GEOS caches and thread-local storage
            try:
                # Clear Shapely's thread-local GEOS handles which can accumulate
                if hasattr(shapely, "_geos"):
                    shapely._geos.clear_all_thread_local()

                # Clear PROJ context caches which can accumulate coordinate system data
                try:
                    if hasattr(pyproj, "proj"):
                        # Clear the global CRS cache
                        pyproj.crs.CRS.clear_cache()
                    if hasattr(pyproj, "_datadir"):
                        # Clear proj data directory cache
                        pyproj._datadir.clear_data_dir()
                except:
                    pass

            except:
                pass

    def run(self):
        self.logger.debug("Starting parcel rollup")

        # Process a single AOI file
        aoi_path = self.aoi_file
        self.logger.info(f"Processing AOI file {aoi_path}")

        cache_path = Path(self.output_dir) / "cache"
        cache_path.mkdir(parents=True, exist_ok=True)
        # Note: chunk_path and final_path created by BaseExporter

        # Get classes
        feature_api = FeatureApi(
            api_key=self.api_key(),
            alpha=self.alpha,
            beta=self.beta,
            prerelease=self.prerelease,
            only3d=self.only3d,
            parcel_mode=self.parcel_mode,
        )
        try:
            if self.packs is not None:
                classes_df = feature_api.get_feature_classes(self.packs)
            else:
                classes_df = feature_api.get_feature_classes()  # All classes
                if self.classes is not None:
                    classes_df = classes_df[classes_df.index.isin(self.classes)]
        finally:
            feature_api.cleanup()

        # Modify output file paths using the AOI file name
        outpath = self.final_path / f"{Path(aoi_path).stem}.{self.rollup_format}"
        outpath_features = self.final_path / f"{Path(aoi_path).stem}_features.parquet"
        outpath_buildings = (
            self.final_path / f"{Path(aoi_path).stem}_buildings.{self.rollup_format}"
        )

        # Check if all required outputs already exist
        outputs_exist = outpath.exists()
        if self.save_features:
            outputs_exist = outputs_exist and outpath_features.exists()
        if self.save_buildings:
            outputs_exist = outputs_exist and outpath_buildings.exists()

        if outputs_exist:
            self.logger.info(f"Output already exists, skipping {Path(aoi_path).stem}")
            return

        aoi_gdf = parcels.read_from_file(aoi_path, id_column=AOI_ID_COLUMN_NAME)

        if isinstance(aoi_gdf, gpd.GeoDataFrame):
            aoi_gdf = aoi_gdf.to_crs(API_CRS)
        else:
            self.logger.info("No geometry found in parcel data - using address fields")
            for field in ADDRESS_FIELDS:
                if field not in aoi_gdf:
                    self.logger.error(f"Missing field {field} in parcel data")
                    sys.exit(1)

        # Print out info around what is being inferred from column names:
        if SURVEY_RESOURCE_ID_COL_NAME in aoi_gdf:
            logger.info(
                f"{SURVEY_RESOURCE_ID_COL_NAME} will be used to get results from the exact Survey Resource ID, instead of using date based filtering."
            )
        else:
            logger.debug(
                f"No {SURVEY_RESOURCE_ID_COL_NAME} column provided, so date based endpoint will be used."
            )
            if SINCE_COL_NAME in aoi_gdf:
                logger.info(
                    f'The column "{SINCE_COL_NAME}" will be used as the earliest permitted date (YYYY-MM-DD) for each Query AOI.'
                )
            elif self.since is not None:
                logger.debug(
                    f"The since date of {self.since} will limit the earliest returned date for all Query AOIs"
                )
            else:
                logger.debug("No earliest date will be used")
            if UNTIL_COL_NAME in aoi_gdf:
                logger.info(
                    f'The column "{UNTIL_COL_NAME}" will be used as the latest permitted date (YYYY-MM-DD) for each Query AOI.'
                )
            elif self.until is not None:
                logger.debug(
                    f"The until date of {self.until} will limit the latest returned date for all Query AOIs"
                )
            else:
                logger.debug("No latest date will used")

        self.logger.debug(f"Using endpoint '{self.endpoint}' for rollups.")

        # Split into chunks and process in parallel (using BaseExporter methods)
        aoi_stem = Path(aoi_path).stem
        chunks_to_process, skipped_chunks, skipped_aois = self.split_into_chunks(
            aoi_gdf, aoi_stem, check_cache=True
        )

        # Calculate initial AOI count for progress tracking (excluding skipped)
        initial_aoi_count = len(aoi_gdf) - skipped_aois

        # Run parallel processing with progress tracking
        self.run_parallel(
            chunks_to_process,
            aoi_stem,
            initial_aoi_count=initial_aoi_count,
            use_progress_tracking=True,  # Enable progress counters for Feature API
            classes_df=classes_df,  # Pass classes_df to process_chunk
        )

        data = []
        data_features = []
        errors = []
        self.logger.debug(
            f"Saving rollup data as {self.rollup_format} file to {outpath}"
        )

        # Calculate total number of chunks (including cached ones)
        num_chunks = max(len(aoi_gdf) // self.chunk_size, 1)

        for i in range(num_chunks):
            chunk_filename = f"rollup_{Path(aoi_path).stem}_{str(i).zfill(4)}.parquet"
            cp = self.chunk_path / chunk_filename
            if cp.exists():
                try:
                    chunk = gpd.read_parquet(cp)
                except ValueError:
                    chunk = pd.read_parquet(cp)
                if len(chunk) > 0:
                    data.append(chunk)
            else:
                error_filename = (
                    f"errors_{Path(aoi_path).stem}_{str(i).zfill(4)}.parquet"
                )
                if (self.chunk_path / error_filename).exists():
                    self.logger.debug(
                        f"Chunk {i} rollup file missing, but error file found."
                    )
                else:
                    self.logger.error(
                        f"Chunk {i} rollup and error files missing. Try rerunning."
                    )
                    sys.exit(1)
        if len(data) > 0:
            data = pd.concat([data for data in data if len(data) > 0])
            if "geometry" in data.columns:
                if not isinstance(data.geometry, gpd.GeoSeries):
                    data["geometry"] = gpd.GeoSeries.from_wkt(data.geometry)
                data = gpd.GeoDataFrame(data, crs=API_CRS)

        else:
            data = pd.DataFrame(data)
        if len(data) > 0:
            if self.rollup_format == "parquet":
                data.to_parquet(outpath, index=True)
            elif self.rollup_format == "csv":
                if "geometry" in data.columns:
                    if hasattr(data.geometry, "to_wkt") and callable(
                        data.geometry.to_wkt
                    ):
                        # If it has a to_wkt method but isn't a GeoSeries
                        data["geometry"] = data.geometry.to_wkt()
                data.to_csv(outpath, index=True)
            else:
                self.logger.info("Invalid output format specified - reverting to csv")
                if "geometry" in data.columns:
                    if hasattr(data.geometry, "to_wkt") and callable(
                        data.geometry.to_wkt
                    ):
                        # If it has a to_wkt method but isn't a GeoSeries
                        data["geometry"] = data.geometry.to_wkt()
                data.to_csv(outpath, index=True)
        outpath_errors = self.final_path / f"{Path(aoi_path).stem}_errors.csv"
        outpath_errors_geoparquet = self.final_path / f"{Path(aoi_path).stem}_errors.parquet"
        self.logger.debug(f"Saving error data as .csv to {outpath_errors}")
        for cp in self.chunk_path.glob(f"errors_{Path(aoi_path).stem}_*.parquet"):
            errors.append(pd.read_parquet(cp))
        if len(errors) > 0:
            errors = pd.concat(errors)
        else:
            errors = pd.DataFrame()

        # Check if we actually have error rows (not just empty DataFrames)
        if len(errors) > 0 and AOI_ID_COLUMN_NAME in errors.columns:
            # Merge with aoi_gdf to add geometry or address information for enhanced error output
            # Use left join to preserve all errors even if aoi_id is missing
            aoi_gdf_for_merge = aoi_gdf.reset_index()

            if isinstance(aoi_gdf, gpd.GeoDataFrame):
                # Geometry mode: merge with geometry column for GeoParquet output
                errors_with_context = errors.merge(
                    aoi_gdf_for_merge[[AOI_ID_COLUMN_NAME, "geometry"]],
                    on=AOI_ID_COLUMN_NAME,
                    how="left",
                )
                # Convert to GeoDataFrame if we have geometry column
                if "geometry" in errors_with_context.columns:
                    errors_gdf = gpd.GeoDataFrame(
                        errors_with_context, geometry="geometry", crs=aoi_gdf.crs
                    )
                else:
                    errors_gdf = errors_with_context
            else:
                # Address mode: merge with address columns for context
                # Get all columns from aoi_gdf except aoi_id (which is already in errors)
                merge_cols = [col for col in aoi_gdf_for_merge.columns if col != AOI_ID_COLUMN_NAME]
                if AOI_ID_COLUMN_NAME in aoi_gdf_for_merge.columns:
                    merge_cols.insert(0, AOI_ID_COLUMN_NAME)
                errors_gdf = errors.merge(
                    aoi_gdf_for_merge[merge_cols],
                    on=AOI_ID_COLUMN_NAME,
                    how="left",
                )

            # Count error types by status code and message
            error_summary = []
            if "status_code" in errors.columns:
                status_counts = errors["status_code"].value_counts()
                error_summary.append(f"status codes: {status_counts.to_dict()}")
            if "message" in errors.columns:
                message_counts = errors["message"].value_counts()
                error_summary.append(f"messages: {message_counts.to_dict()}")

            if error_summary:
                self.logger.info(
                    f"Processing completed with {len(errors)} total failures - {', '.join(error_summary)}"
                )
            else:
                self.logger.info(
                    f"Processing completed with {len(errors)} total failures"
                )
        else:
            # No errors or errors DataFrame is empty/malformed
            errors_gdf = errors
            self.logger.info("Processing completed with no failures")

        # Save CSV format (always - includes address fields if in address mode)
        if isinstance(aoi_gdf, gpd.GeoDataFrame):
            # Geometry mode: CSV without geometry for backward compatibility
            errors.to_csv(outpath_errors, index=True)
        else:
            # Address mode: CSV with address fields merged in for easy viewing in Excel
            errors_gdf.to_csv(outpath_errors, index=True)

        # Save GeoParquet format only for geometry mode (for QGIS visualization)
        # In address mode, users prefer CSV for Excel compatibility
        if isinstance(errors_gdf, gpd.GeoDataFrame) and len(errors_gdf) > 0:
            self.logger.info(
                f"Saving error data with geometry as geoparquet to {outpath_errors_geoparquet}"
            )
            errors_gdf.to_parquet(outpath_errors_geoparquet, index=True)
        if self.save_features:
            feature_paths = [
                p for p in self.chunk_path.glob(f"features_{Path(aoi_path).stem}_*.parquet")
            ]
            self.logger.info(
                f"Saving feature data from {len(feature_paths)} geoparquet chunks to {outpath_features}"
            )

            features_gdf = self._stream_and_convert_features(
                feature_paths, outpath_features
            )

            # If buildings export is enabled, process building features
            if self.save_buildings:
                self.logger.info(
                    f"Saving building-level data as {self.rollup_format} to {outpath_buildings}"
                )
                # Define geoparquet path for buildings
                outpath_buildings_geoparquet = (
                    self.final_path / f"{Path(aoi_path).stem}_building_features.parquet"
                )

                buildings_gdf = parcels.extract_building_features(
                    parcels_gdf=aoi_gdf, features_gdf=features_gdf, country=self.country
                )
                if len(buildings_gdf) > 0:
                    # First, save the geoparquet version with intact geometries
                    self.logger.info(
                        f"Saving building-level data as geoparquet to {outpath_buildings_geoparquet}"
                    )
                    try:
                        # Save with explicit schema version for better QGIS compatibility
                        # Requires geopandas >= 1.1.0
                        try:
                            buildings_gdf.to_parquet(
                                outpath_buildings_geoparquet, schema_version="1.0.0"
                            )
                        except (TypeError, ValueError) as e:
                            # Fallback for older geopandas or pyarrow versions
                            self.logger.debug(
                                f"Could not use schema_version parameter: {e}. Falling back to default."
                            )
                            buildings_gdf.to_parquet(outpath_buildings_geoparquet)
                    except Exception as e:
                        self.logger.error(
                            f"Failed to save buildings geoparquet file: {str(e)}"
                        )

                    # Then convert geodataframe to plain dataframe for tabular output
                    # Keep geometry as WKT representation if needed
                    buildings_df = pd.DataFrame(buildings_gdf)
                    if "geometry" in buildings_df.columns:
                        buildings_df["geometry"] = buildings_df.geometry.apply(
                            lambda geom: geom.wkt if geom else None
                        )

                    # Save in the same format as rollup
                    if self.rollup_format == "parquet":
                        buildings_df.to_parquet(outpath_buildings, index=True)
                    elif self.rollup_format == "csv":
                        buildings_df.to_csv(outpath_buildings, index=True)
                    else:
                        self.logger.info(
                            "Invalid output format specified for buildings - reverting to csv"
                        )
                        buildings_df.to_csv(outpath_buildings, index=True)
                else:
                    self.logger.info(
                        f"No building features found for {Path(aoi_path).stem}"
                    )


def main():
    # Set higher file descriptor limits for running many processes in parallel.
    import resource
    import sys

    if sys.platform != "win32":
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = 32000  # Same as ulimit -n 32000
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
            new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            logger.info(
                f"File descriptor limits - Previous: {soft}, New: {new_soft}, Hard limit: {hard}"
            )
        except ValueError as e:
            # If desired limit is too high, try setting to hard limit
            logger.warning(
                f"Could not set file descriptor limit to {desired}, trying hard limit {hard}"
            )
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
                new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
                logger.info(
                    f"File descriptor limits - Previous: {soft}, New: {new_soft}, Hard limit: {hard}"
                )
            except ValueError as e:
                logger.warning(f"Could not increase file descriptor limits: {e}")
    args = parse_arguments()
    exporter = AOIExporter(
        aoi_file=args.aoi_file,
        output_dir=args.output_dir,
        packs=args.packs,
        classes=args.classes,
        include=args.include,
        primary_decision=args.primary_decision,
        aoi_grid_min_pct=args.aoi_grid_min_pct,
        aoi_grid_inexact=args.aoi_grid_inexact,
        processes=args.processes,
        threads=args.threads,
        chunk_size=args.chunk_size,
        include_parcel_geometry=args.include_parcel_geometry,
        save_features=args.save_features,
        save_buildings=args.save_buildings,
        rollup_format=args.rollup_format,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
        overwrite_cache=args.overwrite_cache,
        compress_cache=args.compress_cache,
        country=args.country,
        alpha=args.alpha,
        beta=args.beta,
        prerelease=args.prerelease,
        only3d=args.only3d,
        since=args.since,
        until=args.until,
        endpoint=args.endpoint,
        url_root=args.url_root,
        system_version_prefix=args.system_version_prefix,
        system_version=args.system_version,
        log_level=args.log_level,
        api_key=args.api_key,  # Pass API key argument
        parcel_mode=not args.no_parcel_mode,  # Default to True unless --no-parcel-mode is set
        rapid=args.rapid,
        order=args.order,
        exclude_tiles_with_occlusion=args.exclude_tiles_with_occlusion,
    )
    exporter.run()


if __name__ == "__main__":
    # Register cleanup handlers
    atexit.register(cleanup_process_resources)
    signal.signal(signal.SIGTERM, lambda *args: cleanup_process_resources())

    main()
