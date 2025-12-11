import os

# Disable PROJ debug logging before importing pyproj/geopandas
# This prevents UnicodeDecodeError from PROJ's internal logging with non-ASCII characters
os.environ.setdefault("PROJ_DEBUG", "OFF")

import argparse
import concurrent.futures
import gc
import json
import logging
import multiprocessing
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
    DEPRECATED_CLASS_IDS,
    FEATURE_CLASS_DESCRIPTIONS,
    GRID_SIZE_DEGREES,
    LAT_PRIMARY_COL_NAME,
    LON_PRIMARY_COL_NAME,
    MAX_RETRIES,
    ROOF_AGE_INSTALLATION_DATE_FIELD,
    ROOF_ID,
    ROOF_INSTANCE_CLASS_ID,
    SINCE_COL_NAME,
    SQUARED_METERS_TO_SQUARED_FEET,
    SURVEY_RESOURCE_ID_COL_NAME,
    UNTIL_COL_NAME,
)
from nmaipy.api_common import format_error_summary_table, sanitize_error_message
from nmaipy.feature_api import FeatureApi
from nmaipy.roof_age_api import RoofAgeApi

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


def export_feature_class(
    features_gdf: gpd.GeoDataFrame,
    class_id: str,
    class_description: str,
    country: str,
    output_stem: Path,
    aoi_columns: list = None,
    export_csv: bool = True,
    export_parquet: bool = True,
) -> tuple:
    """
    Export features of a single class to CSV and/or GeoParquet.

    Args:
        features_gdf: GeoDataFrame with all features (will be filtered to class_id)
        class_id: UUID of the feature class to export
        class_description: Human-readable description (used in filename)
        country: Country code for units (us, au, etc.)
        output_stem: Base path for output files (without extension)
        aoi_columns: Additional columns from the AOI input file to include (e.g., ["Property Id"])
        export_csv: Whether to export CSV files (attributes only, no geometry)
        export_parquet: Whether to export GeoParquet files (with geometry)

    Returns:
        Tuple of (csv_path, parquet_path) or (None, None) if no features
    """
    # Filter to this class
    class_features = features_gdf[features_gdf["class_id"] == class_id].copy()
    if len(class_features) == 0:
        return (None, None)

    # Normalize class description for filename
    class_name = class_description.lower().replace(" ", "_").replace("-", "_")
    csv_path = Path(f"{output_stem}_{class_name}.csv")
    parquet_path = Path(f"{output_stem}_{class_name}_features.parquet")

    # Build output DataFrame using vectorized operations (much faster than iterrows)
    # Start with required columns
    flat_df = pd.DataFrame()
    added_cols = set()  # Track columns to avoid duplicates (e.g., area_sqm from Feature API vs Roof Age API)

    # Add aoi_id from index or column
    if class_features.index.name == AOI_ID_COLUMN_NAME:
        flat_df[AOI_ID_COLUMN_NAME] = class_features.index
        added_cols.add(AOI_ID_COLUMN_NAME)
    elif AOI_ID_COLUMN_NAME in class_features.columns:
        flat_df[AOI_ID_COLUMN_NAME] = class_features[AOI_ID_COLUMN_NAME].values
        added_cols.add(AOI_ID_COLUMN_NAME)

    # Add metadata columns (address fields, coordinates, etc.) if present
    # These come from the merged features parquet which includes AOI metadata
    metadata_cols = list(ADDRESS_FIELDS) + [
        "lat", "lon", "latitude", "longitude",  # Coordinates
        "match_quality", "matchQuality",  # Geocoding quality
        "geocode_source", "geocodeSource",  # Geocoding source
    ]
    # Add any additional AOI columns from the input file (e.g., "Property Id")
    if aoi_columns:
        metadata_cols = metadata_cols + [c for c in aoi_columns if c not in metadata_cols]
    for col in metadata_cols:
        if col in class_features.columns and col not in added_cols:
            flat_df[col] = class_features[col].values
            added_cols.add(col)

    # Add standard columns
    if "feature_id" in class_features.columns:
        flat_df["feature_id"] = class_features["feature_id"].values
        added_cols.add("feature_id")
    flat_df["class_id"] = class_id
    added_cols.add("class_id")
    flat_df["class_description"] = class_description
    added_cols.add("class_description")

    # Roof instances don't have confidence/fidelity (they have trust_score instead)
    # Only add confidence and fidelity for non-roof-instance classes
    if class_id != ROOF_INSTANCE_CLASS_ID:
        if "confidence" in class_features.columns:
            flat_df["confidence"] = class_features["confidence"].values
            added_cols.add("confidence")
        if "fidelity" in class_features.columns:
            flat_df["fidelity"] = class_features["fidelity"].values
            added_cols.add("fidelity")

    # Add area fields (vectorized) - skip if already added
    # Roof instances only have area_sqm (no clipped/unclipped distinction)
    if class_id == ROOF_INSTANCE_CLASS_ID:
        area_cols = ["area_sqm", "area_sqft"]
    else:
        area_cols = ["area_sqm", "clipped_area_sqm", "unclipped_area_sqm",
                     "area_sqft", "clipped_area_sqft", "unclipped_area_sqft"]
    for col in area_cols:
        if col in class_features.columns and col not in added_cols:
            flat_df[col] = class_features[col].values
            added_cols.add(col)

    # Add date fields (vectorized) - not applicable to roof instances
    if class_id != ROOF_INSTANCE_CLASS_ID:
        for col in ["survey_date", "mesh_date"]:
            if col in class_features.columns and col not in added_cols:
                flat_df[col] = class_features[col].values
                added_cols.add(col)

    # Add class-specific attributes for roof instances
    if class_id == ROOF_INSTANCE_CLASS_ID:
        # Add roof instance linkage columns (parent_id = parent roof, parent_iou = IoU with parent)
        for col in ["parent_id", "parent_iou"]:
            if col in class_features.columns and col not in added_cols:
                flat_df[col] = class_features[col].values
                added_cols.add(col)

        try:
            from nmaipy.feature_attributes import flatten_roof_instance_attributes_vectorized
            attr_df = flatten_roof_instance_attributes_vectorized(class_features, country=country)
            if attr_df is not None and len(attr_df) > 0:
                # Remove columns that would be duplicates
                attr_df = attr_df.drop(columns=[c for c in attr_df.columns if c in added_cols], errors="ignore")
                flat_df = pd.concat([flat_df.reset_index(drop=True), attr_df.reset_index(drop=True)], axis=1)
                added_cols.update(attr_df.columns)
        except ImportError:
            # Fall back to row-by-row if vectorized version not available
            from nmaipy.feature_attributes import flatten_roof_instance_attributes
            attr_records = []
            for _, row in class_features.iterrows():
                try:
                    attr_records.append(flatten_roof_instance_attributes(row, country=country))
                except Exception:
                    attr_records.append({})
            if attr_records:
                attr_df = pd.DataFrame(attr_records)
                # Remove columns that would be duplicates
                attr_df = attr_df.drop(columns=[c for c in attr_df.columns if c in added_cols], errors="ignore")
                flat_df = pd.concat([flat_df.reset_index(drop=True), attr_df.reset_index(drop=True)], axis=1)
        except Exception as e:
            logger.debug(f"Could not flatten roof instance attributes: {e}")

    # Add class-specific linkage columns for roofs (linking to roof instances)
    if class_id == ROOF_ID:
        for col in ["primary_child_roof_instance_feature_id", "primary_child_roof_instance_iou", "child_roof_instances", "child_roof_instance_count"]:
            if col in class_features.columns and col not in added_cols:
                flat_df[col] = class_features[col].values
                added_cols.add(col)

        # Add flattened attributes of the primary child roof instance
        # Look up roof instances from the full features_gdf and join on primary_child_roof_instance_feature_id
        if "primary_child_roof_instance_feature_id" in class_features.columns:
            # Get roof instances from the full features_gdf
            roof_instances = features_gdf[features_gdf["class_id"] == ROOF_INSTANCE_CLASS_ID].copy()
            if len(roof_instances) > 0 and "feature_id" in roof_instances.columns:
                from nmaipy.feature_attributes import flatten_roof_instance_attributes

                # Build a mapping from feature_id to flattened attributes
                ri_attrs = {}
                for _, ri_row in roof_instances.iterrows():
                    try:
                        attrs = flatten_roof_instance_attributes(ri_row, country=country, prefix="primary_child_roof_instance_")
                        ri_attrs[ri_row["feature_id"]] = attrs
                    except Exception:
                        pass

                # Add attributes for each roof based on its primary child
                if ri_attrs:
                    # Get all unique attribute keys
                    all_attr_keys = set()
                    for attrs in ri_attrs.values():
                        all_attr_keys.update(attrs.keys())

                    # Add columns for each attribute
                    for attr_key in sorted(all_attr_keys):
                        if attr_key not in added_cols:
                            flat_df[attr_key] = class_features["primary_child_roof_instance_feature_id"].apply(
                                lambda fid: ri_attrs.get(fid, {}).get(attr_key) if pd.notna(fid) else None
                            ).values
                            added_cols.add(attr_key)

    # Add mapbrowser link column
    # Uses geometry centroid for location and survey_date/installation_date for date
    if "geometry" in class_features.columns:
        def make_mapbrowser_link(row):
            try:
                geom = row.geometry
                if geom is None or geom.is_empty:
                    return None
                centroid = geom.centroid
                lat, lon = centroid.y, centroid.x

                # Use survey_date for Feature API classes, installation_date for roof instances
                date_val = None
                if class_id == ROOF_INSTANCE_CLASS_ID:
                    # Try to get installation_date - check both camelCase (API) and snake_case
                    if ROOF_AGE_INSTALLATION_DATE_FIELD in row.index:
                        date_val = row[ROOF_AGE_INSTALLATION_DATE_FIELD]
                    elif "installation_date" in row.index:
                        date_val = row["installation_date"]
                else:
                    if "survey_date" in row.index:
                        date_val = row["survey_date"]

                # Format date (remove dashes if present)
                if date_val is not None and pd.notna(date_val):
                    date_str = str(date_val).replace("-", "")[:8]  # YYYYMMDD format
                else:
                    date_str = ""

                if date_str:
                    return f"https://apps.nearmap.com/maps/#/@{lat},{lon},21.00z,0d/V/{date_str}?locationMarker"
                else:
                    return f"https://apps.nearmap.com/maps/#/@{lat},{lon},21.00z,0d?locationMarker"
            except Exception:
                return None

        flat_df["link"] = class_features.apply(make_mapbrowser_link, axis=1).values
        added_cols.add("link")

    # Save CSV (attributes only, no geometry)
    if export_csv:
        flat_df.to_csv(csv_path, index=False)
    else:
        csv_path = None

    # Save GeoParquet (with geometry)
    if export_parquet and "geometry" in class_features.columns:
        geo_df = gpd.GeoDataFrame(
            flat_df.copy(),
            geometry=class_features.geometry.values,
            crs=API_CRS
        )
        try:
            geo_df.to_parquet(parquet_path, index=False, schema_version="1.0.0")
        except (TypeError, ValueError):
            geo_df.to_parquet(parquet_path, index=False)
    else:
        parquet_path = None

    return (csv_path, parquet_path)


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
        help="List of AI packs (default: building)",
        type=str,
        nargs="+",
        required=False,
        default=["building"],
    )
    parser.add_argument(
        "--roof-age",
        help="Include Roof Age API data (US only). Adds roof instance features with installation dates.",
        action="store_true",
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
        help="Primary feature decision method: largest_intersection|nearest|optimal",
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
        "--aoi-grid-cell-size",
        help=f"Grid cell size in degrees for subdividing large AOIs (default: {GRID_SIZE_DEGREES}, approx 200m). Smaller values = finer grid.",
        type=float,
        required=False,
        default=GRID_SIZE_DEGREES,
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
        "--no-class-level-files",
        help="If set, disable per-feature-class CSV exports (e.g., roof.csv, roof_instance.csv). By default, these are enabled.",
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
        "--max-retries",
        help=f"Maximum number of retry attempts for failed API requests (default: {MAX_RETRIES})",
        type=int,
        required=False,
        default=MAX_RETRIES,
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


class NearmapAIExporter(BaseExporter):
    """
    Unified exporter for Nearmap AI data from Feature API and Roof Age API.

    Processes AOIs against both APIs in parallel, producing:
    - AOI-level rollup with attributes from all feature classes
    - Optional detailed feature exports (GeoParquet)
    - Separate error tracking per API
    """

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
        aoi_grid_cell_size=GRID_SIZE_DEGREES,  # Grid cell size in degrees for subdividing large AOIs
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
        api_key=None,
        parcel_mode=True,
        rapid=False,
        order=None,
        exclude_tiles_with_occlusion=False,
        roof_age=False,  # Include Roof Age API data
        class_level_files=True,  # Export per-feature-class CSV files (attributes only)
        max_retries=MAX_RETRIES,  # Maximum retry attempts for failed API requests
    ):
        # Initialize base exporter first
        super().__init__(
            output_dir=output_dir,
            processes=processes,
            chunk_size=chunk_size,
            log_level=log_level,
        )

        # Assign NearmapAIExporter-specific parameters to instance variables
        self.aoi_file = aoi_file
        self.packs = packs
        self.classes = classes
        self.include = include
        self.primary_decision = primary_decision
        self.aoi_grid_min_pct = aoi_grid_min_pct
        self.aoi_grid_inexact = aoi_grid_inexact
        self.aoi_grid_cell_size = aoi_grid_cell_size
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
        self.roof_age = roof_age
        self.class_level_files = class_level_files
        self.max_retries = max_retries

        # Validate roof_age usage
        if self.roof_age and self.country.lower() != "us":
            logger.warning(
                f"Roof Age API is currently only available for US properties. "
                f"Got country='{self.country}'. Roof age data will not be retrieved."
            )
            self.roof_age = False

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
        roof_age_api = None
        try:
            if self.cache_dir is None and not self.no_cache:
                cache_dir = Path(self.output_dir)
            else:
                cache_dir = (
                    Path(self.cache_dir) if self.cache_dir else Path(self.output_dir)
                )

            # Separate cache paths for each API
            if not self.no_cache:
                feature_api_cache_path = cache_dir / "cache" / "feature_api"
                roof_age_cache_path = cache_dir / "cache" / "roof_age"
            else:
                feature_api_cache_path = None
                roof_age_cache_path = None

            outfile = self.chunk_path / f"rollup_{chunk_id}.parquet"
            outfile_features = self.chunk_path / f"features_{chunk_id}.parquet"
            outfile_errors = self.chunk_path / f"feature_api_errors_{chunk_id}.parquet"
            outfile_roof_age_errors = self.chunk_path / f"roof_age_errors_{chunk_id}.parquet"
            if outfile.exists():
                return

            # Get additional parcel attributes from parcel geometry
            if isinstance(aoi_gdf, gpd.GeoDataFrame):
                rep_point = aoi_gdf.representative_point()
                aoi_gdf["query_aoi_lat"] = rep_point.y
                aoi_gdf["query_aoi_lon"] = rep_point.x

            # Get features from Feature API
            feature_api = FeatureApi(
                api_key=self.api_key(),
                cache_dir=feature_api_cache_path,
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
                grid_size=self.aoi_grid_cell_size,
                maxretry=self.max_retries,
                rapid=self.rapid,
                order=self.order,
                exclude_tiles_with_occlusion=self.exclude_tiles_with_occlusion,
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
                        # Sanitize URLs in messages before aggregating (truncate query params)
                        sanitized_messages = errors_df["message"].apply(sanitize_error_message)
                        error_counts = sanitized_messages.value_counts().to_dict()
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

                # Filter out deprecated feature classes early
                if len(features_gdf) > 0 and "class_id" in features_gdf.columns:
                    pre_filter_count = len(features_gdf)
                    features_gdf = features_gdf[~features_gdf["class_id"].isin(DEPRECATED_CLASS_IDS)]
                    if len(features_gdf) < pre_filter_count:
                        logger.debug(
                            f"Chunk {chunk_id}: Filtered {pre_filter_count - len(features_gdf)} deprecated features"
                        )

                mem = psutil.virtual_memory()
                self.logger.debug(
                    f"Chunk {chunk_id} failed {len(errors_df)} of {len(aoi_gdf)} AOI requests. "
                    f"Memory: {(mem.total - mem.available)/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB ({mem.percent}%)"
                )
                if len(errors_df) > 0:
                    if "message" in errors_df:
                        # Sanitize URLs in messages before aggregating (truncate query params)
                        sanitized_messages = errors_df["message"].apply(sanitize_error_message)
                        error_counts = sanitized_messages.value_counts().to_dict()
                        self.logger.debug(
                            f"Found {len(errors_df)} errors by type: {error_counts}"
                        )
                    else:
                        self.logger.debug(f"Found {len(errors_df)} errors")
                # Track Feature API success per AOI
                feature_api_errors_df = errors_df.copy()
                feature_api_success_aois = set(aoi_gdf.index) - set(errors_df.index) if len(errors_df) > 0 else set(aoi_gdf.index)

                # Query Roof Age API if enabled
                roof_age_gdf = gpd.GeoDataFrame(columns=[AOI_ID_COLUMN_NAME, "geometry"], crs=API_CRS)
                roof_age_errors_df = pd.DataFrame()

                if self.roof_age:
                    logger.debug(f"Chunk {chunk_id}: Querying Roof Age API for {len(aoi_gdf)} AOIs")

                    try:
                        roof_age_api = RoofAgeApi(
                            api_key=self.api_key(),
                            cache_dir=roof_age_cache_path,
                            overwrite_cache=self.overwrite_cache,
                            compress_cache=self.compress_cache,
                            threads=self.threads,
                            country=self.country,
                            progress_counters=progress_counters,
                        )
                        roof_age_gdf, roof_age_metadata_df, roof_age_errors_df = roof_age_api.get_roof_age_bulk(
                            aoi_gdf,
                        )
                        logger.debug(
                            f"Chunk {chunk_id}: Roof Age API returned {len(roof_age_gdf)} roof instances, "
                            f"{len(roof_age_errors_df)} errors"
                        )
                    except Exception as e:
                        logger.warning(f"Chunk {chunk_id}: Roof Age API query failed: {e}")
                        # Mark all AOIs as failed for roof age
                        roof_age_errors_df = pd.DataFrame({
                            AOI_ID_COLUMN_NAME: aoi_gdf.index.tolist(),
                            "status_code": [-1] * len(aoi_gdf),
                            "message": [str(e)] * len(aoi_gdf),
                        }).set_index(AOI_ID_COLUMN_NAME)

                # Track Roof Age API success per AOI
                roof_age_success_aois = set(aoi_gdf.index) - set(roof_age_errors_df.index) if len(roof_age_errors_df) > 0 else set(aoi_gdf.index)

                # If all Feature API requests failed, save errors and return
                if len(feature_api_errors_df) == len(aoi_gdf):
                    feature_api_errors_df.to_parquet(outfile_errors)
                    if len(roof_age_errors_df) > 0:
                        roof_age_errors_df.to_parquet(outfile_roof_age_errors)
                    return

                # Combine features from both APIs (roof instances are treated as a feature class)
                # Note: Most field mappings (area_sqm, confidence, fidelity, feature_id) are done
                # in roof_age_api._parse_response(). Here we only add country-specific sqft columns
                # and prepare for concat.
                if len(roof_age_gdf) > 0:
                    # Add sqft columns for US (sqm columns are set in roof_age_api._parse_response)
                    # Note: Roof instances only have 'area' (not clipped/unclipped distinction)
                    if self.country.lower() == "us" and "area_sqm" in roof_age_gdf.columns:
                        roof_age_gdf["area_sqft"] = roof_age_gdf["area_sqm"] * SQUARED_METERS_TO_SQUARED_FEET

                    # Ensure roof_age_gdf has aoi_id as index (Feature API returns index, Roof Age returns column)
                    if roof_age_gdf.index.name != AOI_ID_COLUMN_NAME and AOI_ID_COLUMN_NAME in roof_age_gdf.columns:
                        roof_age_gdf = roof_age_gdf.set_index(AOI_ID_COLUMN_NAME)

                    logger.debug(
                        f"Chunk {chunk_id}: Combining {len(features_gdf)} Feature API features with "
                        f"{len(roof_age_gdf)} Roof Age features"
                    )
                    dfs_to_concat = [df for df in [features_gdf, roof_age_gdf] if len(df) > 0]
                    if dfs_to_concat:
                        # Concatenating DataFrames with different schemas (Feature API vs Roof Age API)
                        # triggers FutureWarning about all-NA column dtype inference - this is expected
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*concatenation with empty or all-NA.*")
                            features_gdf = gpd.GeoDataFrame(
                                pd.concat(dfs_to_concat, ignore_index=False),
                                crs=API_CRS
                            )
                    logger.debug(f"Chunk {chunk_id}: Combined features_gdf has {len(features_gdf)} rows")

                    # Perform spatial matching between roof instances and roofs
                    roofs_gdf = features_gdf[features_gdf["class_id"] == ROOF_ID].copy()
                    if len(roofs_gdf) > 0 and len(roof_age_gdf) > 0:
                        logger.debug(
                            f"Chunk {chunk_id}: Linking {len(roof_age_gdf)} roof instances to {len(roofs_gdf)} roofs"
                        )
                        roof_age_gdf_linked, roofs_gdf_linked = parcels.link_roof_instances_to_roofs(
                            roof_age_gdf, roofs_gdf
                        )

                        # Update features_gdf with linked data
                        # Remove old roof instances and roofs, add linked versions
                        non_roof_features = features_gdf[
                            (features_gdf["class_id"] != ROOF_ID) &
                            (features_gdf["class_id"] != roof_age_gdf["class_id"].iloc[0])
                        ]
                        dfs_to_concat = [df for df in [non_roof_features, roofs_gdf_linked, roof_age_gdf_linked] if len(df) > 0]
                        if dfs_to_concat:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", message=".*concatenation with empty or all-NA.*")
                                features_gdf = gpd.GeoDataFrame(
                                    pd.concat(dfs_to_concat, ignore_index=False),
                                    crs=API_CRS
                                )
                        logger.debug(
                            f"Chunk {chunk_id}: After linking, features_gdf has {len(features_gdf)} rows"
                        )

                # Create rollup
                rollup_df = parcels.parcel_rollup(
                    aoi_gdf,
                    features_gdf,
                    classes_df,
                    country=self.country,
                    primary_decision=self.primary_decision,
                )

                # Add API success columns (Y/N)
                rollup_df["feature_api_success"] = rollup_df.index.map(
                    lambda x: "Y" if x in feature_api_success_aois else "N"
                )
                if self.roof_age:
                    rollup_df["roof_age_api_success"] = rollup_df.index.map(
                        lambda x: "Y" if x in roof_age_success_aois else "N"
                    )

                # Save Roof Age API errors separately
                if self.roof_age and len(roof_age_errors_df) > 0:
                    roof_age_errors_df.to_parquet(outfile_roof_age_errors)

                # Use Feature API errors as the main errors file
                errors_df = feature_api_errors_df

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

            # Use rollup_df as base to preserve all AOIs (including those where Feature API
            # failed but Roof Age API succeeded). Left-merge with metadata_df to add
            # survey metadata where available.
            final_df = rollup_df.merge(
                metadata_df, on=AOI_ID_COLUMN_NAME, how="left"
            ).merge(
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
                # Convert errors_df to GeoDataFrame if it has geometry (from failed grid squares)
                # This ensures proper geoparquet output that can be read in GIS software
                if "geometry" in errors_df.columns and len(errors_df) > 0:
                    errors_gdf = gpd.GeoDataFrame(errors_df, geometry="geometry", crs=API_CRS)
                    errors_gdf.to_parquet(outfile_errors)
                else:
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
                            if val is None:
                                return None
                            # Handle scalar pd.isna check carefully - it returns array for array input
                            try:
                                if pd.isna(val):
                                    return None
                            except (TypeError, ValueError):
                                # pd.isna fails on arrays/lists - handle below
                                pass
                            if isinstance(val, dict):
                                return json.dumps(val)
                            if isinstance(val, (list, np.ndarray)):
                                return json.dumps(val if isinstance(val, list) else val.tolist())
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
                                outfile_features, index=False, schema_version="1.0.0"
                            )
                        except (TypeError, ValueError) as e:
                            # Fallback for older geopandas or pyarrow versions
                            self.logger.debug(
                                f"Could not use schema_version parameter: {e}. Falling back to default."
                            )
                            final_features_df.to_parquet(outfile_features, index=False)
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
            # Clean up API clients to close network connections
            if "feature_api" in locals() and feature_api is not None:
                try:
                    feature_api.cleanup()
                    del feature_api
                except:
                    pass

            if "roof_age_api" in locals() and roof_age_api is not None:
                try:
                    roof_age_api.cleanup()
                    del roof_age_api
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
        self.logger.info(f"nmaipy version: {__version__}")
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

        # Filter out deprecated classes from rollups
        classes_df = classes_df[~classes_df.index.isin(DEPRECATED_CLASS_IDS)]

        # Add Roof Instance class to classes_df when roof_age is enabled
        # This allows parcel_rollup to generate rollup columns for roof instances
        if self.roof_age:
            roof_instance_row = pd.DataFrame(
                {"description": [FEATURE_CLASS_DESCRIPTIONS[ROOF_INSTANCE_CLASS_ID]]},
                index=[ROOF_INSTANCE_CLASS_ID],
            )
            classes_df = pd.concat([classes_df, roof_instance_row])

        # Modify output file paths using the AOI file name
        # Renamed from {stem}.csv to {stem}_aoi_rollup.csv for clarity
        outpath = self.final_path / f"{Path(aoi_path).stem}_aoi_rollup.{self.rollup_format}"
        outpath_features = self.final_path / f"{Path(aoi_path).stem}_features.parquet"
        outpath_buildings = (
            self.final_path / f"{Path(aoi_path).stem}_buildings.{self.rollup_format}"
        )
        # Base stem for per-class output files
        output_stem = self.final_path / Path(aoi_path).stem

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

        # Validate lat/lon columns exist for primary_decision modes that require them
        if self.primary_decision in ("nearest", "optimal"):
            missing_cols = []
            if LAT_PRIMARY_COL_NAME not in aoi_gdf.columns:
                missing_cols.append(LAT_PRIMARY_COL_NAME)
            if LON_PRIMARY_COL_NAME not in aoi_gdf.columns:
                missing_cols.append(LON_PRIMARY_COL_NAME)
            if missing_cols:
                self.logger.error(
                    f"primary_decision='{self.primary_decision}' requires columns {missing_cols} "
                    f"in the input AOI file. These columns should contain the lat/lon coordinates "
                    f"of the point to use for primary feature selection. "
                    f"Available columns: {list(aoi_gdf.columns)}"
                )
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
        # If roof_age is enabled, each AOI gets both Feature API and Roof Age API queries
        initial_aoi_count = len(aoi_gdf) - skipped_aois
        if self.roof_age:
            initial_aoi_count *= 2

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
        # Collect and save Feature API errors
        outpath_feature_api_errors = self.final_path / f"{Path(aoi_path).stem}_feature_api_errors.csv"
        outpath_feature_api_errors_geoparquet = self.final_path / f"{Path(aoi_path).stem}_feature_api_errors.parquet"
        self.logger.debug(f"Collecting Feature API errors")
        feature_api_errors = []
        for cp in self.chunk_path.glob(f"feature_api_errors_{Path(aoi_path).stem}_*.parquet"):
            # Use geopandas to read to preserve geometry if present
            try:
                feature_api_errors.append(gpd.read_parquet(cp))
            except Exception:
                # Fall back to pandas if not a valid geoparquet
                feature_api_errors.append(pd.read_parquet(cp))
        if len(feature_api_errors) > 0:
            feature_api_errors = pd.concat(feature_api_errors)
        else:
            feature_api_errors = pd.DataFrame()

        # Collect and save Roof Age API errors (if roof_age was enabled)
        roof_age_errors = pd.DataFrame()
        if self.roof_age:
            outpath_roof_age_errors = self.final_path / f"{Path(aoi_path).stem}_roof_age_errors.csv"
            outpath_roof_age_errors_geoparquet = self.final_path / f"{Path(aoi_path).stem}_roof_age_errors.parquet"
            self.logger.debug(f"Collecting Roof Age API errors")
            roof_age_errors_list = []
            for cp in self.chunk_path.glob(f"roof_age_errors_{Path(aoi_path).stem}_*.parquet"):
                roof_age_errors_list.append(pd.read_parquet(cp))
            if len(roof_age_errors_list) > 0:
                roof_age_errors = pd.concat(roof_age_errors_list)

        # Helper function to save errors
        def save_errors_to_files(errors_df, outpath_csv, outpath_parquet, error_type):
            # Handle both cases: AOI_ID_COLUMN_NAME as column or as index
            has_aoi_id = AOI_ID_COLUMN_NAME in errors_df.columns or errors_df.index.name == AOI_ID_COLUMN_NAME
            if len(errors_df) > 0 and has_aoi_id:
                # If aoi_id is the index, reset it to be a column for merging
                if errors_df.index.name == AOI_ID_COLUMN_NAME:
                    errors_df = errors_df.reset_index()
                aoi_gdf_for_merge = aoi_gdf.reset_index()

                if isinstance(aoi_gdf, gpd.GeoDataFrame):
                    # Check if errors_df already has geometry (from failed grid squares)
                    # If so, preserve it and merge AOI geometry under a different name
                    if "geometry" in errors_df.columns:
                        # Errors already have grid cell geometry - merge AOI geometry as aoi_geometry
                        errors_with_context = errors_df.merge(
                            aoi_gdf_for_merge[[AOI_ID_COLUMN_NAME, "geometry"]].rename(
                                columns={"geometry": "aoi_geometry"}
                            ),
                            on=AOI_ID_COLUMN_NAME,
                            how="left",
                        )
                        # Use the grid cell geometry as primary (it's more specific for troubleshooting)
                        errors_gdf = gpd.GeoDataFrame(
                            errors_with_context, geometry="geometry", crs=API_CRS
                        )
                    else:
                        # No geometry yet - merge AOI geometry
                        errors_with_context = errors_df.merge(
                            aoi_gdf_for_merge[[AOI_ID_COLUMN_NAME, "geometry"]],
                            on=AOI_ID_COLUMN_NAME,
                            how="left",
                        )
                        if "geometry" in errors_with_context.columns:
                            errors_gdf = gpd.GeoDataFrame(
                                errors_with_context, geometry="geometry", crs=aoi_gdf.crs
                            )
                        else:
                            errors_gdf = errors_with_context
                else:
                    merge_cols = [col for col in aoi_gdf_for_merge.columns if col != AOI_ID_COLUMN_NAME]
                    if AOI_ID_COLUMN_NAME in aoi_gdf_for_merge.columns:
                        merge_cols.insert(0, AOI_ID_COLUMN_NAME)
                    errors_gdf = errors_df.merge(
                        aoi_gdf_for_merge[merge_cols],
                        on=AOI_ID_COLUMN_NAME,
                        how="left",
                    )

                # Log error summary as ASCII table
                status_counts = None
                message_counts = None
                if "status_code" in errors_df.columns:
                    status_counts = errors_df["status_code"].value_counts()
                if "message" in errors_df.columns:
                    # Sanitize URLs in messages before aggregating (truncate query params)
                    sanitized_messages = errors_df["message"].apply(sanitize_error_message)
                    message_counts = sanitized_messages.value_counts()

                error_table = format_error_summary_table(status_counts, message_counts)
                self.logger.info(f"{error_type}: {len(errors_df)} failures{error_table}")

                # Save CSV
                if isinstance(aoi_gdf, gpd.GeoDataFrame):
                    errors_df.to_csv(outpath_csv, index=False)
                else:
                    errors_gdf.to_csv(outpath_csv, index=False)

                # Save GeoParquet for geometry mode
                if isinstance(errors_gdf, gpd.GeoDataFrame) and len(errors_gdf) > 0:
                    self.logger.info(f"Saving {error_type} errors as geoparquet to {outpath_parquet}")
                    errors_gdf.to_parquet(outpath_parquet, index=False)
            else:
                self.logger.info(f"{error_type}: No failures")

        # Save Feature API errors
        save_errors_to_files(
            feature_api_errors,
            outpath_feature_api_errors,
            outpath_feature_api_errors_geoparquet,
            "Feature API"
        )

        # Save Roof Age API errors
        if self.roof_age:
            save_errors_to_files(
                roof_age_errors,
                outpath_roof_age_errors,
                outpath_roof_age_errors_geoparquet,
                "Roof Age API"
            )
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

        # Export per-class CSV and GeoParquet files if enabled
        if self.class_level_files:
            self.logger.info("Exporting per-feature-class CSV and GeoParquet files...")

            # Load combined features for per-class export
            features_gdf = None
            if self.save_features and outpath_features.exists():
                # Read from the final merged parquet we just created
                try:
                    features_gdf = gpd.read_parquet(outpath_features)
                except Exception as e:
                    self.logger.warning(f"Failed to read {outpath_features}: {e}")

            if features_gdf is None:
                # Fall back to reading from chunk files
                feature_paths = [
                    p for p in self.chunk_path.glob(f"features_{Path(aoi_path).stem}_*.parquet")
                ]
                if feature_paths:
                    all_features = []
                    for fp in feature_paths:
                        try:
                            chunk_gdf = gpd.read_parquet(fp)
                            if len(chunk_gdf) > 0:
                                all_features.append(chunk_gdf)
                        except Exception as e:
                            self.logger.warning(f"Failed to read {fp}: {e}")
                    if all_features:
                        features_gdf = gpd.GeoDataFrame(
                            pd.concat(all_features, ignore_index=False),
                            crs=API_CRS
                        )

            if features_gdf is not None and len(features_gdf) > 0 and "class_id" in features_gdf.columns:
                # Get unique classes in the data
                unique_classes = features_gdf["class_id"].dropna().unique()
                self.logger.debug(f"Found {len(unique_classes)} unique feature classes to export")

                # Build class_id -> description mapping
                class_descriptions = {**FEATURE_CLASS_DESCRIPTIONS}
                # Add descriptions from classes_df if available
                if classes_df is not None and "description" in classes_df.columns:
                    for class_id in classes_df.index:
                        class_descriptions[class_id] = classes_df.loc[class_id, "description"]

                # Extract input-file columns from AOI (excluding system columns)
                # These will be added to the per-class exports
                system_columns = {
                    AOI_ID_COLUMN_NAME, "geometry", SINCE_COL_NAME, UNTIL_COL_NAME,
                    SURVEY_RESOURCE_ID_COL_NAME, "query_aoi_lat", "query_aoi_lon",
                }
                aoi_input_columns = [
                    c for c in aoi_gdf.columns
                    if c not in system_columns and c not in ADDRESS_FIELDS
                ]

                # Export each class
                for class_id in unique_classes:
                    description = class_descriptions.get(class_id, f"class_{class_id[:8]}")
                    csv_path, parquet_path = export_feature_class(
                        features_gdf=features_gdf,
                        class_id=class_id,
                        class_description=description,
                        country=self.country,
                        output_stem=output_stem,
                        aoi_columns=aoi_input_columns,
                        export_csv=self.class_level_files,
                        export_parquet=self.class_level_files and self.save_features,
                    )
                    if csv_path or parquet_path:
                        files = [f.name for f in [csv_path, parquet_path] if f]
                        self.logger.info(f"  Exported {description}: {', '.join(files)}")
            else:
                self.logger.info("No features found for per-class export")


# Backward compatibility alias
AOIExporter = NearmapAIExporter


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
    exporter = NearmapAIExporter(
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
        api_key=args.api_key,
        parcel_mode=not args.no_parcel_mode,
        rapid=args.rapid,
        order=args.order,
        exclude_tiles_with_occlusion=args.exclude_tiles_with_occlusion,
        roof_age=args.roof_age,
        class_level_files=not args.no_class_level_files,
        aoi_grid_cell_size=args.aoi_grid_cell_size,
        max_retries=args.max_retries,
    )
    exporter.run()


if __name__ == "__main__":
    # Register cleanup handlers
    atexit.register(cleanup_process_resources)
    signal.signal(signal.SIGTERM, lambda *args: cleanup_process_resources())

    main()
