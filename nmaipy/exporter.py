import os

# Disable PROJ debug logging before importing pyproj/geopandas
# This prevents UnicodeDecodeError from PROJ's internal logging with non-ASCII characters
os.environ.setdefault("PROJ_DEBUG", "OFF")

import argparse
import collections
import concurrent.futures
import cProfile
import gc
import json
import logging
import multiprocessing
import pstats
import re
import resource
import sys
import traceback
import warnings
from enum import Enum
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyproj
import shapely
import shapely.geometry
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")

import atexit
import signal
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone

import psutil

from nmaipy import log, parcels, storage
from nmaipy.__version__ import __version__
from nmaipy.api_common import (
    collect_latency_stats_from_apis,
    combine_chunk_latency_stats,
    compute_global_latency_stats,
    format_error_summary_table,
    sanitize_error_message,
    save_chunk_latency_stats,
)
from nmaipy.base_exporter import BaseExporter, resource_postfix
from nmaipy.cgroup_memory import get_cpu_info_cgroup_aware, get_memory_info_cgroup_aware
from nmaipy.constants import (
    ADDRESS_FIELDS,
    AOI_ID_COLUMN_NAME,
    API_CRS,
    AREA_CRS,
    BUILDING_LIFECYCLE_ID,
    BUILDING_NEW_ID,
    BUILDING_STYLE_CLASS_IDS,
    DEFAULT_URL_ROOT,
    DEPRECATED_CLASS_IDS,
    FEATURE_CLASS_DESCRIPTIONS,
    FEATURE_PREFETCH_WORKERS,
    GRID_SIZE_DEGREES,
    IMPERIAL_COUNTRIES,
    LAT_PRIMARY_COL_NAME,
    LON_PRIMARY_COL_NAME,
    MAX_RETRIES,
    METERS_TO_FEET,
    PARALLEL_READ_WORKERS,
    PER_CLASS_FILE_CLASS_IDS,
    PRIMARY_FEATURE_COLUMN_TO_CLASS,
    ROOF_AGE_PREFIX_COLUMNS,
    ROOF_ID,
    ROOF_INSTANCE_CLASS_ID,
    S3_PARALLEL_READ_WORKERS,
    SINCE_COL_NAME,
    SQUARED_METERS_TO_SQUARED_FEET,
    SURVEY_RESOURCE_ID_COL_NAME,
    UNTIL_COL_NAME,
    _write_class_descriptions,
)
from nmaipy.feature_api import FeatureApi
from nmaipy.feature_attributes import (
    FALSE_STRING,
    TRUE_STRING,
    calculate_roof_age_years,
    convert_bool_columns_to_yn,
    flatten_building_attributes,
    flatten_roof_attributes,
)
from nmaipy.parcels import (
    extract_rsi_from_feature,
    build_parent_lookup,
    link_roofs_to_buildings,
    resolve_footprint_rsi,
)
from nmaipy.readme_generator import ReadmeGenerator
from nmaipy.roof_age_api import RoofAgeApi


def _read_parquet_chunks_parallel(
    paths: List[str],
    max_workers: int = PARALLEL_READ_WORKERS,
    desc: str = "Reading chunks",
    logger=None,
    strict: bool = True,
    geo: bool = True,
) -> List[pd.DataFrame]:
    """
    Read parquet files in parallel using threads, with a tqdm progress bar.

    Each file is attempted as geoparquet first (gpd.read_parquet), falling back
    to plain parquet (pd.read_parquet) on failure. Empty DataFrames are excluded.

    Args:
        paths: List of parquet file paths (local or S3 URIs).
        max_workers: Number of concurrent reader threads.
        desc: Description for the tqdm progress bar.
        logger: Optional logger for warnings on read failures.
        strict: If True (default), raise on any read failure. If False,
            log a warning and continue, collecting as many results as possible.
        geo: If True (default), attempt geoparquet read first. If False,
            use plain pd.read_parquet directly (avoids a wasted S3 round-trip
            for files known to have no geometry, such as error files).

    Returns:
        List of non-empty DataFrames, in arbitrary order.

    Raises:
        RuntimeError: If strict=True and any file fails to read.
    """
    if not paths:
        return []

    def _read_one(path: str):
        if geo:
            try:
                df = gpd.read_parquet(path)
            except Exception:
                df = pd.read_parquet(path)
        else:
            df = pd.read_parquet(path)
        if len(df) > 0:
            return df
        return None

    results = []
    failed_paths = []
    empty_count = 0
    last_resource_update = 0.0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(_read_one, p): p for p in paths}
        pbar = tqdm(
            as_completed(future_to_path),
            total=len(paths),
            desc=desc,
            file=sys.stdout,
            position=0,
            leave=True,
        )
        for future in pbar:
            now = time.time()
            if now - last_resource_update >= 5.0:
                pbar.set_postfix_str(resource_postfix())
                last_resource_update = now
            path = future_to_path[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    empty_count += 1
            except Exception as e:
                if strict:
                    raise RuntimeError(f"Failed to read {path}: {e}") from e
                failed_paths.append(path)
                if logger:
                    logger.warning(f"Failed to read {path}: {e}")
    if logger and (failed_paths or empty_count > 0):
        parts = []
        if failed_paths:
            parts.append(f"{len(failed_paths)} failed")
        if empty_count > 0:
            parts.append(f"{empty_count} empty")
        logger.info(f"Read {len(results)}/{len(paths)} chunks with data " f"({', '.join(parts)})")
    return results


def _add_is_primary_column(
    features_gdf: gpd.GeoDataFrame,
    rollup_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Add is_primary column to features based on primary feature IDs in rollup.

    Matches on (aoi_id, feature_id, class_id) to ensure class-specific primary:
    - Roofs are marked primary based on primary_roof_feature_id
    - Buildings are marked primary based on primary_building_feature_id

    Args:
        features_gdf: GeoDataFrame with features to mark
        rollup_df: DataFrame with primary_*_feature_id columns from parcel rollup

    Returns:
        features_gdf with is_primary column added
    """
    if rollup_df is None or len(rollup_df) == 0:
        features_gdf["is_primary"] = False
        return features_gdf

    if "feature_id" not in features_gdf.columns or "class_id" not in features_gdf.columns:
        features_gdf["is_primary"] = False
        return features_gdf

    # Build DataFrame of primary features: (aoi_id, feature_id, class_id)
    primary_records = []
    for col, class_id in PRIMARY_FEATURE_COLUMN_TO_CLASS.items():
        if col in rollup_df.columns:
            for aoi_id, feature_id in rollup_df[col].dropna().items():
                primary_records.append(
                    {
                        AOI_ID_COLUMN_NAME: aoi_id,
                        "feature_id": str(feature_id),
                        "class_id": class_id,
                    }
                )

    if not primary_records:
        features_gdf["is_primary"] = False
        return features_gdf

    primary_df = pd.DataFrame(primary_records)

    # Ensure feature_id is string for consistent matching
    features_gdf["feature_id"] = features_gdf["feature_id"].astype(str)

    # Get aoi_id as column (may be index)
    if features_gdf.index.name == AOI_ID_COLUMN_NAME:
        features_gdf = features_gdf.reset_index()
        reset_index = True
    else:
        reset_index = False

    # Merge to mark primary features
    features_gdf = features_gdf.merge(
        primary_df.assign(_is_primary=True),
        on=[AOI_ID_COLUMN_NAME, "feature_id", "class_id"],
        how="left",
    )
    features_gdf["is_primary"] = features_gdf["_is_primary"].notna()
    features_gdf = features_gdf.drop(columns=["_is_primary"])

    # Restore index if needed
    if reset_index:
        features_gdf = features_gdf.set_index(AOI_ID_COLUMN_NAME)

    return features_gdf


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
                            cleaned_comp = {k: v for k, v in comp.items() if k != "internalClassId"}
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


def _group_children_by_aoi(
    non_roof_features: gpd.GeoDataFrame,
    features_gdf: gpd.GeoDataFrame,
) -> dict:
    """Group non-roof child features by aoi_id for efficient per-AOI lookup.

    Child features only need to be from the same AOI as the parent feature, since
    features are queried per-AOI and cannot span AOI boundaries. Grouping up front
    reduces child features from ~270k globally to ~86 per AOI.

    Note: returns views into the source DataFrame (no .copy()) since downstream
    consumers (flatten_roof_attributes / calculate_child_feature_attributes) do
    not mutate the child features.

    Returns:
        Dict mapping aoi_id -> GeoDataFrame of non-roof features for that AOI.
        Empty dict if aoi_id column is not present or source is empty.
    """
    source = non_roof_features if non_roof_features is not None else features_gdf[features_gdf["class_id"] != ROOF_ID]
    if len(source) == 0:
        return {}
    if AOI_ID_COLUMN_NAME not in source.columns:
        return {}
    return {aoi: group for aoi, group in source.groupby(AOI_ID_COLUMN_NAME)}


def _batch_project_geometries(
    parent_gdf: gpd.GeoDataFrame,
    child_by_aoi: dict,
    country: str,
) -> tuple:
    """Batch-project parent and child geometries to the local area CRS.

    Projecting all geometries up front avoids initializing a PROJ transformer
    per row inside the iterrows loop (the dominant cost in per-class export).

    Args:
        parent_gdf: GeoDataFrame whose geometry column will be projected.
        child_by_aoi: Dict mapping aoi_id -> GeoDataFrame of child features.
        country: Country code used to select the projected CRS.

    Returns:
        (parent_projected, child_proj_by_aoi) where parent_projected is a
        GeoSeries and child_proj_by_aoi is a dict mapping aoi_id -> projected
        GeoSeries.
    """
    projected_crs = AREA_CRS[country.lower()]
    parent_projected = parent_gdf.geometry.to_crs(projected_crs)
    child_proj_by_aoi = {}
    for aoi_id, children in child_by_aoi.items():
        if len(children) > 0:
            child_proj_by_aoi[aoi_id] = children.geometry.to_crs(projected_crs)
    return parent_projected, child_proj_by_aoi


_LARGE_TYPE_MAP = {
    pa.string(): pa.large_string(),
    pa.binary(): pa.large_binary(),
}


def _promote_schema_types(schema: pa.Schema, column_types: dict = None) -> pa.Schema:
    """Promote null-type fields and 32-bit string/binary to 64-bit equivalents.

    This prevents ArrowInvalid offset overflow when many chunks are written
    to the same ParquetWriter, and avoids null-type columns that can't accept
    real data in later chunks.

    Args:
        schema: The PyArrow schema to promote.
        column_types: Optional dict mapping column name to the first non-null
            PyArrow type seen across all chunks (from a pre-scan).  When provided,
            null fields are promoted to the real type (itself promoted to 64-bit
            if string/binary).  When omitted, null fields default to large_string.
    """
    promoted = []
    for field in schema:
        if field.type == pa.null():
            if column_types is not None:
                real_type = column_types.get(field.name, pa.large_string())
                real_type = _LARGE_TYPE_MAP.get(real_type, real_type)
            else:
                real_type = pa.large_string()
            promoted.append(pa.field(field.name, real_type, nullable=True))
        elif field.type in _LARGE_TYPE_MAP:
            promoted.append(
                pa.field(
                    field.name,
                    _LARGE_TYPE_MAP[field.type],
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
            )
        else:
            promoted.append(field)
    return pa.schema(promoted, metadata=schema.metadata)


def _cast_table_to_schema(table: pa.Table, target_schema: pa.Schema) -> pa.Table:
    """Cast a table's columns to match a target schema (type promotions only)."""
    if table.schema.equals(target_schema):
        return table
    arrays = []
    for field in target_schema:
        col = table.column(field.name)
        if col.type == field.type:
            arrays.append(col)
        elif col.type == pa.null():
            arrays.append(pa.nulls(len(table), type=field.type))
        else:
            try:
                arrays.append(col.cast(field.type))
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                logger.warning(
                    f"_cast_table_to_schema: cannot cast '{field.name}' "
                    f"from {col.type} to {field.type}, replacing with nulls"
                )
                arrays.append(pa.nulls(len(table), type=field.type))
    return pa.Table.from_arrays(arrays, schema=target_schema)


def _unify_and_concat_tables(tables: list[pa.Table]) -> pa.Table:
    """Concatenate tables after unifying their schemas.

    Uses ``pa.unify_schemas(promote_options="permissive")`` to resolve type
    mismatches across chunks (e.g. int64 vs float64 when nullable integers
    are inferred as float, int32 vs int64, timestamp unit differences, etc.),
    then applies ``_promote_schema_types`` for string/binary → large_string/
    large_binary offset overflow protection.
    """
    unified = pa.unify_schemas([t.schema for t in tables], promote_options="permissive")
    target = _promote_schema_types(unified)
    cast_tables = [_reconcile_table_schema(t, target) for t in tables]
    return pa.concat_tables(cast_tables)


def _reconcile_table_schema(table: pa.Table, ref_schema: pa.Schema) -> pa.Table:
    """Reconcile an Arrow table's schema to match a reference schema.

    Adds missing columns as null arrays and casts compatible types.
    Used when appending to a ParquetWriter whose schema was established
    by the first chunk written.
    """
    if table.schema.equals(ref_schema):
        return table
    arrays = []
    for field in ref_schema:
        if field.name in table.column_names:
            col = table.column(field.name)
            if col.type == field.type:
                arrays.append(col)
            elif col.type == pa.null():
                arrays.append(pa.nulls(len(table), type=field.type))
            else:
                try:
                    arrays.append(col.cast(field.type))
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    arrays.append(pa.nulls(len(table), type=field.type))
        else:
            arrays.append(pa.nulls(len(table), type=field.type))
    return pa.Table.from_arrays(arrays, schema=ref_schema)


# Raw RSI columns excluded from per-class output; the resolved
# roof_spotlight_index/confidence/model_version columns are added separately.
_EXCLUDE_RSI = {
    "primary_child_roof_roof_spotlight_index",
    "primary_child_roof_roof_spotlight_index_confidence",
    "primary_child_roof_roof_spotlight_index_model_version",
}

# RSI column names emitted by _resolve_rsi_batch.
_RSI_COLUMNS = (
    "roof_spotlight_index",
    "roof_spotlight_index_confidence",
    "roof_spotlight_index_model_version",
)


def _resolve_rsi_batch(n_rows, resolve_fn):
    """Resolve RSI for *n_rows* rows, returning (vals, conf, mver) numpy arrays.

    *resolve_fn(i)* is called for each row index and must return a dict with
    roof_spotlight_index / roof_spotlight_index_confidence /
    roof_spotlight_index_model_version keys, or a falsy value (None/{}).

    Returns None if no RSI was resolved for any row.
    """
    vals = np.full(n_rows, None, dtype=object)
    conf = np.full(n_rows, None, dtype=object)
    mver = np.full(n_rows, None, dtype=object)

    for i in range(n_rows):
        resolved = resolve_fn(i)
        if resolved:
            vals[i] = resolved.get("roof_spotlight_index")
            conf[i] = resolved.get("roof_spotlight_index_confidence")
            mver[i] = resolved.get("roof_spotlight_index_model_version")

    if not np.any(pd.notna(vals)):
        return None
    return vals, conf, mver


def _dataframe_to_records_with_index(df):
    """Convert DataFrame to list of dicts, re-injecting the index if it is AOI_ID_COLUMN_NAME.

    to_dict("records") drops the index. When the DataFrame is indexed by AOI ID
    (common after set_index in rollup/feature pipelines), this helper preserves it.
    """
    records = df.to_dict("records")
    if df.index.name == AOI_ID_COLUMN_NAME:
        for rec, idx_val in zip(records, df.index):
            rec[AOI_ID_COLUMN_NAME] = idx_val
    return records


def _description_to_cname(description: str) -> str:
    """Convert a class description like 'Roof Instance' to a filename-safe slug like 'roof_instance'."""
    return re.sub(r"[^a-z0-9]+", "_", description.lower()).strip("_")


def _per_class_chunk_regexes(cname: str) -> tuple:
    """Return compiled regexes for matching tabular and geo per-class chunk filenames.

    Chunk files are named {cname}_{chunk_id}.parquet (tabular) and
    {cname}_features_{chunk_id}.parquet (geo). The regex anchors on a digit after the
    prefix so that e.g. "roof_*" does not match "roof_instance_*".
    """
    tabular_re = re.compile(rf"^{re.escape(cname)}_\d+\.parquet$")
    geo_re = re.compile(rf"^{re.escape(cname)}_features_\d+\.parquet$")
    return tabular_re, geo_re


def _compute_all_per_class_data(
    chunk_gdf: gpd.GeoDataFrame,
    country: str,
    aoi_input_columns: list,
    whitelisted_classes: list = None,
    logger_instance=None,
) -> dict:
    """Compute per-class tabular + geo Arrow tables for all whitelisted classes in a chunk.

    Handles cross-class data preparation (roof linkage, roof attrs cache, IoU linkage) once,
    then calls _compute_feature_class_data() for each class. Returns a dict of
    {class_id: {"tabular": pa.Table, "geo": pa.Table}}.

    This is the shared core used by both process_chunk() (in-memory path) and
    _merge_per_class_chunks() (fallback recomputation path).
    """
    if whitelisted_classes is None:
        whitelisted_classes = sorted(PER_CLASS_FILE_CLASS_IDS)
    _log = logger_instance or logger
    cross_class_gdf = chunk_gdf

    chunk_class_ids = set(chunk_gdf["class_id"].dropna().unique()) if "class_id" in chunk_gdf.columns else set()

    # Build parent lookup once for all classes in this chunk
    shared_parent_lookup = (
        build_parent_lookup(cross_class_gdf) if cross_class_gdf is not None and len(cross_class_gdf) > 0 else {}
    )

    # Build class_descriptions from chunk data
    chunk_class_descriptions = {}
    if "class_id" in chunk_gdf.columns and "description" in chunk_gdf.columns:
        for cid, desc in zip(
            chunk_gdf["class_id"].values,
            chunk_gdf["description"].values,
        ):
            if cid and desc and cid not in chunk_class_descriptions:
                chunk_class_descriptions[cid] = desc

    # Prepare cross-class data for dependent classes (roof, building, building lifecycle)
    dependent_class_ids = [c for c in [BUILDING_NEW_ID, BUILDING_LIFECYCLE_ID, ROOF_ID] if c in whitelisted_classes]
    roof_features_chunk = None
    roof_instance_chunk = None
    non_roof_chunk = None
    roof_attrs_cache_chunk = None

    roof_to_building_lookup = {}
    has_dependent = any(c in chunk_class_ids for c in dependent_class_ids)
    if has_dependent:
        if ROOF_ID in chunk_class_ids:
            roof_features_chunk = chunk_gdf[chunk_gdf["class_id"] == ROOF_ID]
        else:
            roof_features_chunk = gpd.GeoDataFrame()
        if ROOF_INSTANCE_CLASS_ID in chunk_class_ids:
            roof_instance_chunk = chunk_gdf[chunk_gdf["class_id"] == ROOF_INSTANCE_CLASS_ID]
        else:
            roof_instance_chunk = gpd.GeoDataFrame()
        non_roof_chunk = chunk_gdf[chunk_gdf["class_id"] != ROOF_ID]

        # Pre-compute IoU-based Roof → Building(New) linkage for RSI BL fallback.
        # Roof's API parent_id points to Building(Deprecated) which is filtered out;
        # the correct path to BL is Roof →(IoU)→ Building(New) →(parent_id)→ BL.
        # Results are reused in _compute_feature_class_data for Building(New) class output.
        roofs_linked_chunk = None
        buildings_linked_chunk = None
        if len(roof_features_chunk) > 0 and BUILDING_NEW_ID in chunk_class_ids:
            building_new_chunk = chunk_gdf[chunk_gdf["class_id"] == BUILDING_NEW_ID]
            if len(building_new_chunk) > 0:
                roofs_linked_chunk, buildings_linked_chunk = link_roofs_to_buildings(
                    roof_features_chunk, building_new_chunk
                )
                roof_to_building_lookup = {
                    fid: bid
                    for fid, bid in zip(
                        roofs_linked_chunk["feature_id"].values,
                        roofs_linked_chunk["parent_building_id"].values,
                    )
                    if pd.notna(bid)
                }

        # Compute roof attrs cache
        if ROOF_ID in whitelisted_classes and roof_features_chunk is not None and len(roof_features_chunk) > 0:
            child_by_aoi = _group_children_by_aoi(non_roof_chunk, None)
            roof_geoms_proj, child_proj_by_aoi = _batch_project_geometries(
                roof_features_chunk,
                child_by_aoi,
                country,
            )
            roof_attrs_cache_chunk = {}
            roof_records = _dataframe_to_records_with_index(roof_features_chunk)
            for ridx, row in enumerate(roof_records):
                try:
                    roof_aoi = row.get(AOI_ID_COLUMN_NAME)
                    aoi_children = child_by_aoi.get(roof_aoi) if roof_aoi is not None else None
                    attrs = flatten_roof_attributes(
                        [row],
                        country=country,
                        child_features=aoi_children,
                        parent_projected=roof_geoms_proj.iloc[ridx],
                        children_projected=child_proj_by_aoi.get(roof_aoi),
                    )
                    roof_attrs_cache_chunk[row["feature_id"]] = attrs
                except Exception as e:
                    _log.debug(f"Could not flatten roof attrs: {e}")
            del child_by_aoi, child_proj_by_aoi, roof_geoms_proj

    # Compute per-class data for each whitelisted class
    results = {}
    for cid in whitelisted_classes:
        if cid not in chunk_class_ids:
            continue
        class_feats = chunk_gdf[chunk_gdf["class_id"] == cid]
        if len(class_feats) == 0:
            continue

        desc = chunk_class_descriptions.get(cid, FEATURE_CLASS_DESCRIPTIONS.get(cid, f"class_{cid[:8]}"))
        try:
            flat_df, geo_gdf = _compute_feature_class_data(
                class_id=cid,
                class_description=desc,
                country=country,
                aoi_columns=aoi_input_columns,
                class_features=class_feats,
                features_gdf=cross_class_gdf,
                roof_features=roof_features_chunk,
                non_roof_features=non_roof_chunk,
                roof_instance_features=roof_instance_chunk,
                roof_attrs_cache=roof_attrs_cache_chunk,
                parent_lookup=shared_parent_lookup,
                roof_to_building_lookup=roof_to_building_lookup,
                roofs_linked=roofs_linked_chunk,
                buildings_linked=buildings_linked_chunk,
            )
        except Exception as e:
            _log.error(f"Per-class export failed for {desc}: {e}")
            continue
        if flat_df is None:
            continue

        entry = {}
        entry["tabular"] = pa.Table.from_pandas(flat_df, preserve_index=False)
        if geo_gdf is not None and len(geo_gdf) > 0:
            geo_wkb_df = pd.DataFrame(geo_gdf)
            geo_wkb_df["geometry"] = geo_gdf.geometry.to_wkb()
            entry["geo"] = pa.Table.from_pandas(geo_wkb_df, preserve_index=False)
        results[cid] = entry
        del flat_df, geo_gdf

    return results


def _compute_feature_class_data(
    class_id: str,
    class_description: str,
    country: str,
    aoi_columns: list = None,
    class_features: gpd.GeoDataFrame = None,
    features_gdf: gpd.GeoDataFrame = None,
    roof_features: gpd.GeoDataFrame = None,
    non_roof_features: gpd.GeoDataFrame = None,
    roof_instance_features: gpd.GeoDataFrame = None,
    roof_attrs_cache: dict = None,
    parent_lookup: dict = None,
    roof_to_building_lookup: dict = None,
    roofs_linked: gpd.GeoDataFrame = None,
    buildings_linked: gpd.GeoDataFrame = None,
) -> tuple:
    """
    Compute the flat attribute DataFrame and GeoDataFrame for a single feature class.

    This is the pure-compute core shared by export_feature_class() (which writes files)
    and the per-chunk streaming export path (which writes to ParquetWriters).

    Args:
        class_id: UUID of the feature class to export
        class_description: Human-readable description
        country: Country code for units (us, au, etc.)
        aoi_columns: Additional columns from the AOI input file to include
        class_features: Pre-filtered features for this class
        features_gdf: GeoDataFrame with all features (fallback for cross-class lookups)
        roof_features: Pre-filtered roof features
        non_roof_features: Pre-filtered non-roof features
        roof_instance_features: Pre-filtered roof instance features
        roof_attrs_cache: Pre-computed dict mapping roof feature_id to flattened attribute dicts

    Returns:
        Tuple of (flat_df, geo_gdf) where flat_df is the tabular DataFrame and
        geo_gdf is the GeoDataFrame with geometry. Returns (None, None) if no features.
    """
    # Use pre-filtered class features if provided, otherwise filter from features_gdf
    if class_features is None:
        class_features = features_gdf[features_gdf["class_id"] == class_id].copy()
    else:
        class_features = class_features.copy()
    if len(class_features) == 0:
        return (None, None)

    # Use pre-built parent lookup if provided (avoids rebuilding per class)
    p_lookup = (
        parent_lookup
        if parent_lookup is not None
        else (build_parent_lookup(features_gdf) if features_gdf is not None and len(features_gdf) > 0 else {})
    )
    _roof_to_building = roof_to_building_lookup or {}

    # Build output DataFrame using vectorized operations (much faster than iterrows)
    # Accumulate DataFrames in a list and concat once at the end to avoid fragmentation
    n_rows = len(class_features)
    df_parts = []
    added_cols = set()  # Track columns to avoid duplicates (e.g., area_sqm from Feature API vs Roof Age API)

    # --- Section A: Initial columns (aoi_id, metadata, standard, confidence, area, dates) ---
    initial_batch = {}

    # Add aoi_id from index or column
    if class_features.index.name == AOI_ID_COLUMN_NAME:
        initial_batch[AOI_ID_COLUMN_NAME] = class_features.index.values
        added_cols.add(AOI_ID_COLUMN_NAME)
    elif AOI_ID_COLUMN_NAME in class_features.columns:
        initial_batch[AOI_ID_COLUMN_NAME] = class_features[AOI_ID_COLUMN_NAME].values
        added_cols.add(AOI_ID_COLUMN_NAME)

    # Add metadata columns (address fields, coordinates, etc.) if present
    # These come from the merged features parquet which includes AOI metadata
    metadata_cols = list(ADDRESS_FIELDS) + [
        "lat",
        "lon",
        "latitude",
        "longitude",  # Coordinates
        "match_quality",
        "matchQuality",  # Geocoding quality
        "geocode_source",
        "geocodeSource",  # Geocoding source
    ]
    # Add any additional AOI columns from the input file (e.g., "Property Id")
    if aoi_columns:
        metadata_cols = metadata_cols + [c for c in aoi_columns if c not in metadata_cols]
    for col in metadata_cols:
        if col in class_features.columns and col not in added_cols:
            initial_batch[col] = class_features[col].values
            added_cols.add(col)

    # Add standard columns
    if "feature_id" in class_features.columns:
        initial_batch["feature_id"] = class_features["feature_id"].values
        added_cols.add("feature_id")
    initial_batch["class_id"] = class_id
    added_cols.add("class_id")
    initial_batch["description"] = class_description
    added_cols.add("description")

    # Add is_primary column if present (only for classes with primary selection)
    if "is_primary" in class_features.columns and "is_primary" not in added_cols:
        initial_batch["is_primary"] = class_features["is_primary"].values
        added_cols.add("is_primary")

    # Roof instances don't have confidence/fidelity (they have trust_score instead)
    # Only add confidence and fidelity for non-roof-instance classes
    if class_id != ROOF_INSTANCE_CLASS_ID:
        if "confidence" in class_features.columns:
            initial_batch["confidence"] = class_features["confidence"].values
            added_cols.add("confidence")
        if "fidelity" in class_features.columns:
            initial_batch["fidelity"] = class_features["fidelity"].values
            added_cols.add("fidelity")

    # Add area fields (vectorized) - skip if already added
    # Roof instances only have area_sqm (no clipped/unclipped distinction)
    if class_id == ROOF_INSTANCE_CLASS_ID:
        area_cols = ["area_sqm", "area_sqft"]
    else:
        area_cols = [
            "area_sqm",
            "clipped_area_sqm",
            "unclipped_area_sqm",
            "area_sqft",
            "clipped_area_sqft",
            "unclipped_area_sqft",
        ]
    for col in area_cols:
        if col in class_features.columns and col not in added_cols:
            initial_batch[col] = class_features[col].values
            added_cols.add(col)

    # Add date fields (vectorized) - not applicable to roof instances
    if class_id != ROOF_INSTANCE_CLASS_ID:
        for col in ["survey_date", "mesh_date"]:
            if col in class_features.columns and col not in added_cols:
                initial_batch[col] = class_features[col].values
                added_cols.add(col)

    convert_bool_columns_to_yn(initial_batch)

    if initial_batch:
        df_parts.append(pd.DataFrame(initial_batch, index=range(n_rows)))

    # --- Section B: Roof instance attributes ---
    if class_id == ROOF_INSTANCE_CLASS_ID:
        ri_batch = {}
        # Add roof instance linkage columns (parent_id = parent roof, parent_iou = IoU with parent)
        for col in ["parent_id", "parent_iou"]:
            if col in class_features.columns and col not in added_cols:
                ri_batch[col] = class_features[col].values
                added_cols.add(col)

        # Add roof_age_ prefix to Roof Age API columns (whitelist).
        # Only known Roof Age columns get prefixed; all other columns are ignored.
        for col in class_features.columns:
            if col in added_cols:
                continue
            if col in ROOF_AGE_PREFIX_COLUMNS:
                dst = f"roof_age_{col}"
            elif col.startswith("roof_age_"):
                dst = col  # calculated fields like roof_age_years_as_of_date
            else:
                continue
            if dst not in added_cols:
                ri_batch[dst] = class_features[col].values
                added_cols.add(dst)

        convert_bool_columns_to_yn(ri_batch)

        if ri_batch:
            df_parts.append(pd.DataFrame(ri_batch, index=range(n_rows)))

    # --- Section C: Roof linkage columns (linking to roof instances) ---
    if class_id == ROOF_ID:
        roof_linkage_batch = {}
        for col in [
            "primary_child_roof_age_feature_id",
            "primary_child_roof_age_iou",
            "child_roof_instances",
            "child_roof_instance_count",
        ]:
            if col in class_features.columns and col not in added_cols:
                roof_linkage_batch[col] = class_features[col].values
                added_cols.add(col)

        if roof_linkage_batch:
            df_parts.append(pd.DataFrame(roof_linkage_batch, index=range(n_rows)))

        # Add flattened attributes of the primary child roof instance
        # Look up roof instances from the full features_gdf and join on primary_child_roof_age_feature_id
        if "primary_child_roof_age_feature_id" in class_features.columns:
            roof_instances = (
                roof_instance_features
                if roof_instance_features is not None
                else features_gdf[features_gdf["class_id"] == ROOF_INSTANCE_CLASS_ID]
            )
            if len(roof_instances) > 0 and "feature_id" in roof_instances.columns:
                # Build lookup table with primary_child_roof_age_ prefixed column names
                ri_cols = ["feature_id"]
                col_rename = {}
                for col in roof_instances.columns:
                    if col == "feature_id":
                        continue
                    if col in ROOF_AGE_PREFIX_COLUMNS:
                        base = f"roof_age_{col}"
                    elif col.startswith("roof_age_"):
                        base = col
                    else:
                        continue
                    prefixed_dst = f"primary_child_{base}"
                    if prefixed_dst not in added_cols:
                        ri_cols.append(col)
                        col_rename[col] = prefixed_dst

                if len(ri_cols) > 1:  # More than just feature_id
                    ri_lookup = roof_instances[ri_cols].drop_duplicates(subset=["feature_id"])
                    ri_lookup = ri_lookup.rename(columns=col_rename).set_index("feature_id")

                    # Map attributes using vectorized lookup
                    primary_child_ids = class_features["primary_child_roof_age_feature_id"]
                    ri_mapped_batch = {}
                    for col in ri_lookup.columns:
                        ri_mapped_batch[col] = primary_child_ids.map(ri_lookup[col]).values
                        added_cols.add(col)
                    convert_bool_columns_to_yn(ri_mapped_batch)
                    if ri_mapped_batch:
                        df_parts.append(pd.DataFrame(ri_mapped_batch, index=range(n_rows)))

        # Flatten roof attributes (RSI, hurricane, defensible space, materials, 3D)
        # These are from include parameters and the roof's own attributes array
        try:
            if roof_attrs_cache is not None:
                # Use pre-computed cache (avoids duplicate flatten loop)
                attr_records = [roof_attrs_cache.get(fid, {}) for fid in class_features["feature_id"].values]
                logger.debug(f"Roof attribute flattening: used cache for {len(class_features)} roofs")
            else:
                # Fallback: compute per-row (for standalone calls without cache)
                child_by_aoi = _group_children_by_aoi(non_roof_features, features_gdf)
                roof_geoms_projected, child_proj_by_aoi = _batch_project_geometries(
                    class_features, child_by_aoi, country
                )

                t_roof_flatten = time.monotonic()
                attr_records = []
                roof_records = _dataframe_to_records_with_index(class_features)

                for idx, row in enumerate(roof_records):
                    try:
                        roof_aoi = row.get(AOI_ID_COLUMN_NAME)
                        aoi_children = child_by_aoi.get(roof_aoi) if roof_aoi is not None else None
                        attrs = flatten_roof_attributes(
                            [row],
                            country=country,
                            child_features=aoi_children,
                            parent_projected=roof_geoms_projected.iloc[idx],
                            children_projected=child_proj_by_aoi.get(roof_aoi),
                        )
                        attr_records.append(attrs)
                    except Exception as e:
                        logger.debug(f"Could not flatten roof attributes for feature: {e}")
                        attr_records.append({})
                logger.debug(
                    f"Roof attribute flattening: {time.monotonic() - t_roof_flatten:.1f}s for {len(class_features)} roofs"
                )

            if attr_records:
                attr_df = pd.DataFrame(attr_records)
                # Remove columns that would be duplicates
                attr_df = attr_df.drop(
                    columns=[c for c in attr_df.columns if c in added_cols],
                    errors="ignore",
                )
                if len(attr_df.columns) > 0:
                    df_parts.append(attr_df.reset_index(drop=True))
                    added_cols.update(attr_df.columns)
        except Exception as e:
            logger.debug(f"Could not flatten roof attributes: {e}")

        # Resolve best RSI per roof.
        # For each roof: check roof's own RSI first, then traverse parent chain to BL.
        # Overwrites roof_spotlight_index/confidence with the resolved value.
        try:
            bl_in_features = (
                BUILDING_LIFECYCLE_ID in features_gdf["class_id"].values if features_gdf is not None else False
            )
            if bl_in_features and "roof_spotlight_index" in class_features.columns:
                roof_records_for_fp = _dataframe_to_records_with_index(class_features)

                def _resolve_roof_rsi(i):
                    row = roof_records_for_fp[i]
                    rsi = extract_rsi_from_feature(row)
                    if not rsi:
                        bn_id = _roof_to_building.get(row.get("feature_id"))
                        if bn_id:
                            bn_row = p_lookup.get(bn_id)
                            if bn_row is not None:
                                rsi = resolve_footprint_rsi(bn_row, parent_lookup=p_lookup)
                    return rsi

                rsi_result = _resolve_rsi_batch(n_rows, _resolve_roof_rsi)
                if rsi_result is not None:
                    rsi_vals, rsi_conf, rsi_mver = rsi_result
                    # Overwrite roof_spotlight_index in existing df_parts
                    for part in df_parts:
                        for col, arr in zip(_RSI_COLUMNS, (rsi_vals, rsi_conf, rsi_mver)):
                            if col in part.columns:
                                part[col] = arr
        except Exception:
            logger.error("Could not resolve footprint RSI for roofs", exc_info=True)

    # --- Section D: Building attributes (3D, pitch, ground height, then link to child roofs and add RSI) ---
    if class_id == BUILDING_NEW_ID:
        # Flatten building's own 3D attributes (height, pitch, ground_height, numStories)
        # Uses dot-notation columns created by process_chunk's _flatten_attribute_list,
        # avoiding per-row JSON parsing of the 'attributes' column.
        try:
            bldg_batch = {}
            h3d_col = "Building 3d attributes.has3dAttributes"
            if h3d_col in class_features.columns:
                has_3d = class_features[h3d_col].infer_objects(copy=False).fillna(False)
                bldg_batch["has_3d_attributes"] = has_3d.map({True: TRUE_STRING, False: FALSE_STRING}).fillna(
                    FALSE_STRING
                )

                height_col = "Building 3d attributes.height"
                if height_col in class_features.columns:
                    height_vals = class_features[height_col].where(has_3d)
                    if country in IMPERIAL_COUNTRIES:
                        bldg_batch["height_ft"] = (height_vals * METERS_TO_FEET).round(1)
                    else:
                        bldg_batch["height_m"] = height_vals.round(1)

                # Discover numStories columns dynamically (e.g. numStories.1, numStories.2, ...)
                num_stories_prefix = "Building 3d attributes.numStories."
                for col in class_features.columns:
                    if isinstance(col, str) and col.startswith(num_stories_prefix):
                        k = col[len(num_stories_prefix) :]
                        out_col = f"num_storeys_{k}_confidence"
                        if out_col not in added_cols:
                            bldg_batch[out_col] = class_features[col].where(has_3d)

                fidelity_col = "Building 3d attributes.fidelity"
                if fidelity_col in class_features.columns and "fidelity" not in added_cols:
                    bldg_batch["fidelity"] = class_features[fidelity_col].where(has_3d)

            pitch_col = "Building pitch.value"
            if pitch_col in class_features.columns:
                bldg_batch["pitch_degrees"] = class_features[pitch_col].round(2)

            ground_height_col = "Ground height.value"
            if ground_height_col in class_features.columns:
                if country in IMPERIAL_COUNTRIES:
                    bldg_batch["ground_height_ft"] = (class_features[ground_height_col] * METERS_TO_FEET).round(1)
                else:
                    bldg_batch["ground_height_m"] = class_features[ground_height_col].round(1)

            # Remove any columns that would be duplicates
            bldg_batch = {k: v for k, v in bldg_batch.items() if k not in added_cols}
            if bldg_batch:
                df_parts.append(pd.DataFrame(bldg_batch, index=range(n_rows)))
                added_cols.update(bldg_batch.keys())
        except Exception as e:
            logger.debug(f"Could not flatten building attributes: {e}")

        # Use pre-computed roof-to-building linkage if available, otherwise compute now.
        if roofs_linked is None or buildings_linked is None:
            roofs = (
                roof_features.copy()
                if roof_features is not None
                else features_gdf[features_gdf["class_id"] == ROOF_ID].copy()
            )
            if len(roofs) > 0:
                roofs_linked, buildings_linked = link_roofs_to_buildings(roofs, class_features)

        if roofs_linked is not None and buildings_linked is not None and len(roofs_linked) > 0:
            try:
                # Add linkage columns
                bldg_linkage_batch = {}
                for col in [
                    "primary_child_roof_id",
                    "primary_child_roof_iou",
                    "child_roofs",
                    "child_roof_count",
                ]:
                    if col in buildings_linked.columns and col not in added_cols:
                        bldg_linkage_batch[col] = buildings_linked[col].values
                        added_cols.add(col)
                if bldg_linkage_batch:
                    df_parts.append(pd.DataFrame(bldg_linkage_batch, index=range(n_rows)))

                # Build a mapping from roof feature_id to flattened attributes.
                if roof_attrs_cache is not None:
                    # Use pre-computed cache (avoids duplicate flatten loop)
                    roof_attrs = {fid: roof_attrs_cache.get(fid, {}) for fid in roofs_linked["feature_id"].values}
                    logger.debug(f"Building roof attribute flattening: used cache for {len(roofs_linked)} roofs")
                else:
                    # Fallback: compute per-row (for standalone calls without cache)
                    child_by_aoi_bldg = _group_children_by_aoi(non_roof_features, features_gdf)
                    bldg_roof_geoms_projected, bldg_child_proj_by_aoi = _batch_project_geometries(
                        roofs_linked, child_by_aoi_bldg, country
                    )

                    t_bldg_roof_flatten = time.monotonic()
                    roof_attrs = {}
                    roof_records = _dataframe_to_records_with_index(roofs_linked)

                    for idx, row in enumerate(roof_records):
                        try:
                            roof_aoi = row.get(AOI_ID_COLUMN_NAME)
                            aoi_children = child_by_aoi_bldg.get(roof_aoi) if roof_aoi is not None else None
                            attrs = flatten_roof_attributes(
                                [row],
                                country=country,
                                child_features=aoi_children,
                                parent_projected=bldg_roof_geoms_projected.iloc[idx],
                                children_projected=bldg_child_proj_by_aoi.get(roof_aoi),
                            )
                            roof_attrs[row["feature_id"]] = attrs
                        except Exception as e:
                            logger.debug(f"Could not flatten building-linked roof attributes: {e}")
                    logger.debug(
                        f"Building roof attribute flattening: {time.monotonic() - t_bldg_roof_flatten:.1f}s for {len(roofs_linked)} roofs"
                    )

                if roof_attrs:
                    # Map primary child roof attributes to buildings via DataFrame reindex
                    # (replaces per-attribute .apply(lambda) with a single vectorized lookup)
                    roof_attrs_df = pd.DataFrame.from_dict(roof_attrs, orient="index")
                    roof_attrs_df.columns = [f"primary_child_roof_{c}" for c in roof_attrs_df.columns]
                    primary_ids = buildings_linked["primary_child_roof_id"].values
                    mapped = roof_attrs_df.reindex(primary_ids).reset_index(drop=True)

                    roof_attr_batch = {}
                    _DOM = (
                        "primary_child_roof_dominant_roof_material_",
                        "primary_child_roof_dominant_roof_types_",
                    )
                    dominant_cols = sorted(c for c in mapped.columns if c.startswith(_DOM))
                    other_cols = sorted(c for c in mapped.columns if not c.startswith(_DOM) and c not in _EXCLUDE_RSI)
                    for col in dominant_cols + other_cols:
                        if col not in added_cols:
                            roof_attr_batch[col] = mapped[col].values
                            added_cols.add(col)

                    # Add min/max resolved RSI across all child roofs.
                    # For each child roof, resolve best RSI (roof's own first,
                    # then BL fallback) so min/max reflect the same logic as
                    # the per-row roof_spotlight_index column.
                    rsi_lookup = {}
                    for fid, attrs in roof_attrs.items():
                        if "roof_spotlight_index" in attrs:
                            rsi_lookup[fid] = attrs["roof_spotlight_index"]
                        else:
                            # Try BL fallback via IoU-linked Building(New) → BL
                            bn_id = _roof_to_building.get(fid)
                            if bn_id:
                                bn_row = p_lookup.get(bn_id)
                                if bn_row is not None:
                                    resolved_rsi = resolve_footprint_rsi(bn_row, parent_lookup=p_lookup)
                                    if resolved_rsi and "roof_spotlight_index" in resolved_rsi:
                                        rsi_lookup[fid] = resolved_rsi["roof_spotlight_index"]

                    if rsi_lookup:
                        n_buildings = len(buildings_linked)
                        min_vals = np.full(n_buildings, np.nan)
                        max_vals = np.full(n_buildings, np.nan)
                        for i, child_json in enumerate(buildings_linked["child_roofs"].values):
                            if not child_json or child_json == "[]":
                                continue
                            try:
                                child_list = json.loads(child_json) if isinstance(child_json, str) else child_json
                                rsi_vals = [
                                    rsi_lookup[c["feature_id"]] for c in child_list if c.get("feature_id") in rsi_lookup
                                ]
                                if rsi_vals:
                                    min_vals[i] = min(rsi_vals)
                                    max_vals[i] = max(rsi_vals)
                            except Exception:
                                continue

                        if "roof_spotlight_index_min" not in added_cols:
                            roof_attr_batch["roof_spotlight_index_min"] = np.where(np.isnan(min_vals), None, min_vals)
                            added_cols.add("roof_spotlight_index_min")

                        if "roof_spotlight_index_max" not in added_cols:
                            roof_attr_batch["roof_spotlight_index_max"] = np.where(np.isnan(max_vals), None, max_vals)
                            added_cols.add("roof_spotlight_index_max")

                    if roof_attr_batch:
                        df_parts.append(pd.DataFrame(roof_attr_batch, index=range(n_rows)))

                # Resolve best RSI per building.
                # For each building: resolve via primary child roof (roof's own RSI first,
                # then BL fallback via Building(New) → BL parent_id), same logic as per-roof.
                if roof_attrs:
                    pcr_ids = buildings_linked["primary_child_roof_id"].values
                    bl_records = _dataframe_to_records_with_index(buildings_linked)

                    def _resolve_bn_rsi(i):
                        pcr_id = pcr_ids[i]
                        roof_row = p_lookup.get(pcr_id) if pd.notna(pcr_id) else None
                        if roof_row is not None:
                            rsi = extract_rsi_from_feature(roof_row)
                            if rsi:
                                return rsi
                        # Roof lacks RSI or no child roof — traverse BN → BL
                        return resolve_footprint_rsi(bl_records[i], parent_lookup=p_lookup)

                    rsi_result = _resolve_rsi_batch(n_rows, _resolve_bn_rsi)
                    if rsi_result is not None:
                        rsi_vals, rsi_conf, rsi_mver = rsi_result
                        rsi_batch = {}
                        for col, arr in zip(_RSI_COLUMNS, (rsi_vals, rsi_conf, rsi_mver)):
                            if col not in added_cols:
                                rsi_batch[col] = arr
                                added_cols.add(col)
                        if rsi_batch:
                            df_parts.append(pd.DataFrame(rsi_batch, index=range(n_rows)))

                # Link roof age data from primary child roof's roof instance
                # Chain: building → primary_child_roof → primary_child_roof_age (roof instance) → roof_age_*
                if "primary_child_roof_age_feature_id" in roofs_linked.columns:
                    roof_instances = (
                        roof_instance_features.copy()
                        if roof_instance_features is not None
                        else features_gdf[features_gdf["class_id"] == ROOF_INSTANCE_CLASS_ID].copy()
                    )
                    if len(roof_instances) > 0:
                        # Create lookup from roof feature_id → roof's primary_child_roof_age_feature_id
                        roof_to_ri = roofs_linked.set_index("feature_id")["primary_child_roof_age_feature_id"].to_dict()

                        # Roof age columns to link through (same as what roofs get)
                        ri_cols = ["feature_id"]
                        col_rename = {}
                        for col in roof_instances.columns:
                            if col == "feature_id":
                                continue
                            if col in ROOF_AGE_PREFIX_COLUMNS:
                                base = f"roof_age_{col}"
                            elif col.startswith("roof_age_"):
                                base = col
                            else:
                                continue
                            prefixed_dst = f"primary_child_{base}"
                            ri_cols.append(col)
                            col_rename[col] = prefixed_dst

                        if len(ri_cols) > 1:  # More than just feature_id
                            ri_lookup = roof_instances[ri_cols].drop_duplicates(subset=["feature_id"])
                            ri_lookup = ri_lookup.rename(columns=col_rename).set_index("feature_id")

                            # Map: building.primary_child_roof_id → roof.primary_child_roof_age_feature_id → ri attributes
                            primary_roof_ids = buildings_linked["primary_child_roof_id"]
                            primary_ri_ids = primary_roof_ids.map(roof_to_ri)

                            bldg_ri_batch = {}
                            for col in ri_lookup.columns:
                                if col not in added_cols:
                                    bldg_ri_batch[col] = primary_ri_ids.map(ri_lookup[col]).values
                                    added_cols.add(col)
                            convert_bool_columns_to_yn(bldg_ri_batch)
                            if bldg_ri_batch:
                                df_parts.append(pd.DataFrame(bldg_ri_batch, index=range(n_rows)))

            except Exception as e:
                logger.debug(f"Could not link roofs to buildings: {e}")

    # --- Section E: Building lifecycle damage ---
    if class_id == BUILDING_LIFECYCLE_ID:
        if "damage" in class_features.columns:
            # Parse JSON and flatten damage columns for class-specific export
            damage_data = class_features["damage"].apply(
                lambda x: _flatten_damage(json.loads(x) if isinstance(x, str) else x)
            )
            flat_damage_df = pd.DataFrame(damage_data.tolist(), index=class_features.index)
            damage_batch = {}
            for col in flat_damage_df.columns:
                if col not in added_cols:
                    damage_batch[col] = flat_damage_df[col].values
                    added_cols.add(col)
            if damage_batch:
                df_parts.append(pd.DataFrame(damage_batch, index=range(n_rows)))

        # Link child roofs to building lifecycles via IoU-based Roof→BN→BL chain.
        # Chain: Roof →(IoU)→ Building(New) →(parent_id)→ BL.
        # For each BL, find the largest roof whose IoU-linked BN leads to it.
        try:
            roofs = (
                roof_features.copy()
                if roof_features is not None
                else features_gdf[features_gdf["class_id"] == ROOF_ID].copy()
            )
            if len(roofs) > 0:
                # Map each roof to its ancestor BL via IoU-linked Building(New)
                roof_to_bl = {}
                for roof_fid, bn_id in _roof_to_building.items():
                    if bn_id:
                        bn_row = p_lookup.get(bn_id)
                        if bn_row is not None:
                            bl_fid = (
                                bn_row.get("parent_id")
                                if hasattr(bn_row, "get")
                                else getattr(bn_row, "parent_id", None)
                            )
                            if bl_fid:
                                roof_to_bl[roof_fid] = bl_fid

                # Group roofs by their BL and pick the largest as primary
                bl_to_roofs = {}
                for roof_fid, bl_fid in roof_to_bl.items():
                    bl_to_roofs.setdefault(bl_fid, []).append(roof_fid)

                bl_primary_roof = {}
                roof_area_col = (
                    "clipped_area_sqm"
                    if "clipped_area_sqm" in roofs.columns
                    else "unclipped_area_sqm" if "unclipped_area_sqm" in roofs.columns else None
                )
                roof_fid_to_idx = {fid: idx for idx, fid in enumerate(roofs["feature_id"].values)}
                for bl_fid, roof_fids in bl_to_roofs.items():
                    if roof_area_col and len(roof_fids) > 1:
                        best = max(
                            roof_fids,
                            key=lambda f: (
                                roofs.iloc[roof_fid_to_idx[f]][roof_area_col]
                                if pd.notna(roofs.iloc[roof_fid_to_idx[f]][roof_area_col])
                                else 0
                            ),
                        )
                    else:
                        best = roof_fids[0]
                    bl_primary_roof[bl_fid] = best

                # Build linkage columns
                bl_linkage_batch = {}
                primary_ids = [bl_primary_roof.get(fid) for fid in class_features["feature_id"].values]
                child_counts = [len(bl_to_roofs.get(fid, [])) for fid in class_features["feature_id"].values]
                if "primary_child_roof_id" not in added_cols:
                    bl_linkage_batch["primary_child_roof_id"] = primary_ids
                    added_cols.add("primary_child_roof_id")
                if "child_roof_count" not in added_cols:
                    bl_linkage_batch["child_roof_count"] = child_counts
                    added_cols.add("child_roof_count")
                if bl_linkage_batch:
                    df_parts.append(pd.DataFrame(bl_linkage_batch, index=range(n_rows)))

                # Map primary child roof attributes to BL features
                roof_attrs = (
                    {fid: roof_attrs_cache.get(fid, {}) for fid in roofs["feature_id"].values}
                    if roof_attrs_cache is not None
                    else {}
                )

                if roof_attrs:
                    roof_attrs_df = pd.DataFrame.from_dict(roof_attrs, orient="index")
                    roof_attrs_df.columns = [f"primary_child_roof_{c}" for c in roof_attrs_df.columns]
                    mapped = roof_attrs_df.reindex(primary_ids).reset_index(drop=True)

                    roof_attr_batch = {}
                    for col in sorted(mapped.columns):
                        if col not in added_cols and col not in _EXCLUDE_RSI:
                            roof_attr_batch[col] = mapped[col].values
                            added_cols.add(col)
                    if roof_attr_batch:
                        df_parts.append(pd.DataFrame(roof_attr_batch, index=range(n_rows)))

                    # Resolve footprint RSI: primary child roof first (IoU-linked via BN),
                    # then fall back to BL's own RSI.
                    bl_records = _dataframe_to_records_with_index(class_features)

                    def _resolve_bl_rsi(i):
                        pcr_id = primary_ids[i]
                        roof_row = p_lookup.get(pcr_id) if pcr_id is not None else None
                        if roof_row is not None:
                            rsi = extract_rsi_from_feature(roof_row)
                            if not rsi:
                                bn_id = _roof_to_building.get(pcr_id)
                                if bn_id:
                                    bn_row = p_lookup.get(bn_id)
                                    if bn_row is not None:
                                        rsi = resolve_footprint_rsi(bn_row, parent_lookup=p_lookup)
                            return rsi
                        # No child roof — use BL's own RSI
                        return extract_rsi_from_feature(bl_records[i])

                    rsi_result = _resolve_rsi_batch(n_rows, _resolve_bl_rsi)
                    if rsi_result is not None:
                        rsi_vals, rsi_conf, rsi_mver = rsi_result
                        rsi_batch = {}
                        for col, arr in zip(_RSI_COLUMNS, (rsi_vals, rsi_conf, rsi_mver)):
                            if col not in added_cols:
                                rsi_batch[col] = arr
                                added_cols.add(col)
                        if rsi_batch:
                            df_parts.append(pd.DataFrame(rsi_batch, index=range(n_rows)))
        except Exception:
            logger.error("Could not link roofs to building lifecycles", exc_info=True)

    # --- Section F: Mapbrowser link ---
    # Uses geometry centroid for location and survey_date/installation_date for date
    if "geometry" in class_features.columns:
        # Vectorized centroid extraction (geographic CRS is fine for URL link coordinates)
        geom_valid = class_features.geometry.notna() & ~class_features.geometry.is_empty
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*geographic CRS.*centroid.*")
            centroids = class_features.geometry.centroid
        lats = centroids.y.astype(str)
        lons = centroids.x.astype(str)

        # Determine date column based on class type
        if class_id == ROOF_INSTANCE_CLASS_ID:
            if "installation_date" in class_features.columns:
                date_col = "installation_date"
            else:
                date_col = None
        else:
            date_col = "survey_date" if "survey_date" in class_features.columns else None

        # Vectorized date formatting
        if date_col is not None and date_col in class_features.columns:
            date_valid = class_features[date_col].notna()
            dates = class_features[date_col].astype(str).str.replace("-", "", regex=False).str[:8]
        else:
            date_valid = pd.Series(False, index=class_features.index)
            dates = pd.Series("", index=class_features.index)

        # Build links vectorized
        has_date = date_valid & (dates != "") & (dates != "None") & (dates != "NaT")
        links = pd.Series(None, index=class_features.index, dtype=object)
        if has_date.any():
            links[has_date & geom_valid] = (
                "https://apps.nearmap.com/maps/#/@"
                + lats[has_date & geom_valid]
                + ","
                + lons[has_date & geom_valid]
                + ",21.00z,0d/V/"
                + dates[has_date & geom_valid]
                + "?locationMarker"
            )
        no_date = ~has_date & geom_valid
        if no_date.any():
            links[no_date] = (
                "https://apps.nearmap.com/maps/#/@" + lats[no_date] + "," + lons[no_date] + ",21.00z,0d?locationMarker"
            )

        df_parts.append(
            pd.DataFrame(
                {"link": links.values},
                index=range(n_rows),
            )
        )
        added_cols.add("link")

    # --- Final assembly: single concat to avoid DataFrame fragmentation ---
    if df_parts:
        flat_df = pd.concat(
            [df.reset_index(drop=True) for df in df_parts],
            axis=1,
        )
    else:
        flat_df = pd.DataFrame()

    # Build GeoDataFrame with geometry
    geo_gdf = None
    if "geometry" in class_features.columns:
        geo_gdf = gpd.GeoDataFrame(flat_df.copy(), geometry=class_features.geometry.values, crs=API_CRS)

    return (flat_df, geo_gdf)


def export_feature_class(
    features_gdf: gpd.GeoDataFrame,
    class_id: str,
    class_description: str,
    country: str,
    output_dir: str,
    aoi_columns: list = None,
    tabular_file_format: str = "csv",
    export_geo_parquet: bool = True,
    class_features: gpd.GeoDataFrame = None,
    roof_features: gpd.GeoDataFrame = None,
    non_roof_features: gpd.GeoDataFrame = None,
    roof_instance_features: gpd.GeoDataFrame = None,
    roof_attrs_cache: dict = None,
) -> tuple:
    """
    Export features of a single class to tabular file (CSV or Parquet) and/or GeoParquet.

    Args:
        features_gdf: GeoDataFrame with all features (used for cross-class lookups).
                      Can be None when class_features and related pre-filtered DataFrames are provided.
        class_id: UUID of the feature class to export
        class_description: Human-readable description (used in filename)
        country: Country code for units (us, au, etc.)
        output_dir: Directory path for output files
        aoi_columns: Additional columns from the AOI input file to include (e.g., ["Property Id"])
        tabular_file_format: Format for the flat attribute file — "csv" or "parquet". None to skip.
        export_geo_parquet: Whether to export GeoParquet files (with geometry)
        class_features: Pre-filtered features for this class (avoids re-filtering features_gdf)
        roof_features: Pre-filtered roof features (avoids repeated features_gdf scans)
        non_roof_features: Pre-filtered non-roof features (avoids repeated features_gdf scans)
        roof_instance_features: Pre-filtered roof instance features (avoids repeated features_gdf scans)
        roof_attrs_cache: Pre-computed dict mapping roof feature_id to flattened attribute dicts.
                         When provided, skips the per-row flatten_roof_attributes loop for both
                         ROOF_ID and BUILDING_NEW_ID paths (avoids duplicate flattening).

    Returns:
        Tuple of (tabular_path, geo_parquet_path) or (None, None) if no features
    """
    flat_df, geo_gdf = _compute_feature_class_data(
        class_id=class_id,
        class_description=class_description,
        country=country,
        aoi_columns=aoi_columns,
        class_features=class_features,
        features_gdf=features_gdf,
        roof_features=roof_features,
        non_roof_features=non_roof_features,
        roof_instance_features=roof_instance_features,
        roof_attrs_cache=roof_attrs_cache,
    )
    if flat_df is None:
        return (None, None)

    # Normalize class description for filename
    class_name = _description_to_cname(class_description)
    tabular_path = storage.join_path(output_dir, f"{class_name}.{tabular_file_format}") if tabular_file_format else None
    geo_parquet_path = storage.join_path(output_dir, f"{class_name}_features.parquet")

    # Save tabular file (attributes only, no geometry)
    if tabular_file_format == "parquet":
        storage.write_parquet(flat_df, tabular_path, index=False)
    elif tabular_file_format == "csv":
        flat_df.to_csv(tabular_path, index=False)
    else:
        tabular_path = None

    # Save GeoParquet (with geometry)
    if export_geo_parquet and geo_gdf is not None:
        try:
            storage.write_parquet(geo_gdf, geo_parquet_path, index=False, schema_version="1.0.0")
        except (TypeError, ValueError):
            storage.write_parquet(geo_gdf, geo_parquet_path, index=False)
    else:
        geo_parquet_path = None

    return (tabular_path, geo_parquet_path)


def cleanup_process_feature_api():
    """
    Clean up the process-level FeatureApi instance
    """
    global _process_feature_api
    if _process_feature_api is not None:
        try:
            _process_feature_api.cleanup()
        except Exception:
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
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--aoi-file", help="Input AOI file path or S3 URL", type=str, required=True)
    parser.add_argument("--output-dir", help="Directory to store results", type=str, required=True)
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
        help="If set, disable per-feature-class tabular exports (e.g., roof.csv, roof_instance.csv). By default, these are enabled.",
        action="store_true",
    )
    parser.add_argument(
        "--tabular-file-format",
        help="csv | parquet: Format for tabular output files — rollup, buildings, and per-class attribute files (defaults to csv)",
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
                    except Exception:
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
        tabular_file_format="csv",
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
        self.tabular_file_format = tabular_file_format
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

        # Save export configuration at start (before processing begins)
        self._save_config(
            {
                "aoi_file": str(aoi_file),
                "packs": packs,
                "classes": classes,
                "include": include,
                "primary_decision": primary_decision,
                "aoi_grid_min_pct": aoi_grid_min_pct,
                "aoi_grid_inexact": aoi_grid_inexact,
                "aoi_grid_cell_size": aoi_grid_cell_size,
                "processes": processes,
                "threads": threads,
                "chunk_size": chunk_size,
                "include_parcel_geometry": include_parcel_geometry,
                "save_features": save_features,
                "save_buildings": save_buildings,
                "tabular_file_format": tabular_file_format,
                "cache_dir": str(cache_dir) if cache_dir else None,
                "no_cache": no_cache,
                "overwrite_cache": overwrite_cache,
                "compress_cache": compress_cache,
                "country": country,
                "alpha": alpha,
                "beta": beta,
                "prerelease": prerelease,
                "only3d": only3d,
                "since": since,
                "until": until,
                "endpoint": endpoint,
                "url_root": url_root,
                "system_version_prefix": system_version_prefix,
                "system_version": system_version,
                "parcel_mode": parcel_mode,
                "rapid": rapid,
                "order": order,
                "exclude_tiles_with_occlusion": exclude_tiles_with_occlusion,
                "roof_age": self.roof_age,  # Use validated value
                "class_level_files": class_level_files,
                "max_retries": max_retries,
            }
        )

    def api_key(self) -> str:
        # Use provided API key if available, otherwise fall back to environment variable
        if hasattr(self, "api_key_param") and self.api_key_param is not None:
            return self.api_key_param
        return os.getenv("API_KEY")

    def _stream_and_convert_features(
        self,
        feature_paths: list,
        outpath_features: str,
    ) -> Optional[str]:
        """
        Stream feature chunks directly to a geoparquet file.
        This approach avoids loading all chunks into memory simultaneously.

        Per-class export is handled separately by _merge_per_class_chunks()
        which reads pre-computed per-class chunk files written during process_chunk().

        Args:
            feature_paths: List of paths to feature chunk parquet files (strings)
            outpath_features: Output path for final geoparquet file (string, may be S3 URI)

        Returns:
            Local file path to the written geoparquet (for subsequent reads),
            or None if no features were written.  For S3 output, returns the local staging
            path (the file is uploaded but kept locally).
        """

        pqwriter = None
        reference_schema = None  # Store PyArrow schema from first chunk
        schema_promotion_count = 0  # Count chunks needing null-type promotions
        mismatch_count = 0
        variable_col_counts = collections.Counter()  # col_name -> num chunks where it differed

        total = len(feature_paths)
        if total == 0:
            self.logger.warning("No feature data found to write")
            return None

        # Pre-scan all chunk schemas to build union column set and type map.
        # pq.read_schema() reads only parquet footer metadata — no row data.
        # Parallelized for S3 where each read is a network round-trip.
        # Corrupt/unreadable chunks are skipped (logged as errors) to match the
        # resilience of the streaming loop, which also skips bad chunks.
        self.logger.info(f"Scanning schemas from {total} chunk files...")
        scan_workers = S3_PARALLEL_READ_WORKERS if self.is_s3_output else PARALLEL_READ_WORKERS
        chunk_schemas = [None] * total
        with ThreadPoolExecutor(max_workers=scan_workers) as scan_executor:
            futures = {scan_executor.submit(pq.read_schema, feature_paths[i]): i for i in range(total)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    chunk_schemas[idx] = future.result()
                except Exception as e:
                    self.logger.error(f"Failed to read schema from {feature_paths[idx]}: {e}")

        valid_indices = {i for i, s in enumerate(chunk_schemas) if s is not None}
        valid_schemas = [chunk_schemas[i] for i in sorted(valid_indices)]
        if not valid_schemas:
            self.logger.error("All chunk schemas failed to read — cannot proceed")
            return None
        if len(valid_schemas) < total:
            self.logger.warning(f"Skipped {total - len(valid_schemas)}/{total} unreadable chunk files")

        # Build reference_columns: first valid chunk's order as base, extras appended sorted
        first_chunk_columns = list(valid_schemas[0].names)
        all_column_names = set()
        for schema in valid_schemas:
            all_column_names.update(schema.names)
        extra_sorted = sorted(all_column_names - set(first_chunk_columns))
        reference_columns = pd.Index(first_chunk_columns + extra_sorted)

        # Build column_name -> first non-null pyarrow type lookup for null promotion
        column_types = {}
        for schema in valid_schemas:
            for field in schema:
                if field.name not in column_types and field.type != pa.null():
                    column_types[field.name] = field.type

        if extra_sorted:
            self.logger.info(
                f"Union schema has {len(reference_columns)} columns "
                f"({len(extra_sorted)} not present in first chunk)"
            )

        # Set up prefetch buffer: read chunks ahead in background threads
        # while the main thread processes and writes the current chunk.
        # Use higher concurrency for S3 to overlap network round-trips.
        prefetch_workers = S3_PARALLEL_READ_WORKERS if self.is_s3_output else FEATURE_PREFETCH_WORKERS
        executor = ThreadPoolExecutor(max_workers=prefetch_workers)
        prefetch_futures = {}
        initial_submit = min(prefetch_workers, total)
        for idx in range(initial_submit):
            if idx in valid_indices:
                prefetch_futures[idx] = executor.submit(pq.read_table, feature_paths[idx])
        next_submit_idx = initial_submit

        try:
            # Stream chunks directly to geoparquet
            last_resource_update = 0.0
            pbar = tqdm(
                range(total),
                desc="Streaming chunks",
                file=sys.stdout,
                position=0,
                leave=True,
            )
            for i in pbar:
                now = time.time()
                if now - last_resource_update >= 5.0:
                    pbar.set_postfix_str(resource_postfix())
                    last_resource_update = now

                # Skip chunks that failed schema scan
                if i not in valid_indices:
                    if next_submit_idx < total:
                        if next_submit_idx in valid_indices:
                            prefetch_futures[next_submit_idx] = executor.submit(
                                pq.read_table, feature_paths[next_submit_idx]
                            )
                        next_submit_idx += 1
                    continue

                # Get the prefetched result for this index
                future = prefetch_futures.pop(i)
                try:
                    table = future.result()
                except Exception as e:
                    self.logger.error(f"Failed to read {feature_paths[i]}: {e}")
                    if next_submit_idx < total:
                        if next_submit_idx in valid_indices:
                            prefetch_futures[next_submit_idx] = executor.submit(
                                pq.read_table, feature_paths[next_submit_idx]
                            )
                        next_submit_idx += 1
                    continue

                # Submit next prefetch read before processing current chunk
                if next_submit_idx < total:
                    if next_submit_idx in valid_indices:
                        prefetch_futures[next_submit_idx] = executor.submit(
                            pq.read_table, feature_paths[next_submit_idx]
                        )
                    next_submit_idx += 1

                if table.num_rows > 0:
                    # Pad missing columns with nulls and reorder to match union schema
                    missing_cols = set(reference_columns) - set(table.column_names)
                    if missing_cols:
                        mismatch_count += 1
                        for col in missing_cols:
                            variable_col_counts[col] += 1
                            table = table.append_column(col, pa.nulls(table.num_rows))
                    table = table.select(reference_columns)

                    if pqwriter is None:
                        # Promote null/32-bit types using pre-scanned column_types
                        reference_schema = _promote_schema_types(table.schema, column_types=column_types)

                        # Create geoparquet metadata
                        crs_projjson = pyproj.CRS(API_CRS).to_json_dict()
                        geo_metadata = {
                            "version": "1.0.0",
                            "primary_column": "geometry",
                            "columns": {
                                "geometry": {
                                    "encoding": "WKB",
                                    "geometry_types": [],
                                    "crs": crs_projjson,
                                    "edges": "planar",
                                    "orientation": "counterclockwise",
                                }
                            },
                        }

                        schema_with_geo = reference_schema.with_metadata(
                            {b"geo": json.dumps(geo_metadata).encode("utf-8")}
                        )

                        # ParquetWriter doesn't support S3 URIs directly;
                        # write to a local staging file if output is S3.
                        if self.is_s3_output:
                            local_write_path = os.path.join(
                                self._local_final_staging,
                                storage.basename(outpath_features),
                            )
                        else:
                            local_write_path = outpath_features
                        pqwriter = pq.ParquetWriter(local_write_path, schema_with_geo)

                        # Cast first chunk to match promoted schema
                        table = _cast_table_to_schema(table, reference_schema)
                    else:
                        # Reconcile schema differences column-by-column.
                        # Null-type columns (from all-null chunks) are promoted silently.
                        # Only genuine type mismatches that lose data are warned.
                        if table.schema != reference_schema:
                            arrays = []
                            type_warnings = []
                            for field in reference_schema:
                                if field.name in table.column_names:
                                    col = table.column(field.name)
                                    if col.type == field.type:
                                        arrays.append(col)
                                    elif col.type == pa.null():
                                        arrays.append(pa.nulls(len(table), type=field.type))
                                    else:
                                        try:
                                            arrays.append(col.cast(field.type))
                                        except (
                                            pa.ArrowInvalid,
                                            pa.ArrowNotImplementedError,
                                        ):
                                            type_warnings.append(f"'{field.name}' ({col.type}->{field.type})")
                                            arrays.append(pa.nulls(len(table), type=field.type))
                                else:
                                    arrays.append(pa.nulls(len(table), type=field.type))

                            table = pa.Table.from_arrays(arrays, schema=reference_schema)
                            if type_warnings:
                                self.logger.warning(
                                    f"Chunk {i}: Incompatible columns replaced with nulls: "
                                    f"{', '.join(type_warnings)}"
                                )
                            schema_promotion_count += 1
                    # Write one row group per class_id so pyarrow's predicate
                    # pushdown can skip non-matching row groups during filtered reads.
                    # Uses zero-copy table.slice() on the sorted table instead of
                    # per-class filter() to avoid N full-table scans and row copies.
                    if "class_id" in table.column_names:
                        table = table.sort_by("class_id")
                        n = table.num_rows
                        if n > 0:
                            class_vals = table.column("class_id").to_pylist()
                            run_start = 0
                            for j in range(1, n):
                                if class_vals[j] != class_vals[run_start]:
                                    pqwriter.write_table(table.slice(run_start, j - run_start))
                                    run_start = j
                            pqwriter.write_table(table.slice(run_start, n - run_start))
                    else:
                        pqwriter.write_table(table)

        finally:
            # Cancel queued futures and wait for running ones to finish,
            # ensuring no background threads hold pyarrow Tables or S3 connections.
            executor.shutdown(wait=True, cancel_futures=True)

        # Close the writer and upload to S3 if needed
        if pqwriter is not None:
            pqwriter.close()

            if schema_promotion_count > 0:
                self.logger.debug(
                    f"Schema promotion: {schema_promotion_count}/{len(feature_paths)} "
                    f"chunks had null-type columns promoted to match reference schema"
                )

            if mismatch_count > 0:
                parts = [f"{col}: {variable_col_counts[col]} chunks" for col in sorted(variable_col_counts)]
                self.logger.debug(
                    f"Schema alignment: {mismatch_count}/{len(feature_paths)} chunks "
                    f"had variable columns (padded with nulls where absent): " + " | ".join(parts)
                )

            # Log final status
            used_gb, total_gb = get_memory_info_cgroup_aware()
            mem_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0.0
            cpu_pct, cpu_count = get_cpu_info_cgroup_aware()
            final_file_size_gb = storage.file_size(local_write_path) / (1024**3)
            self.logger.debug(
                f"Successfully streamed to geoparquet without temporary files. "
                f"Memory: {used_gb:.2f}GB / {total_gb:.2f}GB ({mem_pct:.1f}%). "
                f"CPU: {cpu_pct:.0f}% of {cpu_count:.0f}. "
                f"Final file size: {final_file_size_gb:.2f}GB"
            )

            # Upload to S3 if needed.  Keep the local staging file so that
            # per-class export can read from local disk instead of re-downloading
            # from S3 for each class.  _cleanup_staging() removes it at the end.
            if self.is_s3_output:
                self.logger.info(f"Uploading features.parquet to S3 ({final_file_size_gb:.1f}GB)...")
                storage.upload_file(local_write_path, outpath_features)

        if pqwriter is not None:
            return local_write_path
        else:
            self.logger.warning("No feature data found to write")
            return None

    def _merge_per_class_chunks(
        self,
        primary_ids_df: pd.DataFrame,
        aoi_input_columns: list,
        tabular_file_format: str = "parquet",
        requested_class_ids: set = None,
    ):
        """Merge per-class chunk files into final per-class output files.

        For each whitelisted class, globs per-class chunk files from chunks/,
        reads them as Arrow tables, concatenates via _unify_and_concat_tables(),
        and writes the final files to final/.

        If per-class chunk files are missing for some chunks (e.g. old exports,
        or per-class computation failed), falls back to recomputing from the
        corresponding feature chunk file.
        """
        whitelisted_classes = sorted(PER_CLASS_FILE_CLASS_IDS)
        if requested_class_ids is not None:
            whitelisted_classes = [cid for cid in whitelisted_classes if cid in requested_class_ids]
        read_workers = S3_PARALLEL_READ_WORKERS if self.is_s3_output else PARALLEL_READ_WORKERS

        # Set up staging directory for atomic writes
        if self.is_s3_output:
            staging_dir = self._local_final_staging
        else:
            staging_dir = os.path.join(self.final_path, ".per_class_staging")
            os.makedirs(staging_dir, exist_ok=True)

        # Build geoparquet metadata (shared across all per-class geo files)
        crs_projjson = pyproj.CRS(API_CRS).to_json_dict()
        geo_metadata = {
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "encoding": "WKB",
                    "geometry_types": [],
                    "crs": crs_projjson,
                    "edges": "planar",
                    "orientation": "counterclockwise",
                }
            },
        }

        tabular_count = 0
        geo_count = 0

        for cid in whitelisted_classes:
            desc = FEATURE_CLASS_DESCRIPTIONS.get(cid, f"class_{cid[:8]}")
            cname = _description_to_cname(desc)

            # Match per-class chunk files by exact class name + numeric chunk ID.
            # A plain glob like "roof_*.parquet" would also match "roof_instance_*.parquet";
            # requiring a digit after the prefix prevents this collision.
            tabular_re, geo_re = _per_class_chunk_regexes(cname)
            all_parquets = storage.glob_files(self.chunk_path, f"{cname}_*.parquet")
            tabular_chunks = sorted(p for p in all_parquets if tabular_re.match(storage.basename(p)))
            geo_chunks = sorted(p for p in all_parquets if geo_re.match(storage.basename(p)))

            if not tabular_chunks and not geo_chunks:
                self.logger.warning(
                    f"No per-class chunk files found for {desc}. "
                    f"To regenerate, delete rollup chunk files and re-run the export. "
                    f"Skipping {desc}."
                )
                continue

            # Happy path: per-class chunk files exist — read and concat
            if tabular_chunks:
                self.logger.info(f"Merging {len(tabular_chunks)} tabular chunks for {desc}")
                tabular_tables = []
                with ThreadPoolExecutor(max_workers=read_workers) as executor:
                    futures = {executor.submit(pq.read_table, p): p for p in tabular_chunks}
                    for future in as_completed(futures):
                        try:
                            tabular_tables.append(future.result())
                        except Exception as e:
                            self.logger.error(f"Failed reading {futures[future]}: {e}")
                if tabular_tables:
                    combined = _unify_and_concat_tables(tabular_tables)
                    staging_path = os.path.join(staging_dir, f"{cname}.parquet")
                    pq.write_table(combined, staging_path)
                    tabular_count += 1
                    del combined, tabular_tables

            if geo_chunks:
                self.logger.info(f"Merging {len(geo_chunks)} geo chunks for {desc}")
                geo_tables = []
                with ThreadPoolExecutor(max_workers=read_workers) as executor:
                    futures = {executor.submit(pq.read_table, p): p for p in geo_chunks}
                    for future in as_completed(futures):
                        try:
                            geo_tables.append(future.result())
                        except Exception as e:
                            self.logger.error(f"Failed reading {futures[future]}: {e}")
                if geo_tables:
                    combined = _unify_and_concat_tables(geo_tables)
                    existing_meta = combined.schema.metadata or {}
                    existing_meta[b"geo"] = json.dumps(geo_metadata).encode("utf-8")
                    combined = combined.replace_schema_metadata(existing_meta)
                    staging_path = os.path.join(staging_dir, f"{cname}_features.parquet")
                    pq.write_table(combined, staging_path)
                    geo_count += 1
                    del combined, geo_tables

        # Convert tabular parquet to CSV if requested
        if tabular_file_format == "csv":
            for cid in whitelisted_classes:
                desc = FEATURE_CLASS_DESCRIPTIONS.get(cid, f"class_{cid[:8]}")
                cname = _description_to_cname(desc)
                parquet_path = os.path.join(staging_dir, f"{cname}.parquet")
                if os.path.exists(parquet_path):
                    csv_path = os.path.join(staging_dir, f"{cname}.csv")
                    df = pd.read_parquet(parquet_path)
                    df.to_csv(csv_path, index=False)
                    os.remove(parquet_path)

        # Move files from staging to final destination (or upload to S3)
        staged_files = (
            [f for f in os.listdir(staging_dir) if f.endswith((".parquet", ".csv"))]
            if os.path.isdir(staging_dir)
            else []
        )

        if self.is_s3_output:
            for fname in staged_files:
                local_path = os.path.join(staging_dir, fname)
                s3_path = storage.join_path(self.final_path, fname)
                storage.upload_file(local_path, s3_path)
                self.logger.info(f"  Uploaded {fname}")
        else:
            for fname in staged_files:
                src = os.path.join(staging_dir, fname)
                dst = os.path.join(self.final_path, fname)
                os.replace(src, dst)
            try:
                os.rmdir(staging_dir)
            except OSError:
                pass

        self.logger.info(f"Per-class merge complete: " f"{tabular_count} tabular + {geo_count} geo files")

    def get_chunk_output_file(self, chunk_id: str) -> str:
        """
        Get the path to the main output file for a chunk.

        Args:
            chunk_id: Unique identifier for this chunk

        Returns:
            Path to the chunk's rollup file (used for cache checking)
        """
        return storage.join_path(self.chunk_path, f"rollup_{chunk_id}.parquet")

    def process_chunk(
        self,
        chunk_id: str,
        aoi_gdf: gpd.GeoDataFrame,
        classes_df: pd.DataFrame = None,
        progress_counters: dict = None,
        **kwargs,
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
        final_features_df = None
        _t_rollup_start = None
        _t_rollup_end = None
        _t_post_merge = None
        _t_features_prep = None
        _t_features_write = None
        _t_per_class_start = None
        _t_per_class_compute = None
        _t_per_class_writes = None

        chunk_start_time = datetime.now(timezone.utc).isoformat()
        chunk_start_monotonic = time.monotonic()

        try:
            if self.cache_dir is None and not self.no_cache:
                cache_dir = self.output_dir
            else:
                cache_dir = self.cache_dir if self.cache_dir else self.output_dir

            # Separate cache paths for each API
            if not self.no_cache:
                feature_api_cache_path = storage.join_path(str(cache_dir), "cache", "feature_api")
                roof_age_cache_path = storage.join_path(str(cache_dir), "cache", "roof_age")
            else:
                feature_api_cache_path = None
                roof_age_cache_path = None

            outfile = storage.join_path(self.chunk_path, f"rollup_{chunk_id}.parquet")
            outfile_features = storage.join_path(self.chunk_path, f"features_{chunk_id}.parquet")
            outfile_errors = storage.join_path(self.chunk_path, f"feature_api_errors_{chunk_id}.parquet")
            outfile_roof_age_errors = storage.join_path(self.chunk_path, f"roof_age_errors_{chunk_id}.parquet")
            if storage.file_exists(outfile) and storage.validate_parquet(outfile):
                return {"chunk_id": chunk_id, "latency_stats": None}

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
                self.logger.debug(f"Chunk {chunk_id}: Getting rollups for {len(aoi_gdf)} AOIs ({self.endpoint=})")
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
                used_gb, total_gb = get_memory_info_cgroup_aware()
                mem_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0.0
                cpu_pct, cpu_count = get_cpu_info_cgroup_aware()
                self.logger.debug(
                    f"Chunk {chunk_id} failed {len(errors_df)} of {len(aoi_gdf)} AOI requests. "
                    f"{len(rollup_df)} rollups returned on {len(rollup_df.index.unique())} unique {rollup_df.index.name}s. "
                    f"Memory: {used_gb:.1f}GB / {total_gb:.1f}GB ({mem_pct:.1f}%). "
                    f"CPU: {cpu_pct:.0f}% of {cpu_count:.0f}"
                )
                if len(errors_df) > 0:
                    if "message" in errors_df:
                        # Sanitize URLs in messages before aggregating (truncate query params)
                        sanitized_messages = errors_df["message"].apply(sanitize_error_message)
                        error_counts = sanitized_messages.value_counts().to_dict()
                        self.logger.debug(f"Found {len(errors_df)} errors by type: {error_counts}")
                    else:
                        self.logger.debug(f"Found {len(errors_df)} errors")
                if len(errors_df) == len(aoi_gdf):
                    storage.write_parquet(errors_df, outfile_errors)
                    latency_stats = feature_api.get_latency_stats()
                    if latency_stats:
                        latency_stats["chunk_id"] = chunk_id
                        latency_stats["start_time"] = chunk_start_time
                        latency_stats["end_time"] = datetime.now(timezone.utc).isoformat()
                        latency_stats["total_duration_ms"] = (time.monotonic() - chunk_start_monotonic) * 1000
                        save_chunk_latency_stats(latency_stats, self.chunk_path, chunk_id)
                    return {"chunk_id": chunk_id, "latency_stats": latency_stats}
            elif self.endpoint == Endpoint.FEATURE.value:
                self.logger.debug(f"Chunk {chunk_id}: Getting features for {len(aoi_gdf)} AOIs ({self.endpoint=})")
                features_gdf, metadata_df, errors_df = feature_api.get_features_gdf_bulk(
                    aoi_gdf,
                    region=self.country,
                    since_bulk=self.since,
                    until_bulk=self.until,
                    packs=self.packs,
                    classes=self.classes,
                    include=self.include,
                    max_allowed_error_pct=100,
                )

                # Filter out deprecated feature classes. RSI resolution now uses IoU-based
                # Roof → Building(New) → BL linkage rather than the API's parent_id chain
                # through Building(Deprecated), so no unfiltered reference is needed.
                if len(features_gdf) > 0 and "class_id" in features_gdf.columns:
                    pre_filter_count = len(features_gdf)
                    features_gdf = features_gdf[~features_gdf["class_id"].isin(DEPRECATED_CLASS_IDS)]
                    if len(features_gdf) < pre_filter_count:
                        logger.debug(
                            f"Chunk {chunk_id}: Filtered {pre_filter_count - len(features_gdf)} deprecated features"
                        )
                # Null Roof parent_id — it previously pointed to Building(Deprecated) which
                # is now filtered out. Prevents accidental traversal to non-existent features.
                if len(features_gdf) > 0 and "class_id" in features_gdf.columns and "parent_id" in features_gdf.columns:
                    roof_mask = features_gdf["class_id"] == ROOF_ID
                    if roof_mask.any():
                        features_gdf = features_gdf.copy()
                        features_gdf.loc[roof_mask, "parent_id"] = None

                used_gb, total_gb = get_memory_info_cgroup_aware()
                mem_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0.0
                cpu_pct, cpu_count = get_cpu_info_cgroup_aware()
                self.logger.debug(
                    f"Chunk {chunk_id} failed {len(errors_df)} of {len(aoi_gdf)} AOI requests. "
                    f"Memory: {used_gb:.1f}GB / {total_gb:.1f}GB ({mem_pct:.1f}%). "
                    f"CPU: {cpu_pct:.0f}% of {cpu_count:.0f}"
                )
                if len(errors_df) > 0:
                    if "message" in errors_df:
                        # Sanitize URLs in messages before aggregating (truncate query params)
                        sanitized_messages = errors_df["message"].apply(sanitize_error_message)
                        error_counts = sanitized_messages.value_counts().to_dict()
                        self.logger.debug(f"Found {len(errors_df)} errors by type: {error_counts}")
                    else:
                        self.logger.debug(f"Found {len(errors_df)} errors")
                # Track Feature API success per AOI
                feature_api_errors_df = errors_df.copy()
                feature_api_success_aois = (
                    set(aoi_gdf.index) - set(errors_df.index) if len(errors_df) > 0 else set(aoi_gdf.index)
                )

                # Query Roof Age API if enabled
                roof_age_gdf = gpd.GeoDataFrame(columns=[AOI_ID_COLUMN_NAME, "geometry"], crs=API_CRS)
                roof_age_errors_df = pd.DataFrame()
                roof_age_metadata_df = pd.DataFrame()

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
                        roof_age_errors_df = pd.DataFrame(
                            {
                                AOI_ID_COLUMN_NAME: aoi_gdf.index.tolist(),
                                "status_code": [-1] * len(aoi_gdf),
                                "message": [str(e)] * len(aoi_gdf),
                            }
                        ).set_index(AOI_ID_COLUMN_NAME)

                # Track Roof Age API success per AOI
                roof_age_success_aois = (
                    set(aoi_gdf.index) - set(roof_age_errors_df.index)
                    if len(roof_age_errors_df) > 0
                    else set(aoi_gdf.index)
                )

                # If all Feature API requests failed and no roof age data, save errors and return
                if len(feature_api_errors_df) == len(aoi_gdf) and len(roof_age_gdf) == 0:
                    storage.write_parquet(feature_api_errors_df, outfile_errors)
                    if len(roof_age_errors_df) > 0:
                        storage.write_parquet(roof_age_errors_df, outfile_roof_age_errors)
                    chunk_end_time = datetime.now(timezone.utc).isoformat()
                    total_duration_ms = (time.monotonic() - chunk_start_monotonic) * 1000
                    latency_stats = collect_latency_stats_from_apis(
                        [feature_api, roof_age_api],
                        chunk_id,
                        chunk_start_time,
                        chunk_end_time,
                        total_duration_ms,
                    )
                    if latency_stats is not None:
                        save_chunk_latency_stats(latency_stats, self.chunk_path, chunk_id)
                    return {"chunk_id": chunk_id, "latency_stats": latency_stats}

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
                            warnings.filterwarnings(
                                "ignore",
                                message=".*concatenation with empty or all-NA.*",
                            )
                            features_gdf = gpd.GeoDataFrame(
                                pd.concat(dfs_to_concat, ignore_index=False),
                                crs=API_CRS,
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
                            (features_gdf["class_id"] != ROOF_ID)
                            & (features_gdf["class_id"] != roof_age_gdf["class_id"].iloc[0])
                        ]
                        dfs_to_concat = [
                            df
                            for df in [
                                non_roof_features,
                                roofs_gdf_linked,
                                roof_age_gdf_linked,
                            ]
                            if len(df) > 0
                        ]
                        if dfs_to_concat:
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    "ignore",
                                    message=".*concatenation with empty or all-NA.*",
                                )
                                features_gdf = gpd.GeoDataFrame(
                                    pd.concat(dfs_to_concat, ignore_index=False),
                                    crs=API_CRS,
                                )
                        logger.debug(f"Chunk {chunk_id}: After linking, features_gdf has {len(features_gdf)} rows")

                        # Calculate roof age in years for roofs with linked roof instances
                        # This adds primary_child_roof_age_years_as_of_date to roofs
                        if (
                            "primary_child_roof_age_installation_date" in features_gdf.columns
                            and "primary_child_roof_age_as_of_date" in features_gdf.columns
                        ):
                            roofs_with_age_mask = (
                                (features_gdf["class_id"] == ROOF_ID)
                                & features_gdf["primary_child_roof_age_installation_date"].notna()
                                & features_gdf["primary_child_roof_age_as_of_date"].notna()
                            )
                            if roofs_with_age_mask.any():
                                age_years = calculate_roof_age_years(
                                    features_gdf.loc[
                                        roofs_with_age_mask,
                                        "primary_child_roof_age_installation_date",
                                    ],
                                    features_gdf.loc[
                                        roofs_with_age_mask,
                                        "primary_child_roof_age_as_of_date",
                                    ],
                                )
                                if age_years is not None:
                                    features_gdf.loc[
                                        roofs_with_age_mask,
                                        "primary_child_roof_age_years_as_of_date",
                                    ] = age_years.values
                                    logger.debug(
                                        f"Chunk {chunk_id}: Calculated roof age in years for {roofs_with_age_mask.sum()} roofs"
                                    )

                        # Calculate roof age in years for roof instances
                        if "installation_date" in features_gdf.columns and "as_of_date" in features_gdf.columns:
                            roof_instance_mask = features_gdf["class_id"] == ROOF_INSTANCE_CLASS_ID
                            if roof_instance_mask.any():
                                age_years = calculate_roof_age_years(
                                    features_gdf.loc[roof_instance_mask, "installation_date"],
                                    features_gdf.loc[roof_instance_mask, "as_of_date"],
                                )
                                if age_years is not None:
                                    features_gdf.loc[roof_instance_mask, "roof_age_years_as_of_date"] = age_years.values
                                    logger.debug(
                                        f"Chunk {chunk_id}: Calculated roof age in years for {roof_instance_mask.sum()} roof instances"
                                    )

                # features_gdf now includes roof instances (and IoU-linked columns on roofs)
                # from roof age processing — no rebuild needed.

                # Build API metadata pairs for parcel_rollup: each pair tells
                # the rollup which classes an API covers and which AOIs it succeeded for.
                feature_api_classes = classes_df[classes_df.index != ROOF_INSTANCE_CLASS_ID]
                api_metadata = [(metadata_df, feature_api_classes)]
                if self.roof_age:
                    roof_age_classes = classes_df[classes_df.index == ROOF_INSTANCE_CLASS_ID]
                    api_metadata.append((roof_age_metadata_df, roof_age_classes))

                # RSI resolution uses IoU-based Roof→Building(New)→BL linkage;
                # Building(Deprecated) has been filtered from features_gdf.
                _t_rollup_start = time.monotonic()
                _profile_chunk = int(os.environ.get("NMAIPY_PROFILE_CHUNK", "-1"))
                if chunk_id == _profile_chunk:
                    _pr = cProfile.Profile()
                    _pr.enable()
                rollup_df = parcels.parcel_rollup(
                    aoi_gdf,
                    features_gdf,
                    classes_df,
                    country=self.country,
                    primary_decision=self.primary_decision,
                    api_metadata=api_metadata,
                )
                if chunk_id == _profile_chunk:
                    _pr.disable()
                    _profile_path = f"/tmp/nmaipy_profile_chunk_{chunk_id}.txt"
                    with open(_profile_path, "w") as _pf:
                        pstats.Stats(_pr, stream=_pf).sort_stats("cumulative").print_stats(40)
                    self.logger.info(f"CHUNK_PROFILE parcel_rollup stats saved to {_profile_path}")
                _t_rollup_end = time.monotonic()

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
                    storage.write_parquet(roof_age_errors_df, outfile_roof_age_errors)

                # Use Feature API errors as the main errors file
                errors_df = feature_api_errors_df

            else:
                self.logger.error(f"Not a valid endpoint selection: {self.endpoint}")
                sys.exit(1)

            # Put it all together and save
            meta_data_columns = [
                "system_version",
                "link",
                "survey_date",
                "survey_id",
                "survey_resource_id",
                "perspective",
                "postcat",
            ]
            # Rename metadata columns that clash with user's AOI columns
            conflicting_columns = [c for c in meta_data_columns if c in aoi_gdf.columns]
            for meta_data_column in conflicting_columns:
                metadata_df = metadata_df.rename(columns={meta_data_column: f"nmaipy_{meta_data_column}"})
                meta_data_columns.remove(meta_data_column)

            # Use rollup_df as base to preserve all AOIs (including those where Feature API
            # failed but Roof Age API succeeded). Left-merge with metadata_df to add
            # survey metadata where available.
            final_df = rollup_df.merge(metadata_df, on=AOI_ID_COLUMN_NAME, how="left").merge(
                aoi_gdf, on=AOI_ID_COLUMN_NAME
            )
            # Ensure metadata columns exist even when metadata_df was empty
            # (e.g. all Feature API requests failed) for consistent schema
            for col in meta_data_columns:
                if col not in final_df.columns:
                    final_df[col] = None
            parcel_columns = [c for c in aoi_gdf.columns if c != "geometry"]
            columns = (
                parcel_columns
                + [c for c in meta_data_columns if c in final_df.columns]
                + [c for c in final_df.columns if c not in parcel_columns + meta_data_columns + ["geometry"]]
            )
            final_df = final_df[columns]
            if self.include_parcel_geometry:
                columns.append("geometry")
            columns = [c for c in columns if c in final_df.columns]
            date2str = lambda d: str(d).replace("-", "")
            make_link = (
                lambda d: f"https://apps.nearmap.com/maps/#/@{d.query_aoi_lat},{d.query_aoi_lon},21.00z,0d/V/{date2str(d.survey_date)}?locationMarker"
            )
            if self.endpoint == Endpoint.ROLLUP.value:
                if "query_aoi_lat" in final_df.columns and "query_aoi_lon" in final_df.columns:
                    final_df["link"] = final_df.apply(make_link, axis=1)
                final_df = final_df.drop(columns=["system_version", "survey_date"])
            _t_post_merge = time.monotonic()
            self.logger.debug(
                f"Chunk {chunk_id}: Writing {len(final_df)} rows for rollups and {len(errors_df)} for errors."
            )
            try:
                # Convert errors_df to GeoDataFrame if it has geometry (from failed grid squares)
                # This ensures proper geoparquet output that can be read in GIS software
                if "geometry" in errors_df.columns and len(errors_df) > 0:
                    errors_gdf = gpd.GeoDataFrame(errors_df, geometry="geometry", crs=API_CRS)
                    storage.write_parquet(errors_gdf, outfile_errors)
                else:
                    storage.write_parquet(errors_df, outfile_errors)
            except Exception as e:
                self.logger.error(
                    f"Chunk {chunk_id}: Failed writing errors_df ({len(errors_df)} rows) to {outfile_errors}."
                )
                self.logger.error(f"Error: {type(e).__name__}: {str(e)}")
            if self.endpoint != Endpoint.ROLLUP.value:
                # Drop survey_date from metadata_df to avoid collision with features_gdf
                # (features_gdf has per-feature survey_date which is more accurate for gridded AOIs)
                metadata_cols_to_drop = [
                    c for c in ["survey_date"] if c in metadata_df.columns and c in features_gdf.columns
                ]
                if metadata_cols_to_drop:
                    metadata_df = metadata_df.drop(columns=metadata_cols_to_drop)

                # Check for column name collisions between any two dataframes
                final_features_df = aoi_gdf.rename(columns=dict(geometry="aoi_geometry"))

                metadata_cols = set(metadata_df.columns)
                features_cols = set(features_gdf.columns)
                aoi_cols = set(final_features_df.columns)
                metadata_features_overlap = metadata_cols & features_cols - {AOI_ID_COLUMN_NAME}
                metadata_aoi_overlap = metadata_cols & aoi_cols - {AOI_ID_COLUMN_NAME}
                features_aoi_overlap = features_cols & aoi_cols - {AOI_ID_COLUMN_NAME}
                all_overlapping = metadata_features_overlap | metadata_aoi_overlap | features_aoi_overlap
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
                geom_cols = [col for col in merged2.columns if "geometry" in col.lower()]

                # Create GeoDataFrame with the appropriate geometry column
                if "geometry" in merged2.columns:
                    final_features_df = gpd.GeoDataFrame(merged2, geometry="geometry", crs=API_CRS)
                elif "geometry_y" in merged2.columns:
                    # Features geometry (from poles)
                    final_features_df = gpd.GeoDataFrame(merged2, geometry="geometry_y", crs=API_CRS)
                elif "geometry_x" in merged2.columns:
                    # AOI geometry
                    final_features_df = gpd.GeoDataFrame(merged2, geometry="geometry_x", crs=API_CRS)
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
                    final_features_df["aoi_geometry"] = final_features_df.aoi_geometry.to_wkt()

                # Apply flattening to attributes if present
                if "attributes" in final_features_df.columns:
                    # Use pd.DataFrame(list_of_dicts) instead of .apply(pd.Series) for 100x+ speedup
                    flat_attr_list = final_features_df["attributes"].apply(_flatten_attribute_list).tolist()
                    flattened_attrs = pd.DataFrame(flat_attr_list, index=final_features_df.index)
                    if not flattened_attrs.empty and len(flattened_attrs.columns) > 0:
                        logger.debug(f"Chunk {chunk_id}: Flattened {len(flattened_attrs.columns)} attribute columns")
                        new_cols = [c for c in flattened_attrs.columns if c not in final_features_df.columns]
                        if new_cols:
                            final_features_df = pd.concat([final_features_df, flattened_attrs[new_cols]], axis=1)

                    # Serialize attributes to JSON string for parquet compatibility
                    # (preserves the original data for rollup/building export paths)
                    final_features_df["attributes"] = final_features_df["attributes"].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                    )

                # Serialize damage to JSON string (flattening happens in class-specific exports)
                # This avoids column explosion in mixed-class parcel_features.parquet
                if "damage" in final_features_df.columns:
                    final_features_df["damage"] = final_features_df["damage"].apply(
                        lambda x: json.dumps(x) if isinstance(x, dict) else x
                    )
                if len(final_features_df) > 0:
                    try:
                        if not self.include_parcel_geometry and "aoi_geometry" in final_features_df.columns:
                            final_features_df = final_features_df.drop(columns=["aoi_geometry"])
                        final_features_df = final_features_df[
                            ~(final_features_df.geometry.is_empty | final_features_df.geometry.isna())
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
                        object_columns = final_features_df.select_dtypes(include=["object"]).columns
                        object_columns = [col for col in object_columns if col != "geometry"]

                        for col in object_columns:
                            final_features_df[col] = final_features_df[col].apply(serialize_include_param)

                        # Ensure it's a proper GeoDataFrame before saving to parquet
                        if not isinstance(final_features_df, gpd.GeoDataFrame):
                            final_features_df = gpd.GeoDataFrame(final_features_df, geometry="geometry", crs=API_CRS)
                        else:
                            final_features_df = final_features_df.set_crs(API_CRS, allow_override=True)

                        # Reset index to preserve aoi_id as a column (needed for building-roof linking)
                        if final_features_df.index.name == AOI_ID_COLUMN_NAME:
                            final_features_df = final_features_df.reset_index()

                        # Save with explicit schema version for better QGIS compatibility
                        # Requires geopandas >= 1.1.0
                        _t_features_prep = time.monotonic()
                        try:
                            storage.write_parquet(
                                final_features_df,
                                outfile_features,
                                index=False,
                                schema_version="1.0.0",
                            )
                        except (TypeError, ValueError) as e:
                            # Fallback for older geopandas or pyarrow versions
                            self.logger.debug(f"Could not use schema_version parameter: {e}. Falling back to default.")
                            storage.write_parquet(final_features_df, outfile_features, index=False)
                        _t_features_write = time.monotonic()
                    except Exception as e:
                        self.logger.error(
                            f"Failed to save features parquet file for chunk_id {chunk_id}. Errors saved to {outfile_errors}."
                        )
                        self.logger.error(f"Error type: {type(e).__name__}, Error message: {str(e)}")
                        self.logger.error(e)

            # Per-class export: compute and write per-class tabular + geo parquet
            # while final_features_df is still in memory from the features section above.
            if (
                self.class_level_files
                and self.endpoint != Endpoint.ROLLUP.value
                and final_features_df is not None
                and len(final_features_df) > 0
            ):
                try:
                    # Extract primary feature IDs from this chunk's rollup (AOI_ID as index)
                    primary_cols = [c for c in PRIMARY_FEATURE_COLUMN_TO_CLASS if c in final_df.columns]
                    if primary_cols:
                        rollup_indexed = final_df
                        if AOI_ID_COLUMN_NAME in rollup_indexed.columns:
                            rollup_indexed = rollup_indexed.set_index(AOI_ID_COLUMN_NAME)
                        primary_ids_df = rollup_indexed[primary_cols].copy()
                    else:
                        primary_ids_df = pd.DataFrame()
                    chunk_gdf = _add_is_primary_column(final_features_df, primary_ids_df)

                    # Compute aoi_input_columns (user columns from AOI file)
                    system_columns = {
                        AOI_ID_COLUMN_NAME,
                        "geometry",
                        SINCE_COL_NAME,
                        UNTIL_COL_NAME,
                        SURVEY_RESOURCE_ID_COL_NAME,
                        "query_aoi_lat",
                        "query_aoi_lon",
                    }
                    aoi_input_columns = [
                        c for c in aoi_gdf.columns if c not in system_columns and c not in ADDRESS_FIELDS
                    ]

                    _t_per_class_start = time.monotonic()
                    per_class_results = _compute_all_per_class_data(
                        chunk_gdf=chunk_gdf,
                        country=self.country,
                        aoi_input_columns=aoi_input_columns,
                        logger_instance=self.logger,
                    )
                    _t_per_class_compute = time.monotonic()

                    for cid, tables in per_class_results.items():
                        desc = FEATURE_CLASS_DESCRIPTIONS.get(cid, f"class_{cid[:8]}")
                        cname = _description_to_cname(desc)

                        storage.write_parquet(
                            tables["tabular"],
                            storage.join_path(self.chunk_path, f"{cname}_{chunk_id}.parquet"),
                        )
                        if "geo" in tables:
                            storage.write_parquet(
                                tables["geo"],
                                storage.join_path(
                                    self.chunk_path,
                                    f"{cname}_features_{chunk_id}.parquet",
                                ),
                            )
                    _t_per_class_writes = time.monotonic()
                    del chunk_gdf, per_class_results
                except Exception as e:
                    self.logger.error(f"Per-class chunk export failed for chunk {chunk_id}: {e}")
                    # Non-fatal: rollup still written below, merge step falls back to
                    # recomputing per-class from feature chunks

            # Write rollup LAST — it serves as the chunk completion marker.
            # If the process is killed before this point, the chunk will be
            # re-processed on resume (errors, features, per-class files rewritten).
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
                    final_df = gpd.GeoDataFrame(final_df, geometry="geometry", crs=API_CRS)

                # Save with explicit schema version for better QGIS compatibility
                # Requires geopandas >= 1.1.0
                try:
                    storage.write_parquet(final_df, outfile, schema_version="1.0.0")
                except (TypeError, ValueError) as e:
                    # Fallback for older geopandas or pyarrow versions
                    self.logger.debug(f"Could not use schema_version parameter: {e}. Falling back to default.")
                    storage.write_parquet(final_df, outfile)
            except Exception as e:
                self.logger.error(f"Chunk {chunk_id}: Failed writing final_df ({len(final_df)} rows) to {outfile}.")
                self.logger.error(f"Error type: {type(e).__name__}, Error message: {str(e)}")

            _t_rollup_write = time.monotonic()

            # Emit structured timing breakdown for profiling the closeout stage.
            # Grep for CHUNK_TIMING in the export log to analyse the dead-zone split.
            # Phases that didn't run (e.g. ROLLUP endpoint skips features; class_level_files=False
            # skips per-class) report 0.0s because start==end for that segment.
            if _t_rollup_start is not None:
                # Build a linear chain: each timer falls back to the previous one so
                # skipped phases contribute 0s rather than corrupting later deltas.
                _tp0 = _t_rollup_start
                _tp1 = _t_rollup_end if _t_rollup_end is not None else _tp0
                _tp2 = _t_post_merge if _t_post_merge is not None else _tp1
                _tp3 = _t_features_prep if _t_features_prep is not None else _tp2
                _tp4 = _t_features_write if _t_features_write is not None else _tp3
                _tp5 = _t_per_class_start if _t_per_class_start is not None else _tp4
                _tp6 = _t_per_class_compute if _t_per_class_compute is not None else _tp5
                _tp7 = _t_per_class_writes if _t_per_class_writes is not None else _tp6
                self.logger.info(
                    f"CHUNK_TIMING chunk_id={chunk_id} "
                    f"parcel_rollup={_tp1 - _tp0:.1f}s "
                    f"post_rollup_merge={_tp2 - _tp1:.1f}s "
                    f"features_prep={_tp3 - _tp2:.1f}s "
                    f"features_write={_tp4 - _tp3:.1f}s "
                    f"per_class_compute={_tp6 - _tp5:.1f}s "
                    f"per_class_writes={_tp7 - _tp6:.1f}s "
                    f"rollup_write={_t_rollup_write - _tp7:.1f}s"
                )

            chunk_end_time = datetime.now(timezone.utc).isoformat()
            total_duration_ms = (time.monotonic() - chunk_start_monotonic) * 1000
            latency_stats = collect_latency_stats_from_apis(
                [feature_api, roof_age_api],
                chunk_id,
                chunk_start_time,
                chunk_end_time,
                total_duration_ms,
            )
            if latency_stats is not None:
                save_chunk_latency_stats(latency_stats, self.chunk_path, chunk_id)

            self.logger.debug(f"Finished saving chunk {chunk_id}")
            return {"chunk_id": chunk_id, "latency_stats": latency_stats}

        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_id}: {e}")
            raise
        finally:
            # Clean up API clients to close network connections
            if feature_api is not None:
                try:
                    feature_api.cleanup()
                    del feature_api
                except Exception:
                    pass

            if roof_age_api is not None:
                try:
                    roof_age_api.cleanup()
                    del roof_age_api
                except Exception:
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
                except Exception:
                    pass

            except Exception:
                pass

    def run(self):
        try:
            self._run_inner()
        finally:
            self._cleanup_staging()

    def _run_inner(self):
        self.logger.info(f"nmaipy version: {__version__}")
        self.logger.debug("Starting parcel rollup")

        # Process a single AOI file
        aoi_path = self.aoi_file
        self.logger.info(f"Processing AOI file {aoi_path}")

        cache_path = storage.join_path(self.output_dir, "cache")
        if storage.is_s3_path(cache_path) and not self.no_cache:
            self.logger.warning(
                "API cache will be written to S3, which may be slow due to many small files. "
                "Consider using --cache-dir to set a local cache directory, or --no-cache to disable caching."
            )
        storage.ensure_directory(cache_path)
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

        # Auto-refresh class_descriptions.json if API descriptions have changed
        api_descriptions = dict(zip(classes_df.index, classes_df["description"]))
        stale = any(FEATURE_CLASS_DESCRIPTIONS.get(cid) != desc for cid, desc in api_descriptions.items())
        if stale:
            try:
                merged = dict(FEATURE_CLASS_DESCRIPTIONS)
                merged.update(api_descriptions)
                _write_class_descriptions(merged)
                FEATURE_CLASS_DESCRIPTIONS.update(api_descriptions)
                self.logger.info("Updated class_descriptions.json from API")
            except Exception as e:
                self.logger.debug(f"Could not update class_descriptions.json: {e}")

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

        # Output file paths in final directory (no stem prefix — directory provides context)
        outpath = storage.join_path(self.final_path, f"rollup.{self.tabular_file_format}")
        outpath_features = storage.join_path(self.final_path, "features.parquet")
        outpath_buildings = storage.join_path(self.final_path, f"buildings.{self.tabular_file_format}")

        # Check for existing output files and warn about overwriting.
        # We always rebuild from chunks (the source of truth) to avoid leaving
        # partial outputs from a previous interrupted run.
        existing_outputs = []
        for check_path in [outpath, outpath_features, outpath_buildings]:
            if storage.file_exists(check_path):
                existing_outputs.append(storage.basename(check_path))
        if existing_outputs:
            self.logger.warning(
                f"Overwriting {len(existing_outputs)} existing output file(s) in final/: " + ", ".join(existing_outputs)
            )

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
            logger.debug(f"No {SURVEY_RESOURCE_ID_COL_NAME} column provided, so date based endpoint will be used.")
            if SINCE_COL_NAME in aoi_gdf:
                logger.info(
                    f'The column "{SINCE_COL_NAME}" will be used as the earliest permitted date (YYYY-MM-DD) for each Query AOI.'
                )
            elif self.since is not None:
                logger.debug(f"The since date of {self.since} will limit the earliest returned date for all Query AOIs")
            else:
                logger.debug("No earliest date will be used")
            if UNTIL_COL_NAME in aoi_gdf:
                logger.info(
                    f'The column "{UNTIL_COL_NAME}" will be used as the latest permitted date (YYYY-MM-DD) for each Query AOI.'
                )
            elif self.until is not None:
                logger.debug(f"The until date of {self.until} will limit the latest returned date for all Query AOIs")
            else:
                logger.debug("No latest date will used")

        self.logger.debug(f"Using endpoint '{self.endpoint}' for rollups.")

        # Split into chunks and process in parallel (using BaseExporter methods)
        chunks_to_process, skipped_chunks, skipped_aois, num_chunks = self.split_into_chunks(aoi_gdf, check_cache=True)

        # Calculate initial AOI count for progress tracking (excluding skipped)
        # If roof_age is enabled, each AOI gets both Feature API and Roof Age API queries
        initial_aoi_count = len(aoi_gdf) - skipped_aois
        if self.roof_age:
            initial_aoi_count *= 2

        latency_csv_path = storage.join_path(self.final_path, "latency_stats.csv")

        self.run_parallel(
            chunks_to_process,
            initial_aoi_count=initial_aoi_count,
            use_progress_tracking=True,  # Enable progress counters for Feature API
            classes_df=classes_df,  # Pass classes_df to process_chunk
        )

        all_latency_stats = combine_chunk_latency_stats(self.chunk_path, latency_csv_path)
        if all_latency_stats:
            global_stats = compute_global_latency_stats(all_latency_stats)
            if global_stats and global_stats.get("count", 0) > 0:
                self.logger.info(
                    f"Global latency stats: "
                    f"mean={global_stats['mean']:.0f}ms, "
                    f"P50={global_stats['p50']:.0f}ms [{global_stats['p50_ci'][0]:.0f}-{global_stats['p50_ci'][1]:.0f}], "
                    f"P90={global_stats['p90']:.0f}ms [{global_stats['p90_ci'][0]:.0f}-{global_stats['p90_ci'][1]:.0f}], "
                    f"P95={global_stats['p95']:.0f}ms [{global_stats['p95_ci'][0]:.0f}-{global_stats['p95_ci'][1]:.0f}], "
                    f"P99={global_stats['p99']:.0f}ms [{global_stats['p99_ci'][0]:.0f}-{global_stats['p99_ci'][1]:.0f}], "
                    f"n={global_stats['count']}"
                )

        data = []
        data_features = []
        errors = []
        self.logger.debug(f"Saving rollup data as {self.tabular_file_format} file to {outpath}")

        # Phase 1: Check which chunk files exist (parallel for S3 HEAD requests)
        def _check_chunk_files(i):
            chunk_filename = f"rollup_{str(i).zfill(4)}.parquet"
            cp = storage.join_path(self.chunk_path, chunk_filename)
            if storage.file_exists(cp):
                if storage.validate_parquet(cp):
                    return (i, cp, None)
                else:
                    self.logger.warning(
                        f"Chunk {i}: rollup file {cp} exists but is corrupted "
                        f"(invalid parquet footer). Treating as missing."
                    )
            error_filename = f"feature_api_errors_{str(i).zfill(4)}.parquet"
            has_error = storage.file_exists(storage.join_path(self.chunk_path, error_filename))
            return (i, None, has_error)

        chunk_check_results = []
        read_workers = S3_PARALLEL_READ_WORKERS if self.is_s3_output else PARALLEL_READ_WORKERS
        last_resource_update = 0.0
        with ThreadPoolExecutor(max_workers=read_workers) as executor:
            futures = {executor.submit(_check_chunk_files, i): i for i in range(num_chunks)}
            pbar = tqdm(
                as_completed(futures),
                total=num_chunks,
                desc="Checking chunk files",
                file=sys.stdout,
                position=0,
                leave=True,
            )
            for future in pbar:
                now = time.time()
                if now - last_resource_update >= 5.0:
                    pbar.set_postfix_str(resource_postfix())
                    last_resource_update = now
                chunk_check_results.append(future.result())

        # Sort by index for deterministic error reporting
        chunk_check_results.sort(key=lambda x: x[0])

        rollup_paths_to_read = []
        for i, cp, has_error in chunk_check_results:
            if cp is not None:
                rollup_paths_to_read.append(cp)
            elif has_error:
                self.logger.debug(f"Chunk {i} rollup file missing, but error file found.")
            else:
                self.logger.error(
                    f"Both error and data files for chunk {i} missing or corrupted, "
                    f"indicating files failed to write, are corrupted, or have been "
                    f"deleted. Check the '{self.chunk_path}' directory to diagnose, "
                    f"then re-run the export."
                )
                sys.exit(1)

        # Phase 2: Read all valid rollup chunks in parallel
        if rollup_paths_to_read:
            data = _read_parquet_chunks_parallel(
                rollup_paths_to_read,
                max_workers=read_workers,
                desc=f"Reading {len(rollup_paths_to_read)} rollup chunks",
                logger=self.logger,
            )
        else:
            data = []
        if len(data) > 0:
            # Filter out DataFrames that are entirely NA (all values NA across all columns)
            data = [d for d in data if not d.isna().all().all()]
            self.logger.info(f"Concatenating {len(data)} rollup chunks...")
            # Suppress FutureWarning about all-NA columns in concat dtype inference.
            # Individual chunks may have all-NA columns for feature types not present
            # in that chunk; this doesn't affect correctness.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=".*concatenation with empty or all-NA.*",
                )
                data = pd.concat(data) if data else pd.DataFrame()
            if "geometry" in data.columns:
                if not isinstance(data.geometry, gpd.GeoSeries):
                    data["geometry"] = gpd.GeoSeries.from_wkt(data.geometry)
                data = gpd.GeoDataFrame(data, crs=API_CRS)

        else:
            data = pd.DataFrame(data)
        if len(data) > 0:
            self.logger.info(f"Writing rollup {self.tabular_file_format} ({len(data)} rows)...")
            if self.tabular_file_format == "parquet":
                storage.write_parquet(data, outpath, index=True)
            elif self.tabular_file_format == "csv":
                if "geometry" in data.columns:
                    if hasattr(data.geometry, "to_wkt") and callable(data.geometry.to_wkt):
                        # If it has a to_wkt method but isn't a GeoSeries
                        data["geometry"] = data.geometry.to_wkt()
                data.to_csv(outpath, index=True)
            else:
                self.logger.info("Invalid output format specified - reverting to csv")
                if "geometry" in data.columns:
                    if hasattr(data.geometry, "to_wkt") and callable(data.geometry.to_wkt):
                        # If it has a to_wkt method but isn't a GeoSeries
                        data["geometry"] = data.geometry.to_wkt()
                data.to_csv(outpath, index=True)

        # Extract only the primary feature ID columns needed for is_primary marking,
        # then free the full rollup DataFrame to reduce peak memory during per-class export.
        primary_cols = [c for c in PRIMARY_FEATURE_COLUMN_TO_CLASS if c in data.columns]
        primary_ids_df = data[primary_cols].copy() if primary_cols else pd.DataFrame()
        del data
        gc.collect()

        # Collect and save Feature API errors
        outpath_feature_api_errors = storage.join_path(self.final_path, "feature_api_errors.csv")
        outpath_feature_api_errors_geoparquet = storage.join_path(self.final_path, "feature_api_errors.parquet")
        self.logger.debug("Collecting Feature API errors")
        feature_api_error_paths = storage.glob_files(self.chunk_path, "feature_api_errors_*.parquet")
        if feature_api_error_paths:
            feature_api_errors_list = _read_parquet_chunks_parallel(
                feature_api_error_paths,
                max_workers=read_workers,
                desc="Reading Feature API error files",
                logger=self.logger,
                strict=False,
            )
            feature_api_errors = pd.concat(feature_api_errors_list) if feature_api_errors_list else pd.DataFrame()
        else:
            feature_api_errors = pd.DataFrame()

        # Collect and save Roof Age API errors (if roof_age was enabled)
        roof_age_errors = pd.DataFrame()
        if self.roof_age:
            outpath_roof_age_errors = storage.join_path(self.final_path, "roof_age_errors.csv")
            outpath_roof_age_errors_geoparquet = storage.join_path(self.final_path, "roof_age_errors.parquet")
            self.logger.debug("Collecting Roof Age API errors")
            roof_age_error_paths = storage.glob_files(self.chunk_path, "roof_age_errors_*.parquet")
            if roof_age_error_paths:
                roof_age_errors_list = _read_parquet_chunks_parallel(
                    roof_age_error_paths,
                    max_workers=read_workers,
                    desc="Reading Roof Age error files",
                    logger=self.logger,
                    strict=False,
                )
                roof_age_errors = pd.concat(roof_age_errors_list) if roof_age_errors_list else pd.DataFrame()

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
                        errors_gdf = gpd.GeoDataFrame(errors_with_context, geometry="geometry", crs=API_CRS)
                    else:
                        # No geometry yet - merge AOI geometry
                        errors_with_context = errors_df.merge(
                            aoi_gdf_for_merge[[AOI_ID_COLUMN_NAME, "geometry"]],
                            on=AOI_ID_COLUMN_NAME,
                            how="left",
                        )
                        if "geometry" in errors_with_context.columns:
                            errors_gdf = gpd.GeoDataFrame(
                                errors_with_context,
                                geometry="geometry",
                                crs=aoi_gdf.crs,
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
                    storage.write_parquet(errors_gdf, outpath_parquet, index=False)
            else:
                self.logger.info(f"{error_type}: No failures")

        # Save Feature API errors
        save_errors_to_files(
            feature_api_errors,
            outpath_feature_api_errors,
            outpath_feature_api_errors_geoparquet,
            "Feature API",
        )

        # Save Roof Age API errors
        if self.roof_age:
            save_errors_to_files(
                roof_age_errors,
                outpath_roof_age_errors,
                outpath_roof_age_errors_geoparquet,
                "Roof Age API",
            )

        # Free error DataFrames now that they've been saved to disk
        del feature_api_errors, roof_age_errors
        gc.collect()

        local_features_path = None
        if self.save_features:
            feature_paths = storage.glob_files(self.chunk_path, "features_*.parquet")
            self.logger.info(f"Saving feature data from {len(feature_paths)} geoparquet chunks to {outpath_features}")

            local_features_path = self._stream_and_convert_features(
                feature_paths,
                outpath_features,
            )

            # If buildings export is enabled, process building features
            if self.save_buildings:
                self.logger.info(f"Saving building-level data as {self.tabular_file_format} to {outpath_buildings}")
                # Define geoparquet path for buildings
                outpath_buildings_geoparquet = storage.join_path(self.final_path, "building_features.parquet")

                buildings_gdf = parcels.extract_building_features(
                    parcels_gdf=aoi_gdf,
                    features_gdf=None,
                    country=self.country,
                )
                if len(buildings_gdf) > 0:
                    # First, save the geoparquet version with intact geometries
                    self.logger.info(f"Saving building-level data as geoparquet to {outpath_buildings_geoparquet}")
                    try:
                        # Save with explicit schema version for better QGIS compatibility
                        # Requires geopandas >= 1.1.0
                        try:
                            storage.write_parquet(
                                buildings_gdf,
                                outpath_buildings_geoparquet,
                                schema_version="1.0.0",
                            )
                        except (TypeError, ValueError) as e:
                            # Fallback for older geopandas or pyarrow versions
                            self.logger.debug(f"Could not use schema_version parameter: {e}. Falling back to default.")
                            storage.write_parquet(buildings_gdf, outpath_buildings_geoparquet)
                    except Exception as e:
                        self.logger.error(f"Failed to save buildings geoparquet file: {str(e)}")

                    # Then convert geodataframe to plain dataframe for tabular output
                    # Keep geometry as WKT representation if needed
                    buildings_df = pd.DataFrame(buildings_gdf)
                    if "geometry" in buildings_df.columns:
                        buildings_df["geometry"] = buildings_df.geometry.apply(lambda geom: geom.wkt if geom else None)

                    # Save in the same format as rollup
                    if self.tabular_file_format == "parquet":
                        storage.write_parquet(buildings_df, outpath_buildings, index=True)
                    elif self.tabular_file_format == "csv":
                        buildings_df.to_csv(outpath_buildings, index=True)
                    else:
                        self.logger.info("Invalid output format specified for buildings - reverting to csv")
                        buildings_df.to_csv(outpath_buildings, index=True)
                else:
                    self.logger.info(f"No building features found for {Path(aoi_path).stem}")

        # Per-class export: merge pre-computed per-class chunk files into final files.
        # Per-class data was computed in process_chunk() while feature data was still
        # in memory, avoiding the expensive re-read of features.parquet.
        if self.class_level_files:
            # Compute aoi_input_columns from the AOI GeoDataFrame.
            system_columns = {
                AOI_ID_COLUMN_NAME,
                "geometry",
                SINCE_COL_NAME,
                UNTIL_COL_NAME,
                SURVEY_RESOURCE_ID_COL_NAME,
                "query_aoi_lat",
                "query_aoi_lon",
            }
            aoi_input_columns = [c for c in aoi_gdf.columns if c not in system_columns and c not in ADDRESS_FIELDS]
            self._merge_per_class_chunks(
                primary_ids_df=primary_ids_df,
                aoi_input_columns=aoi_input_columns,
                tabular_file_format=self.tabular_file_format,
                requested_class_ids=set(classes_df.index),
            )

        # Generate README data dictionary
        try:
            readme_gen = ReadmeGenerator(output_dir=self.final_path)
            readme_path = readme_gen.generate_and_save()
            self.logger.info(f"Generated README: {storage.basename(str(readme_path))}")
        except Exception as e:
            self.logger.warning(f"README generation warning: {e}")


# Backward compatibility alias
AOIExporter = NearmapAIExporter


def main():
    # Set higher file descriptor limits for running many processes in parallel.
    if sys.platform != "win32":
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = 32000  # Same as ulimit -n 32000
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
            new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            logger.info(f"File descriptor limits - Previous: {soft}, New: {new_soft}, Hard limit: {hard}")
        except ValueError as e:
            # If desired limit is too high, try setting to hard limit
            logger.warning(f"Could not set file descriptor limit to {desired}, trying hard limit {hard}")
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
                new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
                logger.info(f"File descriptor limits - Previous: {soft}, New: {new_soft}, Hard limit: {hard}")
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
        tabular_file_format=args.tabular_file_format,
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
