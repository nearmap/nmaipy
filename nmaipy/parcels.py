"""
Parcel/AOI Processing and Rollup Utilities

This module provides functions for:
- Creating rollup summaries of AI features at the parcel/AOI level
- Extracting building-style features for detailed export
- Linking roof instances to parent roof objects

The rollup functions aggregate multiple features within each AOI into summary statistics
and select "primary" features for detailed attribute extraction.
"""

import json
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.strtree import STRtree

from nmaipy import log
from nmaipy.aoi_io import read_from_file  # Re-export for backwards compatibility
from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    API_CRS,
    AREA_CRS,
    BUILDING_LIFECYCLE_ID,
    BUILDING_NEW_ID,
    BUILDING_STYLE_CLASS_IDS,
    CLASSES_WITH_PRIMARY_FEATURE,
    DEPRECATED_CLASS_IDS,
    IMPERIAL_COUNTRIES,
    LAT_PRIMARY_COL_NAME,
    LON_PRIMARY_COL_NAME,
    MIN_ROOF_INSTANCE_IOU_THRESHOLD,
    ROOF_ID,
    ROOF_INSTANCE_CLASS_ID,
    SQUARED_METERS_TO_SQUARED_FEET,
    MeasurementUnits,
)
from nmaipy.feature_attributes import (
    FALSE_STRING,
    TRUE_STRING,
    _parse_include_param,
    flatten_building_attributes,
    flatten_building_lifecycle_damage_attributes,
    flatten_roof_attributes,
    flatten_roof_instance_attributes,
)
from nmaipy.primary_feature_selection import DEFAULT_HIGH_CONFIDENCE_THRESHOLD as PRIMARY_FEATURE_HIGH_CONF_THRESH
from nmaipy.primary_feature_selection import (
    select_primary,
)

logger = log.get_logger()

# Re-export for backwards compatibility
__all__ = [
    "read_from_file",
    "parcel_rollup",
    "extract_building_features",
    "feature_attributes",
    "link_roof_instances_to_roofs",
    "link_roofs_to_buildings",
    "build_parent_lookup",
    "resolve_footprint_rsi",
    "calculate_child_feature_attributes",
    # Also re-export flattening functions for backwards compatibility
    "flatten_building_attributes",
    "flatten_building_lifecycle_damage_attributes",
    "flatten_roof_attributes",
    "flatten_roof_instance_attributes",
]


def _compute_iou(geom_a, geom_b) -> float:
    """
    Compute Intersection over Union (IoU) between two geometries.

    IoU = intersection_area / union_area

    Args:
        geom_a: First geometry (Shapely Polygon/MultiPolygon)
        geom_b: Second geometry (Shapely Polygon/MultiPolygon)

    Returns:
        IoU value between 0.0 and 1.0
    """
    if geom_a is None or geom_b is None:
        return 0.0
    if geom_a.is_empty or geom_b.is_empty:
        return 0.0
    try:
        intersection = geom_a.intersection(geom_b).area
        if intersection == 0:
            return 0.0
        union = geom_a.union(geom_b).area
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0


def calculate_child_feature_attributes(
    parent_geometry,
    components: list,
    child_features: gpd.GeoDataFrame,
    country: str,
    name_prefix: str = "",
    parent_projected=None,
    children_projected: gpd.GeoSeries = None,
) -> dict:
    """
    Calculate component attributes by spatial intersection for clipped features.

    This is data-driven: it extracts classId values from the components list,
    finds matching child features by class_id, and calculates areas/ratios based on
    the parent's clipped geometry.

    This function is used when a roof's geometry has been clipped by a parcel boundary.
    The original component ratios from the API are based on the full roof, so we need
    to recalculate them based on the clipped geometry.

    Geometries are projected to an equal-area CRS (AREA_CRS) before computing areas.

    Args:
        parent_geometry: The clipped parent polygon (e.g., roof geometry in EPSG:4326)
        components: List of component dicts with classId, description fields
                   (from roof's attributes.components)
        child_features: GeoDataFrame of all child features in the AOI (filtered
                       by the function to only those with matching classIds)
        country: Country code for units ("us" for imperial, "au" for metric)
        name_prefix: Optional prefix for output keys (e.g., "low_conf_" for
                    low-confidence attribute groups), must match the caller's naming
        parent_projected: Pre-projected parent geometry in equal-area CRS. When
                         provided, skips the per-call CRS projection (batch callers
                         project all geometries once up front for performance).
        children_projected: Pre-projected child geometries as a GeoSeries aligned
                           with child_features index. When provided, matching children
                           are looked up by index instead of re-projecting per call.

    Returns:
        Flattened dict with recalculated attributes (e.g., metal_area_sqft, hip_ratio),
        or None if recalculation could not be attempted (missing geometry/components/children).
    """
    if parent_geometry is None or parent_geometry.is_empty:
        return None
    if not components or child_features is None:
        return None

    # Use pre-projected parent geometry if provided, otherwise project per call
    projected_crs = AREA_CRS[country.lower()]
    if parent_projected is None:
        parent_projected = gpd.GeoSeries([parent_geometry], crs=API_CRS).to_crs(projected_crs).iloc[0]
    parent_area = parent_projected.area
    if parent_area <= 0:
        return None

    flattened = {}
    child_features_empty = len(child_features) == 0

    for component in components:
        class_id = component.get("classId")
        description = component.get("description", "unknown")
        if not class_id:
            continue

        name = name_prefix + description.lower().replace(" ", "_")
        flattened[f"{name}_class_id"] = class_id
        if child_features_empty:
            flattened[f"{name}_present"] = FALSE_STRING
            if country in IMPERIAL_COUNTRIES:
                flattened[f"{name}_area_sqft"] = 0.0
            else:
                flattened[f"{name}_area_sqm"] = 0.0
            flattened[f"{name}_ratio"] = 0.0
            continue
        matching_features = child_features[child_features.class_id == class_id]
        if len(matching_features) == 0:
            flattened[f"{name}_present"] = FALSE_STRING
            if country in IMPERIAL_COUNTRIES:
                flattened[f"{name}_area_sqft"] = 0.0
            else:
                flattened[f"{name}_area_sqm"] = 0.0
            flattened[f"{name}_ratio"] = 0.0
            continue

        # Use pre-projected children if available, otherwise project per component
        if children_projected is not None:
            matching_projected = children_projected.loc[matching_features.index]
        else:
            matching_projected = gpd.GeoSeries(matching_features.geometry.values, crs=API_CRS).to_crs(projected_crs)

        total_intersection_area = 0.0
        max_confidence = None
        for idx, projected_geom in enumerate(matching_projected):
            if projected_geom is not None and projected_geom.intersects(parent_projected):
                intersection = projected_geom.intersection(parent_projected)
                total_intersection_area += intersection.area
                feat = matching_features.iloc[idx]
                if hasattr(feat, "confidence") and feat.confidence is not None:
                    if max_confidence is None or feat.confidence > max_confidence:
                        max_confidence = feat.confidence

        if total_intersection_area > 0:
            flattened[f"{name}_present"] = TRUE_STRING
            if country in IMPERIAL_COUNTRIES:
                flattened[f"{name}_area_sqft"] = total_intersection_area * SQUARED_METERS_TO_SQUARED_FEET
            else:
                flattened[f"{name}_area_sqm"] = total_intersection_area
            flattened[f"{name}_ratio"] = total_intersection_area / parent_area
            if max_confidence is not None:
                flattened[f"{name}_confidence"] = max_confidence
        else:
            flattened[f"{name}_present"] = FALSE_STRING
            if country in IMPERIAL_COUNTRIES:
                flattened[f"{name}_area_sqft"] = 0.0
            else:
                flattened[f"{name}_area_sqm"] = 0.0
            flattened[f"{name}_ratio"] = 0.0

    return flattened


def link_roof_instances_to_roofs(
    roof_instances_gdf: gpd.GeoDataFrame,
    roofs_gdf: gpd.GeoDataFrame,
) -> tuple:
    """
    Link roof instances from Roof Age API to their parent roof objects from Feature API.

    This function spatially matches roof instances (temporal slices with installation dates)
    to their corresponding roof polygons from the Feature API using Intersection over Union (IoU).

    The matching is bidirectional:
    - Each roof instance gets a parent_id (the roof with highest IoU above threshold)
    - Each roof gets a primary_child_roof_age_feature_id (the roof instance with highest IoU above threshold)
      plus a list of ALL matched instances ordered by IoU

    Note:
        A minimum IoU threshold (MIN_ROOF_INSTANCE_IOU_THRESHOLD = 0.005) is applied.
        Matches below this threshold are not trusted and will not be assigned as parent/primary.
        The child_roof_instances list still contains all intersecting instances for reference.

    Args:
        roof_instances_gdf: GeoDataFrame with roof instance features from Roof Age API.
                           Must have aoi_id as index or column, feature_id, and geometry.
        roofs_gdf: GeoDataFrame with roof features from Feature API.
                   Must have aoi_id as index or column, feature_id, and geometry.

    Returns:
        Tuple of (roof_instances_with_links, roofs_with_links):
            - roof_instances_with_links: Original GDF with added columns:
                - parent_id: feature_id of best matching roof (None if IoU below threshold)
                - parent_iou: IoU score with parent roof
            - roofs_with_links: Original GDF with added columns:
                - primary_child_roof_age_feature_id: feature_id of best matching instance (None if IoU below threshold)
                - primary_child_roof_age_iou: IoU score with primary instance
                - child_roof_instances: List of dicts [{feature_id, iou}, ...] ordered by IoU desc

    Example:
        >>> ri_linked, roofs_linked = link_roof_instances_to_roofs(roof_instances, roofs)
        >>> # Get primary roof instance for a roof
        >>> roof = roofs_linked.iloc[0]
        >>> print(f"Primary roof age: {roof.primary_child_roof_age_feature_id}, IoU: {roof.primary_child_roof_age_iou}")
        >>> # Get all child instances
        >>> for child in roof.child_roof_instances:
        ...     print(f"  Instance {child['feature_id']}: IoU={child['iou']:.3f}")
    """

    # Handle empty inputs
    if len(roof_instances_gdf) == 0 or len(roofs_gdf) == 0:
        # Add empty columns and return
        ri_out = roof_instances_gdf.copy()
        ri_out["parent_id"] = None
        ri_out["parent_iou"] = 0.0

        rf_out = roofs_gdf.copy()
        rf_out["primary_child_roof_age_feature_id"] = None
        rf_out["primary_child_roof_age_iou"] = 0.0
        rf_out["child_roof_instances"] = "[]"  # JSON-serialized empty list
        rf_out["child_roof_instance_count"] = 0

        return ri_out, rf_out

    # Ensure we have aoi_id as a column for grouping
    ri_gdf = roof_instances_gdf.copy()
    rf_gdf = roofs_gdf.copy()

    # Handle index vs column for aoi_id
    if ri_gdf.index.name == AOI_ID_COLUMN_NAME:
        ri_gdf = ri_gdf.reset_index()
    if rf_gdf.index.name == AOI_ID_COLUMN_NAME:
        rf_gdf = rf_gdf.reset_index()

    # Initialize output columns
    ri_gdf["parent_id"] = None
    ri_gdf["parent_iou"] = 0.0

    rf_gdf["primary_child_roof_age_feature_id"] = None
    rf_gdf["primary_child_roof_age_iou"] = 0.0
    rf_gdf["child_roof_instances"] = "[]"  # JSON-serialized empty list for parquet compatibility
    rf_gdf["child_roof_instance_count"] = 0

    # Get unique AOIs that have both roof instances and roofs
    ri_aois = set(ri_gdf[AOI_ID_COLUMN_NAME].unique())
    rf_aois = set(rf_gdf[AOI_ID_COLUMN_NAME].unique())
    common_aois = ri_aois & rf_aois

    logger.debug(f"Linking roof instances to roofs for {len(common_aois)} AOIs")

    # Process per AOI for locality
    for aoi_id in common_aois:
        # Get roof instances and roofs for this AOI
        ri_mask = ri_gdf[AOI_ID_COLUMN_NAME] == aoi_id
        rf_mask = rf_gdf[AOI_ID_COLUMN_NAME] == aoi_id

        aoi_instances = ri_gdf[ri_mask]
        aoi_roofs = rf_gdf[rf_mask]

        if len(aoi_instances) == 0 or len(aoi_roofs) == 0:
            continue

        # Build spatial index for roofs
        roof_geoms = aoi_roofs.geometry.values
        roof_tree = STRtree(roof_geoms)
        roof_df_indices = aoi_roofs.index.values
        roof_feature_ids = aoi_roofs["feature_id"].values

        # Build mapping from roof df index to position in arrays
        roof_idx_to_pos = {idx: pos for pos, idx in enumerate(roof_df_indices)}

        # For each roof, collect all matching instances with IoU
        roof_to_instances = {idx: [] for idx in roof_df_indices}

        # For each roof instance, find best matching roof
        for ri_idx, instance_row in aoi_instances.iterrows():
            instance_geom = instance_row.geometry
            instance_feature_id = instance_row["feature_id"]

            if instance_geom is None or instance_geom.is_empty:
                continue

            # Get candidate roofs (those that intersect bounding box)
            candidates = roof_tree.query(instance_geom)

            if len(candidates) == 0:
                continue

            best_iou = 0.0
            best_roof_idx = None
            best_roof_feature_id = None

            for candidate_pos in candidates:
                roof_df_idx = roof_df_indices[candidate_pos]
                roof_geom = roof_geoms[candidate_pos]
                roof_fid = roof_feature_ids[candidate_pos]

                iou = _compute_iou(instance_geom, roof_geom)

                if iou > 0:
                    # Record this match for the roof's child list
                    # Include "kind" for sorting (prioritize "roof" over "parcel")
                    instance_kind = instance_row.get("kind") if "kind" in instance_row.index else None
                    roof_to_instances[roof_df_idx].append(
                        {
                            "feature_id": instance_feature_id,
                            "iou": round(iou, 4),
                            "kind": instance_kind,
                        }
                    )

                if iou > best_iou:
                    best_iou = iou
                    best_roof_idx = roof_df_idx
                    best_roof_feature_id = roof_fid

            # Assign parent to roof instance (only if IoU meets threshold)
            if best_roof_idx is not None and best_iou >= MIN_ROOF_INSTANCE_IOU_THRESHOLD:
                ri_gdf.at[ri_idx, "parent_id"] = best_roof_feature_id
                ri_gdf.at[ri_idx, "parent_iou"] = round(best_iou, 4)

        # For each roof, sort instances by IoU and assign primary
        for roof_df_idx, instances in roof_to_instances.items():
            if len(instances) == 0:
                continue

            # Sort by kind first (prioritize "roof" over "parcel"), then by IoU descending
            # kind_priority: "roof"=0 (first), "parcel"=1, other/None=2
            def instance_sort_key(x):
                kind = x.get("kind")
                kind_priority = 0 if kind == "roof" else (1 if kind == "parcel" else 2)
                return (kind_priority, -x["iou"])

            sorted_instances = sorted(instances, key=instance_sort_key)

            # Assign to roof (JSON-serialize list for parquet compatibility)
            rf_gdf.at[roof_df_idx, "child_roof_instances"] = json.dumps(sorted_instances)
            rf_gdf.at[roof_df_idx, "child_roof_instance_count"] = len(sorted_instances)
            # Only assign primary if IoU meets threshold - below that we don't trust the match
            if sorted_instances[0]["iou"] >= MIN_ROOF_INSTANCE_IOU_THRESHOLD:
                rf_gdf.at[roof_df_idx, "primary_child_roof_age_feature_id"] = sorted_instances[0]["feature_id"]
                rf_gdf.at[roof_df_idx, "primary_child_roof_age_iou"] = sorted_instances[0]["iou"]

    # Restore aoi_id as index
    ri_gdf = ri_gdf.set_index(AOI_ID_COLUMN_NAME)
    rf_gdf = rf_gdf.set_index(AOI_ID_COLUMN_NAME)

    logger.debug(
        f"Linked {(ri_gdf['parent_id'].notna()).sum()} roof instances to parent roofs, "
        f"{(rf_gdf['primary_child_roof_age_feature_id'].notna()).sum()} roofs have primary instances"
    )

    return ri_gdf, rf_gdf


def link_roofs_to_buildings(
    roofs_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    min_iou_threshold: float = MIN_ROOF_INSTANCE_IOU_THRESHOLD,
) -> tuple:
    """
    Link roof features to building features using spatial IoU matching.

    This function spatially matches roofs from the Feature API to their parent
    buildings using Intersection over Union (IoU). The algorithm is identical to
    link_roof_instances_to_roofs() but with different entity types.

    The matching is bidirectional:
    - Each roof gets a parent_building_id (the building with highest IoU above threshold)
    - Each building gets a primary_child_roof_id (the roof with highest IoU above threshold)
      plus a list of ALL matched roofs ordered by IoU

    Note:
        Buildings should already be filtered to BUILDING_NEW_ID class by the caller.
        The API's parentId field on roofs points to deprecated Building class, not the
        new Building class, so spatial matching is required.

    Args:
        roofs_gdf: GeoDataFrame with roof features from Feature API.
                   Must have aoi_id as index or column, feature_id, and geometry.
        buildings_gdf: GeoDataFrame with building features from Feature API.
                       Must have aoi_id as index or column, feature_id, and geometry.
        min_iou_threshold: Minimum IoU required to assign parent/primary (default: MIN_ROOF_INSTANCE_IOU_THRESHOLD)

    Returns:
        Tuple of (roofs_with_links, buildings_with_links):
            - roofs_with_links: Original GDF with added columns:
                - parent_building_id: feature_id of best matching building (None if IoU below threshold)
                - parent_building_iou: IoU score with parent building
            - buildings_with_links: Original GDF with added columns:
                - primary_child_roof_id: feature_id of best matching roof (None if IoU below threshold)
                - primary_child_roof_iou: IoU score with primary roof
                - child_roofs: List of dicts [{feature_id, iou}, ...] ordered by IoU desc
                - child_roof_count: Number of matched child roofs
    """

    # Handle empty inputs
    if len(roofs_gdf) == 0 or len(buildings_gdf) == 0:
        # Add empty columns and return
        rf_out = roofs_gdf.copy()
        rf_out["parent_building_id"] = None
        rf_out["parent_building_iou"] = 0.0

        bldg_out = buildings_gdf.copy()
        bldg_out["primary_child_roof_id"] = None
        bldg_out["primary_child_roof_iou"] = 0.0
        bldg_out["child_roofs"] = "[]"  # JSON-serialized empty list
        bldg_out["child_roof_count"] = 0

        return rf_out, bldg_out

    # Ensure we have aoi_id as a column for grouping
    rf_gdf = roofs_gdf.copy()
    bldg_gdf = buildings_gdf.copy()

    # Handle index vs column for aoi_id
    if rf_gdf.index.name == AOI_ID_COLUMN_NAME:
        rf_gdf = rf_gdf.reset_index()
    if bldg_gdf.index.name == AOI_ID_COLUMN_NAME:
        bldg_gdf = bldg_gdf.reset_index()

    # Initialize output columns
    rf_gdf["parent_building_id"] = None
    rf_gdf["parent_building_iou"] = 0.0

    bldg_gdf["primary_child_roof_id"] = None
    bldg_gdf["primary_child_roof_iou"] = 0.0
    bldg_gdf["child_roofs"] = "[]"  # JSON-serialized empty list for parquet compatibility
    bldg_gdf["child_roof_count"] = 0

    # Get unique AOIs that have both roofs and buildings
    rf_aois = set(rf_gdf[AOI_ID_COLUMN_NAME].unique())
    bldg_aois = set(bldg_gdf[AOI_ID_COLUMN_NAME].unique())
    common_aois = rf_aois & bldg_aois

    logger.debug(f"Linking roofs to buildings for {len(common_aois)} AOIs")

    # Process per AOI for locality
    for aoi_id in common_aois:
        # Get roofs and buildings for this AOI
        rf_mask = rf_gdf[AOI_ID_COLUMN_NAME] == aoi_id
        bldg_mask = bldg_gdf[AOI_ID_COLUMN_NAME] == aoi_id

        aoi_roofs = rf_gdf[rf_mask]
        aoi_buildings = bldg_gdf[bldg_mask]

        if len(aoi_roofs) == 0 or len(aoi_buildings) == 0:
            continue

        # Build spatial index for buildings
        bldg_geoms = aoi_buildings.geometry.values
        bldg_tree = STRtree(bldg_geoms)
        bldg_df_indices = aoi_buildings.index.values
        bldg_feature_ids = aoi_buildings["feature_id"].values

        # For each building, collect all matching roofs with IoU
        bldg_to_roofs = {idx: [] for idx in bldg_df_indices}

        # For each roof, find best matching building
        for rf_idx, roof_row in aoi_roofs.iterrows():
            roof_geom = roof_row.geometry
            roof_feature_id = roof_row["feature_id"]

            if roof_geom is None or roof_geom.is_empty:
                continue

            # Get candidate buildings (those that intersect bounding box)
            candidates = bldg_tree.query(roof_geom)

            if len(candidates) == 0:
                continue

            best_iou = 0.0
            best_bldg_idx = None
            best_bldg_feature_id = None

            for candidate_pos in candidates:
                bldg_df_idx = bldg_df_indices[candidate_pos]
                bldg_geom = bldg_geoms[candidate_pos]
                bldg_fid = bldg_feature_ids[candidate_pos]

                iou = _compute_iou(roof_geom, bldg_geom)

                if iou > 0:
                    # Record this match for the building's child list
                    bldg_to_roofs[bldg_df_idx].append(
                        {
                            "feature_id": roof_feature_id,
                            "iou": round(iou, 4),
                        }
                    )

                if iou > best_iou:
                    best_iou = iou
                    best_bldg_idx = bldg_df_idx
                    best_bldg_feature_id = bldg_fid

            # Assign parent to roof (only if IoU meets threshold)
            if best_bldg_idx is not None and best_iou >= min_iou_threshold:
                rf_gdf.at[rf_idx, "parent_building_id"] = best_bldg_feature_id
                rf_gdf.at[rf_idx, "parent_building_iou"] = round(best_iou, 4)

        # For each building, sort roofs by IoU and assign primary
        for bldg_df_idx, roofs in bldg_to_roofs.items():
            if len(roofs) == 0:
                continue

            # Sort by IoU descending
            sorted_roofs = sorted(roofs, key=lambda x: -x["iou"])

            # Assign to building (JSON-serialize list for parquet compatibility)
            bldg_gdf.at[bldg_df_idx, "child_roofs"] = json.dumps(sorted_roofs)
            bldg_gdf.at[bldg_df_idx, "child_roof_count"] = len(sorted_roofs)
            # Only assign primary if IoU meets threshold
            if sorted_roofs[0]["iou"] >= min_iou_threshold:
                bldg_gdf.at[bldg_df_idx, "primary_child_roof_id"] = sorted_roofs[0]["feature_id"]
                bldg_gdf.at[bldg_df_idx, "primary_child_roof_iou"] = sorted_roofs[0]["iou"]

    # Restore aoi_id as index
    rf_gdf = rf_gdf.set_index(AOI_ID_COLUMN_NAME)
    bldg_gdf = bldg_gdf.set_index(AOI_ID_COLUMN_NAME)

    logger.debug(
        f"Linked {(rf_gdf['parent_building_id'].notna()).sum()} roofs to parent buildings, "
        f"{(bldg_gdf['primary_child_roof_id'].notna()).sum()} buildings have primary roofs"
    )

    return rf_gdf, bldg_gdf


def _extract_rsi_from_feature(feature) -> dict:
    """Extract RSI fields from a feature row, returning a dict or empty dict."""
    rsi_raw = (
        feature.get("roof_spotlight_index")
        if hasattr(feature, "get")
        else getattr(feature, "roof_spotlight_index", None)
    )
    rsi_data = _parse_include_param(rsi_raw)
    if not rsi_data:
        return {}
    result = {}
    if "value" in rsi_data:
        result["roof_spotlight_index"] = rsi_data["value"]
    if "confidence" in rsi_data:
        result["roof_spotlight_index_confidence"] = rsi_data["confidence"]
    if "modelVersion" in rsi_data:
        result["roof_spotlight_index_model_version"] = rsi_data["modelVersion"]
    return result


def build_parent_lookup(features_gdf: gpd.GeoDataFrame) -> dict:
    """Build a feature_id → row dict for fast parent_id chain traversal."""
    if features_gdf is None or len(features_gdf) == 0 or "feature_id" not in features_gdf.columns:
        return {}
    mask = features_gdf["feature_id"].notna()
    filtered = features_gdf[mask]
    return {fid: filtered.iloc[i] for i, fid in enumerate(filtered["feature_id"].values)}


def resolve_footprint_rsi(
    feature,
    features_gdf: gpd.GeoDataFrame = None,
    parent_lookup: dict = None,
) -> dict:
    """
    Resolve the 'best' RSI for a feature by checking the feature itself then
    traversing its parent_id chain to find Building Lifecycle RSI.

    The API calculates RSI on the roof polygon normally, but switches to the
    building_lifecycle polygon when structural damage is present. This function
    returns RSI from whichever feature has it.

    Callers are responsible for IoU-based Roof → Building(New) linking.
    Pass a Building(New) row (not a Roof row) when doing BL fallback so the
    traversal is a direct 1-hop: Building(New) → Building Lifecycle.

    Args:
        feature: A feature row (pd.Series or dict) with parent_id and roof_spotlight_index.
            For BL fallback, pass the IoU-linked Building(New) row, not the Roof row.
        features_gdf: The full features GeoDataFrame for parent_id lookups.
            Used to build parent_lookup if not provided.
        parent_lookup: Pre-built feature_id → row dict for fast traversal.
            Pass this when calling in a loop to avoid rebuilding per call.

    Returns:
        dict with roof_spotlight_index, roof_spotlight_index_confidence, and
        optionally roof_spotlight_index_model_version,
        or empty dict if no RSI found anywhere in the chain.
    """
    # Check if the feature itself has RSI
    rsi = _extract_rsi_from_feature(feature)
    if rsi:
        return rsi

    # Build lookup if not provided
    if parent_lookup is None:
        parent_lookup = build_parent_lookup(features_gdf)
    if not parent_lookup:
        return {}

    # Traverse parent_id chain to find the building_lifecycle
    visited = set()
    pid = feature.get("parent_id") if hasattr(feature, "get") else getattr(feature, "parent_id", None)
    while pd.notna(pid) and pid not in visited:
        visited.add(pid)
        parent = parent_lookup.get(pid)
        if parent is None:
            break
        if parent.get("class_id") == BUILDING_LIFECYCLE_ID:
            return _extract_rsi_from_feature(parent)
        pid = parent.get("parent_id") if hasattr(parent, "get") else getattr(parent, "parent_id", None)

    return {}


def feature_attributes(
    features_gdf: gpd.GeoDataFrame,
    classes_df: pd.DataFrame,
    country: str,
    parcel_geom: Union[MultiPolygon, Polygon, None],
    primary_decision: str,
    primary_lat: float = None,
    primary_lon: float = None,
    geometry_projected_col: str = None,
    projected_crs: str = None,
) -> dict:
    """
    Flatten features for a parcel into a flat dictionary.

    Args:
        features_gdf: Features for a parcel
        classes_df: Class name and ID lookup (index of the dataframe) to include.
        country: The country code for map projections and units.
        parcel_geom: The geometry for the parcel, or None if no parcel geometry is known.
        primary_decision: "largest_intersection" default is just the largest feature by area intersected with Query AOI. "nearest" finds the nearest primary object to the provided coordinates, preferring objects with high confidence if present.
        primary_lat: Latitude of centroid to denote primary feature (e.g. primary building location).
        primary_lon: Longitude of centroid to denote primary feature (e.g. primary building location).
        geometry_projected_col: Optional name of a column containing pre-projected geometries
                               for performance optimization in distance-based primary selection.
        projected_crs: CRS of the pre-projected geometries (required if geometry_projected_col provided).

    Returns: Flat dictionary

    """
    mu = MeasurementUnits(country)
    area_units = mu.area_units()

    # Add present, object count, area, and confidence for all used feature classes
    parcel = {}

    # Pre-select primary roof for two purposes:
    # 1. Derive IoU-linked roof instance ID for primary roof instance selection
    # 2. Reuse when processing roofs in the loop (avoid redundant select_primary call)
    _primary_roof = None
    _primary_roof_child_ri_id = None
    roof_features = features_gdf[features_gdf.class_id == ROOF_ID]
    if len(roof_features) > 0:
        _primary_roof = select_primary(
            roof_features,
            method=primary_decision,
            area_col="clipped_area_sqm",
            secondary_area_col="unclipped_area_sqm",
            target_lat=primary_lat,
            target_lon=primary_lon,
            confidence_col="confidence",
            high_confidence_threshold=PRIMARY_FEATURE_HIGH_CONF_THRESH,
            geometry_col="geometry_feature",
            geometry_projected_col=geometry_projected_col,
            projected_crs=projected_crs,
        )
        # Get IoU-linked roof instance from primary roof
        if (
            _primary_roof is not None
            and "primary_child_roof_age_feature_id" in _primary_roof.index
            and pd.notna(_primary_roof.primary_child_roof_age_feature_id)
        ):
            _primary_roof_child_ri_id = _primary_roof.primary_child_roof_age_feature_id

    # Build parent lookup once for all RSI resolution in this parcel
    _parent_lookup = build_parent_lookup(features_gdf) if len(features_gdf) > 0 else {}

    # Pre-compute IoU-based Roof → Building(New) linkage for BL RSI fallback.
    # Roof's API parent_id points to Building(Deprecated) which is filtered out;
    # the correct path to BL is Roof →(IoU)→ Building(New) →(parent_id)→ BL.
    _roof_to_building = {}
    if len(roof_features) > 0:
        bn_features = features_gdf[features_gdf.class_id == BUILDING_NEW_ID]
        if len(bn_features) > 0:
            # features_gdf may have geometry_feature/geometry_aoi columns (post-spatial-join);
            # link_roofs_to_buildings accesses row.geometry via iterrows which requires a
            # column named "geometry". Rename geometry_feature → geometry before passing.
            if "geometry_feature" in roof_features.columns:
                roofs_for_link = roof_features.drop(columns=["geometry_aoi"], errors="ignore").rename(
                    columns={"geometry_feature": "geometry"}
                )
                bns_for_link = bn_features.drop(columns=["geometry_aoi"], errors="ignore").rename(
                    columns={"geometry_feature": "geometry"}
                )
            else:
                roofs_for_link = roof_features
                bns_for_link = bn_features
            roofs_linked_parcel, _ = link_roofs_to_buildings(roofs_for_link, bns_for_link)
            _roof_to_building = {
                fid: bid
                for fid, bid in zip(
                    roofs_linked_parcel["feature_id"].values,
                    roofs_linked_parcel["parent_building_id"].values,
                )
                if pd.notna(bid)
            }

    # Pre-select primary building lifecycle for reuse in the class loop
    _primary_bl = None
    bl_features = features_gdf[features_gdf.class_id == BUILDING_LIFECYCLE_ID]
    if len(bl_features) > 0:
        _primary_bl = select_primary(
            bl_features,
            method=primary_decision,
            area_col="clipped_area_sqm",
            secondary_area_col="unclipped_area_sqm",
            target_lat=primary_lat,
            target_lon=primary_lon,
            confidence_col="confidence",
            high_confidence_threshold=PRIMARY_FEATURE_HIGH_CONF_THRESH,
            geometry_col="geometry_feature",
            geometry_projected_col=geometry_projected_col,
            projected_crs=projected_crs,
        )

    for class_id, name in classes_df.description.items():
        name = name.lower().replace(" ", "_")
        class_features_gdf = features_gdf[features_gdf.class_id == class_id]

        # For roof instances, filter to only "roof" kind for count/area aggregations
        # "parcel" kind features are property boundaries, not actual roof instances
        if class_id == ROOF_INSTANCE_CLASS_ID and "kind" in class_features_gdf.columns:
            roof_kind_features = class_features_gdf[class_features_gdf["kind"] == "roof"]
        else:
            roof_kind_features = class_features_gdf

        # Add attributes that apply to all feature classes
        # TODO: This sets a column to "N" even if it's not possible to return it with the query (e.g. alpha/beta attribute permissions, or version issues). Need to filter out columns that pertain to this. Need to parse "availability" column in classes_df and determine what system version this row is.
        # For roof instances, use filtered features (roof kind only) for count
        features_for_count = roof_kind_features if class_id == ROOF_INSTANCE_CLASS_ID else class_features_gdf
        parcel[f"{name}_present"] = TRUE_STRING if len(features_for_count) > 0 else FALSE_STRING
        parcel[f"{name}_count"] = len(features_for_count)

        # Roof instances only have area (not clipped/unclipped) and trust_score (not confidence)
        if class_id == ROOF_INSTANCE_CLASS_ID:
            # Use filtered features (roof kind only) for area aggregation
            if country in IMPERIAL_COUNTRIES:
                parcel[f"{name}_total_area_sqft"] = (
                    roof_kind_features.area_sqft.sum() if "area_sqft" in roof_kind_features.columns else 0.0
                )
            else:
                parcel[f"{name}_total_area_sqm"] = (
                    roof_kind_features.area_sqm.sum() if "area_sqm" in roof_kind_features.columns else 0.0
                )
        else:
            # Standard Feature API classes have clipped/unclipped areas and confidence
            if country in IMPERIAL_COUNTRIES:
                parcel[f"{name}_total_area_sqft"] = (
                    class_features_gdf.area_sqft.sum() if "area_sqft" in class_features_gdf.columns else 0.0
                )
                parcel[f"{name}_total_clipped_area_sqft"] = (
                    round(class_features_gdf.clipped_area_sqft.sum(), 1)
                    if "clipped_area_sqft" in class_features_gdf.columns
                    else 0.0
                )
                parcel[f"{name}_total_unclipped_area_sqft"] = (
                    round(class_features_gdf.unclipped_area_sqft.sum(), 1)
                    if "unclipped_area_sqft" in class_features_gdf.columns
                    else 0.0
                )
            else:
                parcel[f"{name}_total_area_sqm"] = (
                    class_features_gdf.area_sqm.sum() if "area_sqm" in class_features_gdf.columns else 0.0
                )
                parcel[f"{name}_total_clipped_area_sqm"] = (
                    round(class_features_gdf.clipped_area_sqm.sum(), 1)
                    if "clipped_area_sqm" in class_features_gdf.columns
                    else 0.0
                )
                parcel[f"{name}_total_unclipped_area_sqm"] = (
                    round(class_features_gdf.unclipped_area_sqm.sum(), 1)
                    if "unclipped_area_sqm" in class_features_gdf.columns
                    else 0.0
                )
            if len(class_features_gdf) > 0 and "confidence" in class_features_gdf.columns:
                parcel[f"{name}_confidence"] = 1 - (1 - class_features_gdf.confidence).prod()
            else:
                parcel[f"{name}_confidence"] = None

        if class_id in BUILDING_STYLE_CLASS_IDS:
            col = "multiparcel_feature"
            if col in class_features_gdf.columns:
                parcel[f"{name}_{col}_count"] = len(class_features_gdf.query(f"{col} == True"))

        # Select and produce results for the primary feature of each feature class
        if class_id in CLASSES_WITH_PRIMARY_FEATURE:
            # Roof instances have different columns than standard Feature API classes
            is_roof_instance = class_id == ROOF_INSTANCE_CLASS_ID

            if len(class_features_gdf) == 0:
                # Fill values if there are no features
                # Roof instances only have area (not clipped/unclipped) and no confidence
                if is_roof_instance:
                    parcel[f"primary_{name}_area_{area_units}"] = 0.0
                else:
                    parcel[f"primary_{name}_area_{area_units}"] = 0.0
                    parcel[f"primary_{name}_clipped_area_{area_units}"] = 0.0
                    parcel[f"primary_{name}_unclipped_area_{area_units}"] = 0.0
                    parcel[f"primary_{name}_confidence"] = None
                continue

            # Select primary feature using shared selection logic
            # Note: For Feature API data, geometry_feature is the feature's own geometry
            # (vs geometry which may be the AOI geometry after merging)
            # Roof instances use area_sqm and trust_score instead of clipped_area_sqm and confidence
            if is_roof_instance:
                primary_feature = None

                # Derive primary roof instance from primary roof's IoU-linked child
                if _primary_roof_child_ri_id is not None:
                    linked_ri_rows = class_features_gdf[class_features_gdf.feature_id == _primary_roof_child_ri_id]
                    if len(linked_ri_rows) > 0:
                        primary_feature = linked_ri_rows.iloc[0]

                # Fallback to independent selection if no IoU link available
                if primary_feature is None:
                    # Prioritize "roof" kind over "parcel" kind for primary selection
                    # Use roof_kind_features if available, otherwise fall back to all features
                    # (roof_kind_features was computed earlier in this loop iteration)
                    features_for_selection = roof_kind_features if len(roof_kind_features) > 0 else class_features_gdf

                    if len(features_for_selection) > 0:
                        primary_feature = select_primary(
                            features_for_selection,
                            method=primary_decision,
                            area_col="area_sqm",
                            secondary_area_col=None,
                            target_lat=primary_lat,
                            target_lon=primary_lon,
                            confidence_col=("trust_score" if "trust_score" in features_for_selection.columns else None),
                            high_confidence_threshold=PRIMARY_FEATURE_HIGH_CONF_THRESH,
                            geometry_col=(
                                "geometry_feature"
                                if "geometry_feature" in features_for_selection.columns
                                else "geometry"
                            ),
                            geometry_projected_col=geometry_projected_col,
                            projected_crs=projected_crs,
                        )

                # Roof instances only have area (not clipped/unclipped)
                if primary_feature is None:
                    # No primary roof instance found - set defaults and skip attribute extraction
                    parcel[f"primary_{name}_area_{area_units}"] = 0.0
                    continue

                if country in IMPERIAL_COUNTRIES:
                    parcel[f"primary_{name}_area_sqft"] = (
                        round(primary_feature.area_sqft, 1)
                        if hasattr(primary_feature, "area_sqft") and primary_feature.area_sqft is not None
                        else 0.0
                    )
                else:
                    parcel[f"primary_{name}_area_sqm"] = (
                        round(primary_feature.area_sqm, 1)
                        if hasattr(primary_feature, "area_sqm") and primary_feature.area_sqm is not None
                        else 0.0
                    )
            else:
                # Reuse pre-selected primaries (avoid redundant select_primary calls)
                if class_id == ROOF_ID and _primary_roof is not None:
                    primary_feature = _primary_roof
                elif class_id == BUILDING_LIFECYCLE_ID and _primary_bl is not None:
                    primary_feature = _primary_bl
                else:
                    primary_feature = select_primary(
                        class_features_gdf,
                        method=primary_decision,
                        area_col="clipped_area_sqm",
                        secondary_area_col="unclipped_area_sqm",
                        target_lat=primary_lat,
                        target_lon=primary_lon,
                        confidence_col="confidence",
                        high_confidence_threshold=PRIMARY_FEATURE_HIGH_CONF_THRESH,
                        geometry_col="geometry_feature",
                        geometry_projected_col=geometry_projected_col,
                        projected_crs=projected_crs,
                    )
                if country in IMPERIAL_COUNTRIES:
                    parcel[f"primary_{name}_clipped_area_sqft"] = round(primary_feature.clipped_area_sqft, 1)
                    parcel[f"primary_{name}_unclipped_area_sqft"] = round(primary_feature.unclipped_area_sqft, 1)
                else:
                    parcel[f"primary_{name}_clipped_area_sqm"] = round(primary_feature.clipped_area_sqm, 1)
                    parcel[f"primary_{name}_unclipped_area_sqm"] = round(primary_feature.unclipped_area_sqm, 1)
                parcel[f"primary_{name}_confidence"] = primary_feature.confidence
                if class_id in BUILDING_STYLE_CLASS_IDS:
                    parcel[f"primary_{name}_fidelity"] = primary_feature.fidelity

            # Add roof and building attributes
            if class_id in BUILDING_STYLE_CLASS_IDS:
                col = "multiparcel_feature"
                if col in primary_feature:
                    parcel[f"primary_{name}_{col}"] = TRUE_STRING if primary_feature[col] else FALSE_STRING
                if class_id == ROOF_ID:
                    # Get non-roof features as children for clipped roof recalculation
                    geom_col = "geometry_feature" if "geometry_feature" in features_gdf.columns else "geometry"
                    non_roof = features_gdf[features_gdf.class_id != ROOF_ID]
                    if len(non_roof) > 0 and geom_col in non_roof.columns:
                        child_feats = gpd.GeoDataFrame(non_roof, geometry=geom_col, crs=API_CRS)
                        if child_feats.geometry.name != "geometry":
                            child_feats = child_feats.rename_geometry("geometry")
                    else:
                        child_feats = gpd.GeoDataFrame(
                            {
                                "class_id": pd.Series(dtype=str),
                                "geometry": pd.Series(dtype=object),
                            },
                            geometry="geometry",
                        )
                    primary_attributes = flatten_roof_attributes(
                        [primary_feature],
                        country=country,
                        child_features=child_feats,
                    )
                    primary_attributes["feature_id"] = primary_feature.feature_id
                elif class_id == BUILDING_NEW_ID:
                    primary_attributes = flatten_building_attributes([primary_feature], country=country)
                elif class_id == ROOF_INSTANCE_CLASS_ID:
                    # Roof instances have different attributes than Feature API classes
                    primary_attributes = flatten_roof_instance_attributes(primary_feature, country=country, prefix="")
                    primary_attributes["feature_id"] = primary_feature.feature_id
                else:
                    primary_attributes = {}

                for key, val in primary_attributes.items():
                    parcel[f"primary_{name}_" + str(key)] = val
            if class_id == BUILDING_LIFECYCLE_ID:
                # Provide the confidence values for each damage rating class for the primary building lifecycle feature
                primary_attributes = flatten_building_lifecycle_damage_attributes([primary_feature])
                for key, val in primary_attributes.items():
                    parcel[f"primary_{name}_" + str(key)] = val

            if class_id == BUILDING_LIFECYCLE_ID:
                # Add aggregated damage across whole parcel, weighted by building lifecycle area
                # TODO: Finish this.
                pass

    # Resolve best RSI for the primary roof.
    # Check the roof's own RSI first; if absent (structural damage case), traverse
    # IoU-linked Building(New) → Building Lifecycle via parent_id.
    _fp_rsi = {}
    if _primary_roof is not None:
        resolved_rsi = _extract_rsi_from_feature(_primary_roof)
        if not resolved_rsi:
            bn_id = _roof_to_building.get(
                _primary_roof.get("feature_id") if hasattr(_primary_roof, "get") else getattr(_primary_roof, "feature_id", None)
            )
            if bn_id:
                bn_row = _parent_lookup.get(bn_id)
                if bn_row is not None:
                    resolved_rsi = resolve_footprint_rsi(bn_row, parent_lookup=_parent_lookup)
        if resolved_rsi:
            _fp_rsi["primary_roof_spotlight_index"] = resolved_rsi.get("roof_spotlight_index")
            _fp_rsi["primary_roof_spotlight_index_confidence"] = resolved_rsi.get("roof_spotlight_index_confidence")
    parcel.update(_fp_rsi)
    # Remove the raw flattened RSI columns — primary_roof_spotlight_index supersedes them
    parcel.pop("primary_roof_roof_spotlight_index", None)
    parcel.pop("primary_roof_roof_spotlight_index_confidence", None)

    # Min/max/area-weighted-mean RSI across all roofs in the parcel.
    # Uses resolved "best" RSI per roof (roof's own first, BL fallback).
    if len(roof_features) > 0:
        rsi_vals = []
        rsi_areas = []
        area_col = (
            "clipped_area_sqft"
            if "clipped_area_sqft" in roof_features.columns
            else "clipped_area_sqm" if "clipped_area_sqm" in roof_features.columns else None
        )
        for rf in roof_features.to_dict("records"):
            resolved_rsi = _extract_rsi_from_feature(rf)
            if not resolved_rsi:
                bn_id = _roof_to_building.get(rf.get("feature_id"))
                if bn_id:
                    bn_row = _parent_lookup.get(bn_id)
                    if bn_row is not None:
                        resolved_rsi = resolve_footprint_rsi(bn_row, parent_lookup=_parent_lookup)
            rsi = resolved_rsi.get("roof_spotlight_index") if resolved_rsi else None
            if rsi is not None:
                rsi_vals.append(rsi)
                area = rf.get(area_col) if area_col else None
                area = area if area is not None and pd.notna(area) else 0
                rsi_areas.append((rsi, area))
        if rsi_vals:
            parcel["roof_spotlight_index_min"] = min(rsi_vals)
            parcel["roof_spotlight_index_max"] = max(rsi_vals)
            total_area = sum(a for _, a in rsi_areas)
            if total_area > 0:
                parcel["roof_spotlight_index_area_weighted_mean"] = round(
                    sum(r * a for r, a in rsi_areas) / total_area, 1
                )

    return parcel


def extract_building_features(
    parcels_gdf: gpd.GeoDataFrame,
    features_gdf: gpd.GeoDataFrame,
    country: str,
) -> gpd.GeoDataFrame:
    """
    Extract building-related features and their attributes to create a building-level export. Note that this gets all building like features, not strictly buildings.

    Args:
        parcels_gdf: GeoDataFrame with AOI information
        features_gdf: GeoDataFrame with all features
        country: Country code for units

    Returns:
        GeoDataFrame with one row per building style feature, including geometry and attributes
    """
    if features_gdf is None or len(features_gdf) == 0:
        return gpd.GeoDataFrame()

    # Filter for building-style features only
    building_gdf = features_gdf[features_gdf.class_id.isin(BUILDING_STYLE_CLASS_IDS)].copy()

    if len(building_gdf) == 0:
        return gpd.GeoDataFrame()

    # Create a list to store processed building records
    building_records = []

    # Process each building feature
    for idx, building in building_gdf.iterrows():
        # Get AOI ID
        aoi_id = building.name if hasattr(building, "name") else idx

        # Start with basic feature info
        building_record = {
            AOI_ID_COLUMN_NAME: aoi_id,
            "feature_id": building.feature_id,
            "class_id": building.class_id,
            "description": building.description,
            "confidence": building.confidence,
            "fidelity": building.fidelity if hasattr(building, "fidelity") else None,
            "area_sqm": building.area_sqm if hasattr(building, "area_sqm") else None,
            "clipped_area_sqm": (building.clipped_area_sqm if hasattr(building, "clipped_area_sqm") else None),
            "unclipped_area_sqm": (building.unclipped_area_sqm if hasattr(building, "unclipped_area_sqm") else None),
            "area_sqft": building.area_sqft if hasattr(building, "area_sqft") else None,
            "clipped_area_sqft": (building.clipped_area_sqft if hasattr(building, "clipped_area_sqft") else None),
            "unclipped_area_sqft": (building.unclipped_area_sqft if hasattr(building, "unclipped_area_sqft") else None),
            "survey_date": (building.survey_date if hasattr(building, "survey_date") else None),
            "mesh_date": building.mesh_date if hasattr(building, "mesh_date") else None,
            "geometry": building.geometry,
            "is_primary": (building.is_primary if hasattr(building, "is_primary") else False),
        }
        if hasattr(building, "parent_id"):
            building_record["parent_id"] = building.parent_id

        # Preserve damage field for building lifecycle features
        if hasattr(building, "damage") and building.damage is not None:
            building_record["damage"] = building.damage

        # Flatten attributes based on the class type
        try:
            if building.class_id == ROOF_ID:
                # For roof attributes, don't wrap in a list if it's already a list
                # This is the key fix - roof attributes should be processed as they are
                if isinstance(building, list):
                    flat_attrs = flatten_roof_attributes(building, country=country)
                else:
                    flat_attrs = flatten_roof_attributes([building], country=country)

                for k, v in flat_attrs.items():
                    building_record[k] = v
            elif building.class_id == BUILDING_NEW_ID:
                flat_attrs = flatten_building_attributes([building], country=country)
                for k, v in flat_attrs.items():
                    building_record[k] = v
            elif building.class_id == BUILDING_LIFECYCLE_ID:
                flat_attrs = flatten_building_lifecycle_damage_attributes([building])
                for k, v in flat_attrs.items():
                    building_record[k] = v
        except Exception as e:
            # If any issues processing attributes, log and continue
            logger.warning(f"Error processing attributes for feature {building.feature_id}: {str(e)}")

        building_records.append(building_record)

    if not building_records:
        return gpd.GeoDataFrame()

    # Create GeoDataFrame from building records
    buildings_df = gpd.GeoDataFrame(building_records, geometry="geometry", crs=API_CRS)

    # Add AOI information if available (merge on AOI ID)
    if parcels_gdf is not None:
        # Identify columns from parcels_gdf to include (exclude geometry to avoid conflicts)
        parcel_cols = [col for col in parcels_gdf.columns if col != "geometry"]
        if parcel_cols:
            # Create copy with reset index to allow merging
            parcel_info = parcels_gdf.reset_index()[parcel_cols + [AOI_ID_COLUMN_NAME]]
            # Merge with buildings dataframe
            buildings_df = buildings_df.merge(parcel_info, on=AOI_ID_COLUMN_NAME, how="left")

    return buildings_df


def parcel_rollup(
    parcels_gdf: gpd.GeoDataFrame,
    features_gdf: gpd.GeoDataFrame,
    classes_df: pd.DataFrame,
    country: str,
    primary_decision: str,
    api_metadata: list = None,
):
    """
    Summarize feature data to parcel attributes.

    Args:
        parcels_gdf: Parcels GeoDataFrame
        features_gdf: Features GeoDataFrame
        classes_df: Class name and ID lookup
        country: Country code for units.
        primary_decision: The basis on which the primary features are chosen
        api_metadata: Optional list of (metadata_df, classes_df_subset) tuples.
            Each tuple pairs an API's metadata (indexed by AOI ID, with rows only
            for successful AOIs) with the classes_df subset that API covers.
            For AOIs missing from a metadata_df, columns for those classes are set
            to null — indicating "no observational data" rather than "checked and
            found nothing."

    Returns:
        Parcel rollup DataFrame
    """
    mu = MeasurementUnits(country)
    area_units = mu.area_units()

    # Pre-compute which classes should be nulled per AOI based on API metadata.
    # For each (metadata_df, classes_subset) pair, AOIs missing from metadata_df
    # had that API fail — their columns for those classes should be null.
    null_classes_by_aoi = {}
    _class_baseline_columns = {}
    if api_metadata:
        all_aoi_ids = set(parcels_gdf.index)
        for meta_df, meta_classes in api_metadata:
            successful_aois = set(meta_df.index) if len(meta_df) > 0 else set()
            failed_aois = all_aoi_ids - successful_aois
            if failed_aois:
                class_ids = set(meta_classes.index)
                for aoi_id in failed_aois:
                    null_classes_by_aoi.setdefault(aoi_id, set()).update(class_ids)
        # Only compute baseline columns if at least one AOI needs nullification
        if null_classes_by_aoi:
            # Discover exact baseline column names per class by running
            # feature_attributes() with empty data. This avoids prefix matching
            # and uses feature_attributes() itself as the source of truth.
            area_name = f"area_{area_units}"
            empty_gdf = gpd.GeoDataFrame(
                [],
                columns=[
                    "class_id",
                    area_name,
                    f"clipped_{area_name}",
                    f"unclipped_{area_name}",
                ],
            )
            for class_id in classes_df.index:
                single_class = classes_df.loc[[class_id]]
                baseline = feature_attributes(
                    empty_gdf,
                    single_class,
                    country=country,
                    parcel_geom=None,
                    primary_decision=primary_decision,
                )
                _class_baseline_columns[class_id] = set(baseline.keys())

    assert parcels_gdf.index.name == AOI_ID_COLUMN_NAME

    # Handle case where features_gdf index name is not set properly
    if features_gdf.index.name != AOI_ID_COLUMN_NAME:
        if AOI_ID_COLUMN_NAME in features_gdf.columns:
            features_gdf = features_gdf.set_index(AOI_ID_COLUMN_NAME)
        else:
            raise ValueError(
                f"features_gdf index name is '{features_gdf.index.name}' but should be '{AOI_ID_COLUMN_NAME}', and column '{AOI_ID_COLUMN_NAME}' not found in dataframe columns: {list(features_gdf.columns)}"
            )

    assert features_gdf.index.name == AOI_ID_COLUMN_NAME

    # Strip deprecated classes before any rollup processing. The exporter does this
    # immediately after the API fetch, but parcel_rollup may also be called directly.
    if len(features_gdf) > 0 and "class_id" in features_gdf.columns:
        features_gdf = features_gdf[~features_gdf["class_id"].isin(DEPRECATED_CLASS_IDS)]

    if len(parcels_gdf.index.unique()) != len(parcels_gdf):
        raise Exception(
            f"AOI id index {AOI_ID_COLUMN_NAME} is NOT unique in parcels/AOI dataframe, but it should be: there are {len(parcels_gdf.index.unique())} unique AOI ids and {len(parcels_gdf)} rows in the dataframe"
        )
    # Methods that use lat/lon for primary feature selection
    uses_lat_lon = primary_decision in ("nearest", "optimal")
    if uses_lat_lon:
        merge_cols = [
            LAT_PRIMARY_COL_NAME,
            LON_PRIMARY_COL_NAME,
        ]
    else:
        merge_cols = []
    # Only include geometry if it exists (address-based queries won't have it)
    if "geometry" in parcels_gdf.columns:
        merge_cols.append("geometry")

    df = features_gdf.merge(
        parcels_gdf[merge_cols],
        left_index=True,
        right_index=True,
        suffixes=["_feature", "_aoi"],
    )

    # Pre-project feature geometries to country-appropriate CRS for distance-based primary selection
    # This avoids repeated CRS transformations inside select_primary_by_nearest()
    # Use Albers Equal Area projection for accurate distance calculations
    geometry_projected_col = None
    projected_crs = AREA_CRS[country.lower()]
    if uses_lat_lon:
        # Determine geometry column (after merge with suffixes, it may be geometry_feature)
        geom_col = "geometry_feature" if "geometry_feature" in df.columns else "geometry"
        if geom_col in df.columns:
            # Create a temporary GeoDataFrame for projection
            temp_gdf = gpd.GeoDataFrame(
                {"idx": df.index},
                geometry=df[geom_col].values,
                crs=features_gdf.crs or API_CRS,
            )
            df["_geometry_projected"] = temp_gdf.to_crs(projected_crs).geometry.values
            geometry_projected_col = "_geometry_projected"

    rollups = []
    # Loop over parcels with features in them
    for aoi_id, group in df.reset_index().groupby(AOI_ID_COLUMN_NAME):
        if uses_lat_lon:
            primary_lon = group[LON_PRIMARY_COL_NAME].unique()
            if len(primary_lon) == 1:
                primary_lon = primary_lon[0]
            else:
                raise ValueError("More than one primary longitude for this query AOI")
            primary_lat = group[LAT_PRIMARY_COL_NAME].unique()
            if len(primary_lat) == 1:
                primary_lat = primary_lat[0]
            else:
                raise ValueError("More than one primary latitude for this query AOI")
        else:
            primary_lon = None
            primary_lat = None

        if "geometry" in parcels_gdf.columns:
            parcel_geom = parcels_gdf.loc[[aoi_id]]
            assert len(parcel_geom) == 1
            parcel_geom = parcel_geom.iloc[0].geometry
        else:
            parcel_geom = None

        parcel = feature_attributes(
            group,
            classes_df,
            country=country,
            parcel_geom=parcel_geom,
            primary_decision=primary_decision,
            primary_lat=primary_lat,
            primary_lon=primary_lon,
            geometry_projected_col=geometry_projected_col,
            projected_crs=projected_crs,
        )
        parcel[AOI_ID_COLUMN_NAME] = aoi_id
        if "mesh_date" in group.columns:
            parcel["mesh_date"] = group.mesh_date.iloc[0]
        if aoi_id in null_classes_by_aoi:
            for class_id in null_classes_by_aoi[aoi_id]:
                for key in _class_baseline_columns.get(class_id, set()):
                    if key in parcel:
                        parcel[key] = None
        rollups.append(parcel)
    # Loop over parcels without features in them
    area_name = f"area_{area_units}"

    hasgeom = "geometry" in parcels_gdf.columns
    for row in parcels_gdf[~parcels_gdf.index.isin(features_gdf.index)].itertuples():
        parcel = feature_attributes(
            gpd.GeoDataFrame(
                [],
                columns=[
                    "class_id",
                    area_name,
                    f"clipped_{area_name}",
                    f"unclipped_{area_name}",
                ],
            ),
            classes_df,
            country=country,
            parcel_geom=row.geometry if hasgeom else None,
            primary_decision=primary_decision,
        )
        aoi_id = row._asdict()["Index"]
        parcel[AOI_ID_COLUMN_NAME] = aoi_id
        if aoi_id in null_classes_by_aoi:
            for class_id in null_classes_by_aoi[aoi_id]:
                for key in _class_baseline_columns.get(class_id, set()):
                    if key in parcel:
                        parcel[key] = None
        rollups.append(parcel)
    # Combine, validate and return
    rollup_df = pd.DataFrame(rollups)
    rollup_df = rollup_df.set_index(AOI_ID_COLUMN_NAME)
    if len(rollup_df) != len(parcels_gdf):
        raise RuntimeError(f"Parcel count validation error: {len(rollup_df)=} not equal to {len(parcels_gdf)=}")

    # Round any columns ending in _confidence to two decimal places (nearest percent)
    for col in rollup_df.columns:
        if col.endswith("_confidence"):
            try:
                # Convert to float first to handle integer columns
                rollup_df[col] = rollup_df[col].astype(float).round(2)
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to round column '{col}' - column description:")
                logger.error(rollup_df[col].describe())
                raise
    # Defragment DataFrame to avoid PerformanceWarning when callers add columns
    return rollup_df.copy()
