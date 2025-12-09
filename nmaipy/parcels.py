"""
Parcel/AOI Processing and Rollup Utilities

This module provides functions for:
- Creating rollup summaries of AI features at the parcel/AOI level
- Extracting building-style features for detailed export
- Linking roof instances to parent roof objects (future)

The rollup functions aggregate multiple features within each AOI into summary statistics
and select "primary" features for detailed attribute extraction.
"""
import json
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

from nmaipy import log
from nmaipy.aoi_io import read_from_file  # Re-export for backwards compatibility
from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    API_CRS,
    BUILDING_ID,
    BUILDING_LIFECYCLE_ID,
    BUILDING_NEW_ID,
    BUILDING_STYLE_CLASS_IDS,
    CLASSES_WITH_PRIMARY_FEATURE,
    IMPERIAL_COUNTRIES,
    LAT_PRIMARY_COL_NAME,
    LON_PRIMARY_COL_NAME,
    MIN_ROOF_INSTANCE_IOU_THRESHOLD,
    ROOF_AGE_TRUST_SCORE_FIELD,
    ROOF_ID,
    ROOF_INSTANCE_CLASS_ID,
    MeasurementUnits,
)
from nmaipy.feature_attributes import (
    TRUE_STRING,
    FALSE_STRING,
    flatten_building_attributes,
    flatten_building_lifecycle_damage_attributes,
    flatten_roof_attributes,
    flatten_roof_instance_attributes,
)
from nmaipy.primary_feature_selection import (
    DEFAULT_HIGH_CONFIDENCE_THRESHOLD as PRIMARY_FEATURE_HIGH_CONF_THRESH,
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
    - Each roof gets a primary_child_roof_instance_feature_id (the instance with highest IoU above threshold)
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
                - primary_child_roof_instance_feature_id: feature_id of best matching instance (None if IoU below threshold)
                - primary_child_roof_instance_iou: IoU score with primary instance
                - child_roof_instances: List of dicts [{feature_id, iou}, ...] ordered by IoU desc

    Example:
        >>> ri_linked, roofs_linked = link_roof_instances_to_roofs(roof_instances, roofs)
        >>> # Get primary roof instance for a roof
        >>> roof = roofs_linked.iloc[0]
        >>> print(f"Primary instance: {roof.primary_child_roof_instance_feature_id}, IoU: {roof.primary_child_roof_instance_iou}")
        >>> # Get all child instances
        >>> for child in roof.child_roof_instances:
        ...     print(f"  Instance {child['feature_id']}: IoU={child['iou']:.3f}")
    """
    from shapely.strtree import STRtree

    # Handle empty inputs
    if len(roof_instances_gdf) == 0 or len(roofs_gdf) == 0:
        # Add empty columns and return
        ri_out = roof_instances_gdf.copy()
        ri_out["parent_id"] = None
        ri_out["parent_iou"] = 0.0

        rf_out = roofs_gdf.copy()
        rf_out["primary_child_roof_instance_feature_id"] = None
        rf_out["primary_child_roof_instance_iou"] = 0.0
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

    rf_gdf["primary_child_roof_instance_feature_id"] = None
    rf_gdf["primary_child_roof_instance_iou"] = 0.0
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
                    roof_to_instances[roof_df_idx].append({
                        "feature_id": instance_feature_id,
                        "iou": round(iou, 4),
                        "kind": instance_kind,
                    })

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
                rf_gdf.at[roof_df_idx, "primary_child_roof_instance_feature_id"] = sorted_instances[0]["feature_id"]
                rf_gdf.at[roof_df_idx, "primary_child_roof_instance_iou"] = sorted_instances[0]["iou"]

    # Restore aoi_id as index
    ri_gdf = ri_gdf.set_index(AOI_ID_COLUMN_NAME)
    rf_gdf = rf_gdf.set_index(AOI_ID_COLUMN_NAME)

    logger.debug(
        f"Linked {(ri_gdf['parent_id'].notna()).sum()} roof instances to parent roofs, "
        f"{(rf_gdf['primary_child_roof_instance_feature_id'].notna()).sum()} roofs have primary instances"
    )

    return ri_gdf, rf_gdf


def feature_attributes(
    features_gdf: gpd.GeoDataFrame,
    classes_df: pd.DataFrame,
    country: str,
    parcel_geom: Union[MultiPolygon, Polygon, None],
    primary_decision: str,
    primary_lat: float = None,
    primary_lon: float = None,
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

    Returns: Flat dictionary

    """
    mu = MeasurementUnits(country)
    area_units = mu.area_units()

    # Add present, object count, area, and confidence for all used feature classes
    parcel = {}
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
                parcel[f"{name}_total_area_sqft"] = roof_kind_features.area_sqft.sum() if "area_sqft" in roof_kind_features.columns else 0.0
            else:
                parcel[f"{name}_total_area_sqm"] = roof_kind_features.area_sqm.sum() if "area_sqm" in roof_kind_features.columns else 0.0
        else:
            # Standard Feature API classes have clipped/unclipped areas and confidence
            if country in IMPERIAL_COUNTRIES:
                parcel[f"{name}_total_area_sqft"] = class_features_gdf.area_sqft.sum() if "area_sqft" in class_features_gdf.columns else 0.0
                parcel[f"{name}_total_clipped_area_sqft"] = round(class_features_gdf.clipped_area_sqft.sum(), 1) if "clipped_area_sqft" in class_features_gdf.columns else 0.0
                parcel[f"{name}_total_unclipped_area_sqft"] = round(class_features_gdf.unclipped_area_sqft.sum(), 1) if "unclipped_area_sqft" in class_features_gdf.columns else 0.0
            else:
                parcel[f"{name}_total_area_sqm"] = class_features_gdf.area_sqm.sum() if "area_sqm" in class_features_gdf.columns else 0.0
                parcel[f"{name}_total_clipped_area_sqm"] = round(class_features_gdf.clipped_area_sqm.sum(), 1) if "clipped_area_sqm" in class_features_gdf.columns else 0.0
                parcel[f"{name}_total_unclipped_area_sqm"] = round(class_features_gdf.unclipped_area_sqm.sum(), 1) if "unclipped_area_sqm" in class_features_gdf.columns else 0.0
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
            is_roof_instance = (class_id == ROOF_INSTANCE_CLASS_ID)

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
                # Prioritize "roof" kind over "parcel" kind for primary selection
                # Use roof_kind_features if available, otherwise fall back to all features
                # (roof_kind_features was computed earlier in this loop iteration)
                features_for_selection = roof_kind_features if len(roof_kind_features) > 0 else class_features_gdf

                primary_feature = select_primary(
                    features_for_selection,
                    method=primary_decision,
                    area_col="area_sqm",
                    secondary_area_col=None,
                    target_lat=primary_lat,
                    target_lon=primary_lon,
                    confidence_col=ROOF_AGE_TRUST_SCORE_FIELD if ROOF_AGE_TRUST_SCORE_FIELD in features_for_selection.columns else None,
                    high_confidence_threshold=PRIMARY_FEATURE_HIGH_CONF_THRESH,
                    geometry_col="geometry_feature" if "geometry_feature" in features_for_selection.columns else "geometry",
                )
                # Roof instances only have area (not clipped/unclipped)
                if country in IMPERIAL_COUNTRIES:
                    parcel[f"primary_{name}_area_sqft"] = round(primary_feature.area_sqft, 1) if hasattr(primary_feature, "area_sqft") and primary_feature.area_sqft is not None else 0.0
                else:
                    parcel[f"primary_{name}_area_sqm"] = round(primary_feature.area_sqm, 1) if hasattr(primary_feature, "area_sqm") and primary_feature.area_sqm is not None else 0.0
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
                    parcel[f"primary_{name}_{col}"] = primary_feature[col]
                if class_id == ROOF_ID:
                    primary_attributes = flatten_roof_attributes([primary_feature], country=country)
                    primary_attributes["feature_id"] = primary_feature.feature_id
                elif class_id in [BUILDING_ID, BUILDING_NEW_ID]:
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
        aoi_id = building.name if hasattr(building, 'name') else idx
        
        # Start with basic feature info
        building_record = {
            AOI_ID_COLUMN_NAME: aoi_id,
            "feature_id": building.feature_id,
            "class_id": building.class_id,
            "class_description": building.description,
            "confidence": building.confidence,
            "fidelity": building.fidelity if hasattr(building, 'fidelity') else None,
            "area_sqm": building.area_sqm if hasattr(building, 'area_sqm') else None,
            "clipped_area_sqm": building.clipped_area_sqm if hasattr(building, 'clipped_area_sqm') else None,
            "unclipped_area_sqm": building.unclipped_area_sqm if hasattr(building, 'unclipped_area_sqm') else None,
            "area_sqft": building.area_sqft if hasattr(building, 'area_sqft') else None,
            "clipped_area_sqft": building.clipped_area_sqft if hasattr(building, 'clipped_area_sqft') else None,
            "unclipped_area_sqft": building.unclipped_area_sqft if hasattr(building, 'unclipped_area_sqft') else None,
            "survey_date": building.survey_date if hasattr(building, 'survey_date') else None,
            "mesh_date": building.mesh_date if hasattr(building, 'mesh_date') else None,
            "geometry": building.geometry
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
            elif building.class_id in [BUILDING_ID, BUILDING_NEW_ID]:
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
        parcel_cols = [col for col in parcels_gdf.columns if col != 'geometry']
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
):
    """
    Summarize feature data to parcel attributes.

    Args:
        parcels_gdf: Parcels GeoDataFrame
        features_gdf: Features GeoDataFrame
        classes_df: Class name and ID lookup
        country: Country code for units.
        primary_decision: The basis on which the primary features are chosen

    Returns:
        Parcel rollup DataFrame
    """
    mu = MeasurementUnits(country)
    area_units = mu.area_units()
    assert parcels_gdf.index.name == AOI_ID_COLUMN_NAME
    
    # Handle case where features_gdf index name is not set properly
    if features_gdf.index.name != AOI_ID_COLUMN_NAME:
        if AOI_ID_COLUMN_NAME in features_gdf.columns:
            features_gdf = features_gdf.set_index(AOI_ID_COLUMN_NAME)
        else:
            raise ValueError(f"features_gdf index name is '{features_gdf.index.name}' but should be '{AOI_ID_COLUMN_NAME}', and column '{AOI_ID_COLUMN_NAME}' not found in dataframe columns: {list(features_gdf.columns)}")
    
    assert features_gdf.index.name == AOI_ID_COLUMN_NAME

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

    df = features_gdf.merge(parcels_gdf[merge_cols], left_index=True, right_index=True, suffixes=["_feature", "_aoi"])

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
        )
        parcel[AOI_ID_COLUMN_NAME] = aoi_id
        parcel["mesh_date"] = group.mesh_date.iloc[0]
        rollups.append(parcel)
    # Loop over parcels without features in them
    if country in IMPERIAL_COUNTRIES:
        area_name = f"area_{area_units}"
    else:
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
        parcel[AOI_ID_COLUMN_NAME] = row._asdict()["Index"]
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
    return rollup_df
