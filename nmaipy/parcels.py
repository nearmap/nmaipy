"""
Parcel/AOI Processing and Rollup Utilities

This module provides functions for:
- Creating rollup summaries of AI features at the parcel/AOI level
- Extracting building-style features for detailed export
- Linking roof instances to parent roof objects (future)

The rollup functions aggregate multiple features within each AOI into summary statistics
and select "primary" features for detailed attribute extraction.
"""
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


def link_roof_instances_to_roofs(
    roof_instances_gdf: gpd.GeoDataFrame,
    roofs_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Link roof instances from Roof Age API to their parent roof objects from Feature API.

    This function spatially matches roof instances (temporal slices with installation dates)
    to their corresponding roof polygons from the Feature API. This enables correlating
    roof age information with roof characteristics (material, condition, etc.).

    Args:
        roof_instances_gdf: GeoDataFrame with roof instance features from Roof Age API.
                           Must have aoi_id index and geometry column.
        roofs_gdf: GeoDataFrame with roof features from Feature API.
                   Must have aoi_id index, feature_id, and geometry columns.
        aoi_gdf: GeoDataFrame with AOI boundaries. Used to scope the spatial matching.

    Returns:
        DataFrame with columns:
            - aoi_id: AOI identifier
            - roof_instance_id: ID of the roof instance (if available)
            - parent_roof_feature_id: feature_id of the matched parent roof
            - intersection_area_sqm: Area of intersection between instance and roof
            - intersection_pct: Percentage of roof instance covered by parent roof

    Note:
        This is a stub for future implementation. The matching algorithm should:
        1. For each AOI, find all roof instances and roofs
        2. Calculate spatial intersections between instances and roofs
        3. Match each instance to the roof with maximum intersection
        4. Handle cases where instances don't match any roof (new construction, etc.)

    Example:
        >>> # Future usage
        >>> links_df = link_roof_instances_to_roofs(roof_instances, roofs, aois)
        >>> # Join to get both roof age and roof characteristics
        >>> combined = roof_instances.merge(links_df, on='aoi_id')
        >>> combined = combined.merge(roofs[['feature_id', 'material']],
        ...                           left_on='parent_roof_feature_id', right_on='feature_id')
    """
    # TODO: Implement spatial matching algorithm
    # For now, return empty DataFrame with expected schema
    logger.warning("link_roof_instances_to_roofs is not yet implemented - returning empty DataFrame")
    return pd.DataFrame(columns=[
        AOI_ID_COLUMN_NAME,
        "roof_instance_id",
        "parent_roof_feature_id",
        "intersection_area_sqm",
        "intersection_pct",
    ])


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

        # Add attributes that apply to all feature classes
        # TODO: This sets a column to "N" even if it's not possible to return it with the query (e.g. alpha/beta attribute permissions, or version issues). Need to filter out columns that pertain to this. Need to parse "availability" column in classes_df and determine what system version this row is.
        parcel[f"{name}_present"] = TRUE_STRING if len(class_features_gdf) > 0 else FALSE_STRING
        parcel[f"{name}_count"] = len(class_features_gdf)
        if country in IMPERIAL_COUNTRIES:
            parcel[f"{name}_total_area_sqft"] = class_features_gdf.area_sqft.sum()
            parcel[f"{name}_total_clipped_area_sqft"] = round(class_features_gdf.clipped_area_sqft.sum(), 1)
            parcel[f"{name}_total_unclipped_area_sqft"] = round(class_features_gdf.unclipped_area_sqft.sum(), 1)
        else:
            parcel[f"{name}_total_area_sqm"] = class_features_gdf.area_sqm.sum()
            parcel[f"{name}_total_clipped_area_sqm"] = round(class_features_gdf.clipped_area_sqm.sum(), 1)
            parcel[f"{name}_total_unclipped_area_sqm"] = round(class_features_gdf.unclipped_area_sqm.sum(), 1)
        if len(class_features_gdf) > 0:
            parcel[f"{name}_confidence"] = 1 - (1 - class_features_gdf.confidence).prod()
        else:
            parcel[f"{name}_confidence"] = None

        if class_id in BUILDING_STYLE_CLASS_IDS:
            col = "multiparcel_feature"
            if col in class_features_gdf.columns:
                parcel[f"{name}_{col}_count"] = len(class_features_gdf.query(f"{col} == True"))

        # Select and produce results for the primary feature of each feature class
        if class_id in CLASSES_WITH_PRIMARY_FEATURE:
            if len(class_features_gdf) == 0:
                # Fill values if there are no features
                parcel[f"primary_{name}_area_{area_units}"] = 0.0
                parcel[f"primary_{name}_clipped_area_{area_units}"] = 0.0
                parcel[f"primary_{name}_unclipped_area_{area_units}"] = 0.0
                parcel[f"primary_{name}_confidence"] = None
                continue

            # Select primary feature using shared selection logic
            # Note: For Feature API data, geometry_feature is the feature's own geometry
            # (vs geometry which may be the AOI geometry after merging)
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
    if primary_decision == "nearest":
        merge_cols = [
            LAT_PRIMARY_COL_NAME,
            LON_PRIMARY_COL_NAME,
            "geometry",
        ]
    else:
        merge_cols = []
        if "geometry" in parcels_gdf.columns:
            merge_cols += ["geometry"]

    df = features_gdf.merge(parcels_gdf[merge_cols], left_index=True, right_index=True, suffixes=["_feature", "_aoi"])

    rollups = []
    # Loop over parcels with features in them
    for aoi_id, group in df.reset_index().groupby(AOI_ID_COLUMN_NAME):
        if primary_decision == "nearest":
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
