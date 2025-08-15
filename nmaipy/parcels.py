import warnings
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon

from nmaipy import log
from nmaipy.constants import (
    AOI_ID_COLUMN_NAME, 
    API_CRS,
    AREA_CRS,
    BUILDING_ID,
    BUILDING_NEW_ID,
    BUILDING_LIFECYCLE_ID,
    BUILDING_UNDER_CONSTRUCTION_ID,
    BUILDING_STYLE_CLASS_IDS,
    CLASSES_WITH_PRIMARY_FEATURE,
    IMPERIAL_COUNTRIES,
    LAT_LONG_CRS,
    LAT_PRIMARY_COL_NAME,
    LON_PRIMARY_COL_NAME,
    METERS_TO_FEET,
    ROOF_ID,
    VEG_MEDHIGH_ID,
    VEG_WOODY_COMPOSITE_ID,
    CLASS_1111_YARD_DEBRIS,
    MeasurementUnits,
)

TRUE_STRING = "Y"
FALSE_STRING = "N"
PRIMARY_FEATURE_HIGH_CONF_THRESH = 0.9

# All area values are in squared metres
BUILDING_STYLE_CLASSE_IDS = [
        BUILDING_LIFECYCLE_ID,
        BUILDING_ID,
        BUILDING_NEW_ID,
        BUILDING_UNDER_CONSTRUCTION_ID,
        ROOF_ID
]


logger = log.get_logger()


def read_from_file(
    path: Path,
    drop_empty: Optional[bool] = True,
    id_column: Optional[str] = AOI_ID_COLUMN_NAME,
    source_crs: Optional[str] = LAT_LONG_CRS,
    target_crs: Optional[str] = LAT_LONG_CRS,
) -> gpd.GeoDataFrame:
    """
    Read parcel data from a file. Supported formats are:
     - CSV with geometries as WKTs
     - GPKG
     - GeoJSON
     - Parquet with geometries as WKBs

    Args:
        path: Path to file
        drop_empty: If true, rows with empty geometries will be dropped.
        id_column: Unique identifier column name. This column will be renamed to the default AOI ID columns name,
                   as used by other functions in this module.
        source_crs: CRS of the sources data - defaults to lat/long. If the source data has a CRS set, this field is
                    ignored.
        target_crs: CRS of data being returned.

    Returns: GeoDataFrame
    """
    if isinstance(path, str):
        suffix = path.split(".")[-1]
    elif isinstance(path, Path):
        suffix = path.suffix[1:]
    if suffix in ("csv", "psv", "tsv"):
        # First read without setting index to avoid failure if id_column doesn't exist
        if suffix == "csv":
            parcels_gdf = pd.read_csv(path)
        elif suffix == "psv":
            parcels_gdf = pd.read_csv(path, sep="|")
        elif suffix == "tsv":
            parcels_gdf = pd.read_csv(path, sep="\t")
            
        # Set the index only if the column exists
        if id_column in parcels_gdf.columns:
            parcels_gdf = parcels_gdf.set_index(id_column)
    elif suffix == "parquet":
        parcels_gdf = gpd.read_parquet(path)
    elif suffix in ("geojson", "gpkg"):
        parcels_gdf = gpd.read_file(path)
    else:
        raise NotImplementedError(f"Source format not supported: {suffix=}")

    if not "geometry" in parcels_gdf:
        logger.warning(f"Input file has no AOI geometries - some operations will not work.")
    else:
        if not isinstance(parcels_gdf, gpd.GeoDataFrame):
            # If from a tabular data source, try to convert to a GeoDataFrame (requires a geometry column)
            geometry = gpd.GeoSeries.from_wkt(parcels_gdf.geometry.fillna("POLYGON(EMPTY)"))
            parcels_gdf = gpd.GeoDataFrame(
                parcels_gdf,
                geometry=geometry,
                crs=source_crs,
            )
    if "geometry" in parcels_gdf:
        # Set CRS and project if data CRS is not equal to target CRS
        if parcels_gdf.crs is None:
            parcels_gdf.set_crs(source_crs)
        if parcels_gdf.crs != target_crs:
            parcels_gdf = parcels_gdf.to_crs(target_crs)

        # Drop any empty geometries
        if drop_empty:
            num_dropped = len(parcels_gdf)
            parcels_gdf = parcels_gdf.dropna(subset=["geometry"])
            parcels_gdf = parcels_gdf[~parcels_gdf.is_empty]
            parcels_gdf = parcels_gdf[parcels_gdf.is_valid]
            # For this we only check if the shape has a non-zero area, the value doesn't matter, so the warning can be
            # ignored.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.")
                parcels_gdf = parcels_gdf[parcels_gdf.area > 0]
            num_dropped -= len(parcels_gdf)
            if num_dropped > 0:
                logger.warning(f"Dropping {num_dropped} rows with empty or invalid geometries, or ones with zero area")

    if len(parcels_gdf) == 0:
        raise RuntimeError(f"No valid parcels in {path=}")

    # Check that identifier is unique
    if parcels_gdf.index.name != id_column:
        # Bump the index to a column in case it's important
        parcels_gdf = parcels_gdf.reset_index()
        if id_column not in parcels_gdf:
            logger.info(f"Missing {AOI_ID_COLUMN_NAME} column in parcel data - generating unique IDs")
            parcels_gdf.index.name = id_column  # Set a new unique ordered index for reference
        else:  # The index must already be there as a column
            logger.warning(f"Moving {AOI_ID_COLUMN_NAME} to be the index - generating unique IDs")
            parcels_gdf = parcels_gdf.set_index(id_column)
    if parcels_gdf.index.duplicated().any():
        raise ValueError(f"Duplicate IDs found for {id_column=}")
    return parcels_gdf


def flatten_building_attributes(buildings: List[dict], country: str) -> dict:
    """
    Flatten building attributes

    Args:
        buildings: List of building features with attributes
        country: Country code for units (e.g. "US" for imperial, "EU" for metric)
    """
    flattened = {}
    for building in buildings:
        attribute = building["attributes"]
        if "has3dAttributes" in attribute:
            flattened["has_3d_attributes"] = TRUE_STRING if attribute["has3dAttributes"] else FALSE_STRING
            if attribute["has3dAttributes"]:
                if country in IMPERIAL_COUNTRIES:
                    flattened["height_ft"] = round(attribute["height"] * METERS_TO_FEET, 1)
                else:
                    flattened["height_m"] = round(attribute["height"], 1)
                for k, v in attribute["numStories"].items():
                    flattened[f"num_storeys_{k}_confidence"] = v
        if "fidelity" in attribute:
            flattened["fidelity"] = attribute["fidelity"]
    return flattened


def flatten_roof_attributes(roofs: List[dict], country: str) -> dict:
    """
    Flatten roof attributes

    Args:
        roofs: List of roof features with attributes
        country: Country code for units (e.g. "US" for imperial, "EU" for metric)
    """
    flattened = {}
    
    # Handle components and other attributes
    for roof in roofs:
        # Handle roofSpotlightIndex - check both camelCase and snake_case versions
        rsi_data = roof.get("roofSpotlightIndex") or roof.get("roof_spotlight_index")
        if rsi_data and isinstance(rsi_data, dict):
            if "value" in rsi_data:
                flattened["roof_spotlight_index"] = rsi_data["value"]
            if "confidence" in rsi_data:
                flattened["roof_spotlight_index_confidence"] = rsi_data["confidence"]
            if "modelVersion" in rsi_data:
                flattened["roof_spotlight_index_model_version"] = rsi_data["modelVersion"]
        
        for attribute in roof["attributes"]:
            if "components" in attribute:
                for component in attribute["components"]:
                    name = component["description"].lower().replace(" ", "_")
                    if "Low confidence" in attribute["description"]:
                        name = f"low_conf_{name}"
                    flattened[f"{name}_present"] = TRUE_STRING if component["areaSqm"] > 0 else FALSE_STRING
                    if country in IMPERIAL_COUNTRIES:
                        flattened[f"{name}_area_sqft"] = component["areaSqft"]
                    else:
                        flattened[f"{name}_area_sqm"] = component["areaSqm"]
                    flattened[f"{name}_confidence"] = component["confidence"]
                    if "dominant" in component:
                        flattened[f"{name}_dominant"] = TRUE_STRING if component["dominant"] else FALSE_STRING
                    # Handle ratio field if present
                    if "ratio" in component:
                        flattened[f"{name}_ratio"] = component["ratio"]
            elif "has3dAttributes" in attribute:
                flattened["has_3d_attributes"] = TRUE_STRING if attribute["has3dAttributes"] else FALSE_STRING
                if attribute["has3dAttributes"]:
                    flattened["pitch_degrees"] = attribute["pitch"]
    return flattened


def flatten_building_lifecycle_damage_attributes(building_lifecycles: List[dict]) -> dict:
    """
    Flatten building lifecycle damage attributes

    Args:
        building_lifecycles: List of building lifecycle features with attributes
    """

    flattened = {}
    for building_lifecycle in building_lifecycles:
        attribute = building_lifecycle.get("attributes", {})
        
        # Check if damage exists and is not None
        if "damage" in attribute and attribute["damage"] is not None:
            # Check if damage is a dictionary (expected) vs scalar or other type
            damage_data = attribute["damage"]
            if not isinstance(damage_data, dict):
                # Damage is scalar or unexpected type - skip processing
                continue
                
            # Check if femaCategoryConfidences exists and is valid
            if "femaCategoryConfidences" not in damage_data:
                continue
                
            damage_dic = damage_data["femaCategoryConfidences"]
            if damage_dic is None or not isinstance(damage_dic, dict) or len(damage_dic) == 0:
                # No valid damage categories - skip processing
                continue
                
            # Process valid damage data
            x = pd.Series(damage_dic)
            flattened["damage_class"] = x.idxmax()
            flattened["damage_class_confidence"] = x.max()
            for damage_class, confidence in damage_dic.items():
                flattened[f"damage_class_{damage_class}_confidence"] = confidence
    return flattened


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

            # Add primary feature attributes for discrete features if there are any
            if primary_decision == "largest_intersection":
                # NB: Sort first by clipped area (priority). However, sometimes clipped areas are zero (in the case of damage detection), so secondary sort on unclipped is necessary.
                primary_feature = class_features_gdf.sort_values(
                    ["clipped_area_sqm", "unclipped_area_sqm"], ascending=False
                ).iloc[0]

            elif primary_decision == "nearest":
                primary_point = Point(primary_lon, primary_lat)
                primary_point = gpd.GeoSeries(primary_point).set_crs("EPSG:4326").to_crs("EPSG:3857")[0]
                class_features_gdf_top = class_features_gdf.query("confidence >= @PRIMARY_FEATURE_HIGH_CONF_THRESH")

                if len(class_features_gdf_top) > 0:
                    nearest_feature_idx = (
                        class_features_gdf_top.set_geometry("geometry_feature")
                        .to_crs("EPSG:3857")
                        .distance(primary_point)
                        .idxmin()
                    )
                else:
                    nearest_feature_idx = (
                        class_features_gdf.set_geometry("geometry_feature")
                        .to_crs("EPSG:3857")
                        .distance(primary_point)
                        .idxmin()
                    )
                primary_feature = class_features_gdf.loc[nearest_feature_idx, :]
            else:
                raise NotImplementedError(f"Have not implemented primary_decision type '{primary_decision}'")
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
