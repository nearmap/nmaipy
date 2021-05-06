from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
import shapely.wkb
import shapely.wkt
import stringcase
import warnings

from nearmap_ai.constants import (
    AOI_ID_COLUMN_NAME,
    AREA_CRS,
    BUILDING_ID,
    CONSTRUCTION_ID,
    FEET_IN_METERS,
    LAT_LONG_CRS,
    POOL_ID,
    ROOF_ID,
    SOLAR_ID,
    SQUARED_FEET_IN_METERS,
    SURFACES_IDS,
    TRAMPOLINE_ID,
    VEG_IDS,
    TREE_OVERHANG_ID,
)

TRUE_STRING = "Y"
FALSE_STRING = "N"

DEFAULT_FILTERING = {
    "min_size": {
        BUILDING_ID: 16,
        ROOF_ID: 16,
        TRAMPOLINE_ID: 9,
        POOL_ID: 9,
        CONSTRUCTION_ID: 9,
        SOLAR_ID: 9,
    },
    "min_confidence": {
        BUILDING_ID: 0.8,
        ROOF_ID: 0.8,
        TRAMPOLINE_ID: 0.7,
        POOL_ID: 0.6,
        CONSTRUCTION_ID: 0.8,
        SOLAR_ID: 0.7,
    },
    "min_area_in_parcel": {
        BUILDING_ID: 25,
        ROOF_ID: 25,
        TRAMPOLINE_ID: 5,
        POOL_ID: 9,
        CONSTRUCTION_ID: 5,
        SOLAR_ID: 5,
    },
    "min_ratio_in_parcel": {
        BUILDING_ID: 0.5,
        ROOF_ID: 0.5,
        TRAMPOLINE_ID: 0.5,
        POOL_ID: 0.5,
        CONSTRUCTION_ID: 0.5,
        SOLAR_ID: 0.5,
    },
}


def read_from_file(
    path: Path,
    drop_empty: Optional[bool] = True,
    id_column: Optional[str] = "id",
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
    if path.suffix == ".csv":
        parcels_df = pd.read_csv(path)
        parcels_gdf = gpd.GeoDataFrame(
            parcels_df.drop("geometry", axis=1),
            geometry=parcels_df.geometry.fillna("POLYGON(EMPTY)").apply(shapely.wkt.loads),
            crs=source_crs,
        )
    elif path.suffix == ".parquet":
        parcels_df = pd.read_parquet(path)
        parcels_gdf = gpd.GeoDataFrame(
            parcels_df.drop("geometry", axis=1),
            geometry=parcels_df.geometry.fillna("POLYGON(EMPTY)").apply(lambda g: shapely.wkb.loads(g, hex=True)),
            crs=source_crs,
        )
    elif path.suffix in (".geojson", ".gpkg"):
        parcels_gdf = gpd.read_file(path)
    else:
        raise NotImplemented(f"Source format not supported: {path.suffix=}")

    # Set CRS and project if data CRS is not equal to target CRS
    if parcels_gdf.crs is None:
        parcels_gdf.set_crs(source_crs)
    if parcels_gdf.crs != target_crs:
        parcels_gdf = parcels_gdf.to_crs(target_crs)

    # Drop any empty geometries
    if drop_empty:
        parcels_gdf = parcels_gdf.dropna(subset=["geometry"])
        parcels_gdf = parcels_gdf[~parcels_gdf.is_empty]
        # For this we only check if the shape has a non-zero area, the value doesn't matter, so the warning can be
        # ignored.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.")
            parcels_gdf = parcels_gdf[parcels_gdf.area > 0]

    # Check that identifier is unique
    if parcels_gdf[id_column].duplicated().any():
        raise ValueError(f"Duplicate IDs found for {id_column=}")

    parcels_gdf = parcels_gdf.rename(columns={id_column: AOI_ID_COLUMN_NAME})
    parcels_gdf.columns = [stringcase.snakecase(c) for c in parcels_gdf.columns]
    return parcels_gdf


def filter_features_in_parcels(
    parcels_gdf: gpd.GeoDataFrame, features_gdf: gpd.GeoDataFrame, country: str, config: Optional[dict] = None
) -> gpd.GeoDataFrame:
    """
    Drop features that are not considered as "inside" or "belonging to" a parcel. These fall into two categories:
     - Features that are considered noise (small or low confidence)
     - Features that intersect with the parcel boundary only because of a misalignment between the parcel boundary and
       the feature data.

    Default thresholds to make theses decisions are defined, but can be overwritten at runtime.

    Args:
        parcels_gdf: Parcel data (see nearmap_ai.parcels.read_from_file)
        features_gdf: Features data (see nearmap_ai.FeatureApi.get_features_gdf_bulk)
        country: Country string used for area calculation ("US", "AU", "NZ", ...)
        config: Config dictionary. Can have any or all keys as the default config
                (see nearmap_ai.parcels.DEFAULT_FILTERING).

    Returns: Filtered features_gdf GeoDataFrame
    """
    if config is None:
        config = {}

    # Project to Albers equal area to enable area calculation in squared metres
    country = country.lower()
    if country not in AREA_CRS.keys():
        raise ValueError(f"Unsupported country ({country=})")
    projected_features_gdf = features_gdf.to_crs(AREA_CRS[country])
    projected_parcels_gdf = parcels_gdf.to_crs(AREA_CRS[country])

    # Merge parcels and features
    gdf = projected_features_gdf.merge(projected_parcels_gdf, on="aoi_id", how="left", suffixes=["_feature", "_aoi"])
    # Calculate the area of each feature that falls within the parcel
    gdf["intersection_area"] = gdf.apply(lambda row: row.geometry_feature.intersection(row.geometry_aoi).area, axis=1)
    # Calculate the ratio of a feature that falls within the parcel
    gdf["intersection_ratio"] = gdf.intersection_area / gdf.area_sqm

    # Filter small features
    filter = config.get("min_size", DEFAULT_FILTERING["min_size"])
    gdf = gdf[gdf.class_id.map(filter).fillna(0) < gdf.area_sqm]

    # Filter low confidence features
    filter = config.get("min_confidence", DEFAULT_FILTERING["min_confidence"])
    gdf = gdf[gdf.class_id.map(filter).fillna(0) < gdf.confidence]

    # Filter based on area and ratio in parcel
    filter = config.get("min_area_in_parcel", DEFAULT_FILTERING["min_area_in_parcel"])
    area_mask = gdf.class_id.map(filter).fillna(0) < gdf.intersection_area
    filter = config.get("min_ratio_in_parcel", DEFAULT_FILTERING["min_ratio_in_parcel"])
    ratio_mask = gdf.class_id.map(filter).fillna(0) < gdf.intersection_ratio
    gdf = gdf[area_mask | ratio_mask]

    return features_gdf.merge(
        gdf[["feature_id", "aoi_id", "intersection_area"]], on=["feature_id", "aoi_id"], how="inner"
    )


def flatten_building_attributes(attributes: List[dict]) -> dict:
    """
    Flatten building attributes
    """
    flattened = {}
    for attribute in attributes:
        if "has3dAttributes" in attribute:
            flattened["has_3d_attributes"] = TRUE_STRING if attribute["has3dAttributes"] else FALSE_STRING
            if attribute["has3dAttributes"]:
                flattened["height_m"] = round(attribute["height"], 1)
                flattened["height_ft"] = round(attribute["height"] * FEET_IN_METERS, 1)
                for k, v in attribute["numStories"].items():
                    flattened[f"num_storeys_{k}_confidence"] = v
    return flattened


def flatten_roof_attributes(attributes: List[dict]) -> dict:
    """
    Flatten roof attributes
    """
    flattened = {}
    for attribute in attributes:
        if "components" in attribute:
            for component in attribute["components"]:
                name = component["description"].lower().replace(" ", "_")
                flattened[f"{name}_present"] = TRUE_STRING if component["areaSqm"] > 0 else FALSE_STRING
                flattened[f"{name}_area_sqm"] = component["areaSqm"]
                flattened[f"{name}_area_sqft"] = component["areaSqft"]
                flattened[f"{name}_confidence"] = component["confidence"]
                if "dominant" in component:
                    flattened[f"{name}_dominant"] = TRUE_STRING if component["dominant"] else FALSE_STRING
        elif "has3dAttributes" in attribute:
            flattened["has_3d_attributes"] = TRUE_STRING if attribute["has3dAttributes"] else FALSE_STRING
            if attribute["has3dAttributes"]:
                flattened["pitch"] = attribute["pitch"]
    return flattened


def feature_attributes(features_gdf: gpd.GeoDataFrame, classes_df: pd.DataFrame) -> dict:
    """
    Flatten features for a parcel into a flat dictionary.

    Args:
        features_gdf: Features for a parcel
        classes_df: Class name and ID lookup (index of the dataframe) to include.

    Returns: Flat dictionary

    """
    # Add present, object count, area, and confidence for all used feature classes
    parcel = {}
    for (class_id, name) in classes_df.description.iteritems():
        name = name.lower().replace(" ", "_")
        class_gdf = features_gdf[features_gdf.class_id == class_id]

        # Add attributes that apply to all feature classes
        parcel[f"{name}_present"] = TRUE_STRING if len(class_gdf) > 0 else FALSE_STRING
        parcel[f"{name}_count"] = len(class_gdf)
        parcel[f"{name}_total_area_sqm"] = class_gdf.area_sqm.sum()
        parcel[f"{name}_total_area_sqft"] = class_gdf.area_sqft.sum()
        parcel[f"{name}_total_clipped_area_sqm"] = round(class_gdf.intersection_area.sum(), 1)
        parcel[f"{name}_total_clipped_area_sqft"] = round(class_gdf.intersection_area.sum() * SQUARED_FEET_IN_METERS, 1)
        if len(class_gdf) > 0:
            parcel[f"{name}_confidence"] = 1 - (1 - class_gdf.confidence).prod()
        else:
            parcel[f"{name}_confidence"] = 1.0

        if class_id not in VEG_IDS + SURFACES_IDS + [TREE_OVERHANG_ID]:
            if len(class_gdf) > 0:
                # Add primary feature attributes for discrete features if there are any
                primary_feature = class_gdf.loc[class_gdf.intersection_area.idxmax()]
                parcel[f"primary_{name}_area_sqm"] = primary_feature.area_sqm
                parcel[f"primary_{name}_area_sqft"] = primary_feature.area_sqft
                parcel[f"primary_{name}_clipped_area_sqm"] = round(primary_feature.intersection_area, 1)
                parcel[f"primary_{name}_clipped_area_sqft"] = round(
                    primary_feature.intersection_area * SQUARED_FEET_IN_METERS, 1
                )
                parcel[f"primary_{name}_confidence"] = primary_feature.confidence

                # Add roof and building attributes
                if class_id in [ROOF_ID, BUILDING_ID]:
                    if class_id == ROOF_ID:
                        primary_attributes = flatten_roof_attributes(primary_feature.attributes)
                    else:
                        primary_attributes = flatten_building_attributes(primary_feature.attributes)

                    for key, val in primary_attributes.items():
                        parcel[f"primary_{name}_" + str(key)] = val
            else:
                # Fill values if there are no features
                parcel[f"primary_{name}_area_sqm"] = 0.0
                parcel[f"primary_{name}_area_sqft"] = 0.0
                parcel[f"primary_{name}_clipped_area_sqm"] = 0.0
                parcel[f"primary_{name}_clipped_area_sqft"] = 0.0
                parcel[f"primary_{name}_confidence"] = 1.0

    return parcel


def parcel_rollup(parcels_gdf: gpd.GeoDataFrame, features_gdf: gpd.GeoDataFrame, classes_df: pd.DataFrame):
    """
    Summarize feature data to parcel attributes.

    Args:
        parcels_gdf: Parcels GeoDataFrame
        features_gdf: Features GeoDataFrame
        classes_df: Class name and ID lookup

    Returns:
        Parcel rollup DataFrame
    """
    df = features_gdf.merge(parcels_gdf[["aoi_id", "geometry"]], on="aoi_id", suffixes=["_feature", "_aoi"])
    rollups = []
    # Loop over parcels with features in them
    for aoi_id, group in df.groupby("aoi_id"):
        parcel = feature_attributes(group, classes_df)
        parcel["aoi_id"] = aoi_id
        parcel["mesh_date"] = group.mesh_date.iloc[0]
        rollups.append(parcel)
    # Loop over parcels without features in them
    for row in parcels_gdf[~parcels_gdf.aoi_id.isin(features_gdf.aoi_id)].itertuples():
        parcel = feature_attributes(
            gpd.GeoDataFrame([], columns=["class_id", "area_sqm", "area_sqft", "intersection_area"]), classes_df
        )
        parcel["aoi_id"] = row.aoi_id
        rollups.append(parcel)
    # Combine, validate and return
    rollup_df = pd.DataFrame(rollups)
    if len(rollup_df) != len(parcels_gdf):
        raise RuntimeError(f"Parcel count validation error: {len(rollup_df)=} not equal to {len(parcels_gdf)=}")
    return rollup_df
