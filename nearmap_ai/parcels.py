from pathlib import Path
from typing import List, Optional, Union
import warnings

import geopandas as gpd
import pandas as pd
import shapely.wkb
import shapely.wkt
from shapely.geometry import MultiPolygon, Polygon
import stringcase

from nearmap_ai import log
from nearmap_ai.constants import (
    AOI_ID_COLUMN_NAME,
    AREA_CRS,
    BUILDING_ID,
    VEG_MEDHIGH_ID,
    VEG_LOW_ID,
    CLASSES_WITH_NO_PRIMARY_FEATURE,
    CONSTRUCTION_ID,
    IMPERIAL_COUNTRIES,
    LAT_LONG_CRS,
    LAT_PRIMARY_COL_NAME,
    LON_PRIMARY_COL_NAME,
    METERS_TO_FEET,
    POOL_ID,
    ROOF_ID,
    SOLAR_ID,
    SQUARED_METERS_TO_SQUARED_FEET,
    TRAMPOLINE_ID,
    MeasurementUnits,
)

TRUE_STRING = "Y"
FALSE_STRING = "N"
PRIMARY_FEATURE_HIGH_CONF_THRESH = 0.9

# All area values are in squared metres
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
    "min_fidelity": {
        BUILDING_ID: 0.0,
        ROOF_ID: 0.0,
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

TREE_BUFFERS_M = dict(
    buffer_5ft=1.524,
    buffer_10ft=3.048,
    buffer_30ft=9.144,
    buffer_100ft=30.48,
)

logger = log.get_logger()


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
    if path.suffix in (".csv", ".psv", ".tsv"):
        if path.suffix == ".csv":
            parcels_df = pd.read_csv(path)
        elif path.suffix == ".psv":
            parcels_df = pd.read_csv(path, sep="|")
        elif path.suffix == ".tsv":
            parcels_df = pd.read_csv(path, sep="\t")
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
        parcels_gdf = parcels_gdf[parcels_gdf.is_valid]
        # For this we only check if the shape has a non-zero area, the value doesn't matter, so the warning can be
        # ignored.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.")
            parcels_gdf = parcels_gdf[parcels_gdf.area > 0]

    if len(parcels_gdf) == 0:
        raise RuntimeError(f"No valid parcels in {path=}")

    # Check that identifier is unique
    if id_column not in parcels_gdf:
        parcels_gdf[id_column] = parcels_gdf.index
    if parcels_gdf[id_column].duplicated().any():
        raise ValueError(f"Duplicate IDs found for {id_column=}")

    parcels_gdf = parcels_gdf.rename(columns={id_column: AOI_ID_COLUMN_NAME})
    return parcels_gdf


def filter_features_in_parcels(features_gdf: gpd.GeoDataFrame, config: Optional[dict] = None) -> gpd.GeoDataFrame:
    """
    Drop features that are not considered as "inside" or "belonging to" a parcel. These fall into two categories:
     - Features that are considered noise (small or low confidence)
     - Features that intersect with the parcel boundary only because of a misalignment between the parcel boundary and
       the feature data.

    Default thresholds to make theses decisions are defined, but can be overwritten at runtime.

    Args:
        features_gdf: Features data (see nearmap_ai.FeatureApi.get_features_gdf_bulk)
        config: Config dictionary. Can have any or all keys as the default config
                (see nearmap_ai.parcels.DEFAULT_FILTERING).

    Returns: Filtered features_gdf GeoDataFrame
    """
    if config is None:
        config = {}
    gdf = features_gdf.copy()

    # Calculate the ratio of a feature that falls within the parcel
    gdf["intersection_ratio"] = gdf.clipped_area_sqm / gdf.unclipped_area_sqm

    # Filter small features
    filter = config.get("min_size", DEFAULT_FILTERING["min_size"])
    gdf = gdf[gdf.class_id.map(filter).fillna(0) < gdf.unclipped_area_sqm]

    # Filter low confidence features
    filter = config.get("min_confidence", DEFAULT_FILTERING["min_confidence"])
    gdf = gdf[gdf.class_id.map(filter).fillna(0) < gdf.confidence]

    # Filter low fidelity features. If fidelity not present, assume 1 (perfect shape) to avoid rejection.
    filter = config.get("min_fidelity", DEFAULT_FILTERING["min_fidelity"])
    gdf = gdf[gdf.class_id.map(filter).fillna(0) < gdf.fidelity.fillna(1)]

    # Filter based on area and ratio in parcel
    filter = config.get("min_area_in_parcel", DEFAULT_FILTERING["min_area_in_parcel"])
    area_mask = gdf.class_id.map(filter).fillna(0) < gdf.clipped_area_sqm
    filter = config.get("min_ratio_in_parcel", DEFAULT_FILTERING["min_ratio_in_parcel"])
    ratio_mask = gdf.class_id.map(filter).fillna(0) < gdf.intersection_ratio
    gdf = gdf[area_mask | ratio_mask]

    # Only drop objects where the parent has been explicitly removed as above (otherwise we drop solar panels without a building request, etc.)
    if "parent_id" in gdf:
        feature_ids_removed = set(features_gdf.feature_id) - set(gdf.feature_id)
        parent_removed = gdf.parent_id.isin(feature_ids_removed)
        no_parent = (gdf.parent_id == "") | gdf.parent_id.isna()
        gdf = gdf.loc[~parent_removed | no_parent]

    return gdf.reset_index(drop=True)


def flatten_building_attributes(attributes: List[dict], country: str) -> dict:
    """
    Flatten building attributes
    """
    flattened = {}
    for attribute in attributes:
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
            flattened["fidelity"] = attributes["fidelity"]
    return flattened


def flatten_roof_attributes(attributes: List[dict], country: str) -> dict:
    """
    Flatten roof attributes
    """
    flattened = {}
    for attribute in attributes:
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
        elif "has3dAttributes" in attribute:
            flattened["has_3d_attributes"] = TRUE_STRING if attribute["has3dAttributes"] else FALSE_STRING
            if attribute["has3dAttributes"]:
                flattened["pitch_degrees"] = attribute["pitch"]
    return flattened


def feature_attributes(
    features_gdf: gpd.GeoDataFrame,
    classes_df: pd.DataFrame,
    country: str,
    parcel_geom: Union[MultiPolygon, Polygon, None],
    primary_decision: str,
    primary_lat: float = None,
    primary_lon: float = None,
    calc_buffers: bool = False,
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
        calc_buffers: Whether to calculate and include buffers

    Returns: Flat dictionary

    """
    mu = MeasurementUnits(country)
    area_units = mu.area_units()

    # Add present, object count, area, and confidence for all used feature classes
    parcel = {}
    for (class_id, name) in classes_df.description.iteritems():
        name = name.lower().replace(" ", "_")
        class_features_gdf = features_gdf[features_gdf.class_id == class_id]

        # Add attributes that apply to all feature classes
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

        # Select and produce results for the primary feature of each feature class
        if class_id not in CLASSES_WITH_NO_PRIMARY_FEATURE:
            if len(class_features_gdf) > 0:

                # Add primary feature attributes for discrete features if there are any
                if primary_decision == "largest_intersection":
                    primary_feature = class_features_gdf.loc[class_features_gdf.clipped_area_sqm.idxmax()]
                elif primary_decision == "nearest":
                    primary_point = shapely.geometry.Point(primary_lon, primary_lat)
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
                    parcel[f"primary_{name}_area_sqft"] = primary_feature.area_sqft
                    parcel[f"primary_{name}_clipped_area_sqft"] = round(primary_feature.clipped_area_sqft, 1)
                    parcel[f"primary_{name}_unclipped_area_sqft"] = round(primary_feature.unclipped_area_sqft, 1)
                else:
                    parcel[f"primary_{name}_area_sqm"] = primary_feature.area_sqm
                    parcel[f"primary_{name}_clipped_area_sqm"] = round(primary_feature.clipped_area_sqm, 1)
                    parcel[f"primary_{name}_unclipped_area_sqm"] = round(primary_feature.unclipped_area_sqm, 1)
                parcel[f"primary_{name}_confidence"] = primary_feature.confidence
                if class_id == BUILDING_ID:
                    parcel[f"primary_{name}_fidelity"] = primary_feature.fidelity

                # Add roof and building attributes
                if class_id in [ROOF_ID, BUILDING_ID]:
                    if class_id == ROOF_ID:
                        primary_attributes = flatten_roof_attributes(primary_feature.attributes, country=country)
                    else:
                        primary_attributes = flatten_building_attributes(primary_feature.attributes, country=country)

                    for key, val in primary_attributes.items():
                        parcel[f"primary_{name}_" + str(key)] = val
            else:
                # Fill values if there are no features
                parcel[f"primary_{name}_area_{area_units}"] = 0.0
                parcel[f"primary_{name}_clipped_area_{area_units}"] = 0.0
                parcel[f"primary_{name}_unclipped_area_{area_units}"] = 0.0
                parcel[f"primary_{name}_confidence"] = None

        if class_id == BUILDING_ID:

            # Initialise buffers to None - only wipe over with correct answers if valid can be produced.
            if len(class_features_gdf) > 0 and calc_buffers:
                for B in TREE_BUFFERS_M:
                    parcel[f"building_{B}_tree_zone_sqm"] = None
                    parcel[f"building_count_nonzero_{B}_tree_zone"] = None

                # Buffers can't be valid if any buildings clip the parcel. Shortcut attempted buffer creation.
                rounding_factor = 0.99  # To account for pre-calculated vs on-the-fly area calc differences
                if (
                    parcel[f"{name}_total_clipped_area_{area_units}"]
                    < rounding_factor * parcel[f"{name}_total_unclipped_area_{area_units}"]
                ):
                    break

                # Create vegetation buffers.
                veg_medhigh_features_gdf = features_gdf[features_gdf.class_id == VEG_MEDHIGH_ID]
                if len(veg_medhigh_features_gdf) > 0:
                    veg_medhigh_features_gdf = gpd.GeoDataFrame(
                        veg_medhigh_features_gdf, crs=LAT_LONG_CRS, geometry="geometry_feature"
                    )

                for B in TREE_BUFFERS_M:
                    gdf_buffered_buildings = gpd.GeoDataFrame(
                        class_features_gdf, geometry="geometry_feature", crs=LAT_LONG_CRS
                    )
                    # Wipe over feature geometries with their buffered version...
                    gdf_buffered_buildings["geometry_feature"] = (
                        gdf_buffered_buildings.to_crs(AREA_CRS[country]).buffer(TREE_BUFFERS_M[B]).to_crs(LAT_LONG_CRS)
                    )

                    if parcel_geom is not None and (
                        gdf_buffered_buildings["geometry_feature"].intersection(parcel_geom).area.sum()
                        / gdf_buffered_buildings["geometry_feature"].area.sum()
                    ) < 1:
                        # Buffer exceeds Query AOI somewhere.
                        break

                    if len(veg_medhigh_features_gdf) > 0:
                        gdf_intersection = gdf_buffered_buildings.overlay(veg_medhigh_features_gdf, how="intersection")
                        gdf_intersection["buff_area_sqm"] = gdf_intersection.to_crs(AREA_CRS[country]).area
                        parcel[f"building_{B}_tree_zone_sqm"] = gdf_intersection["buff_area_sqm"].sum()
                        bldg_count = (
                            gdf_intersection.groupby("feature_id_1")
                            .aggregate({"buff_area_sqm": "sum"})
                            .query("buff_area_sqm > 0")
                        )
                        bldg_count = len(bldg_count)
                        parcel[f"building_count_nonzero_{B}_tree_zone"] = bldg_count
    return parcel


def parcel_rollup(
    parcels_gdf: gpd.GeoDataFrame,
    features_gdf: gpd.GeoDataFrame,
    classes_df: pd.DataFrame,
    country: str,
    calc_buffers: bool,
    primary_decision: str,
):
    """
    Summarize feature data to parcel attributes.

    Args:
        parcels_gdf: Parcels GeoDataFrame
        features_gdf: Features GeoDataFrame
        classes_df: Class name and ID lookup
        country: Country code for units.
        calc_buffers: Calculate buffered features
        primary_decision: The basis on which the primary features are chosen

    Returns:
        Parcel rollup DataFrame
    """
    mu = MeasurementUnits(country)
    area_units = mu.area_units()

    if len(parcels_gdf[AOI_ID_COLUMN_NAME].unique()) != len(parcels_gdf):
        raise Exception(
            f"AOI id column {AOI_ID_COLUMN_NAME} is NOT unique in parcels/AOI dataframe, but it should be: there are {len(parcels_gdf[AOI_ID_COLUMN_NAME].unique())} unique AOI ids and {len(parcels_gdf)} rows in the dataframe"
        )
    if primary_decision == "nearest":
        merge_cols = [AOI_ID_COLUMN_NAME, LAT_PRIMARY_COL_NAME, LON_PRIMARY_COL_NAME, "geometry"]
    else:
        merge_cols = [AOI_ID_COLUMN_NAME]
        if "geometry" in parcels_gdf.columns:
            merge_cols += ["geometry"]

    df = features_gdf.merge(parcels_gdf[merge_cols], on=AOI_ID_COLUMN_NAME, suffixes=["_feature", "_aoi"])

    rollups = []
    # Loop over parcels with features in them
    for aoi_id, group in df.groupby(AOI_ID_COLUMN_NAME):
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
            parcel_geom = parcels_gdf[parcels_gdf[AOI_ID_COLUMN_NAME] == aoi_id]
            assert len(parcel_geom) == 1
            parcel_geom = parcel_geom.geometry.values[0]
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
            calc_buffers=calc_buffers,
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
    for row in parcels_gdf[~parcels_gdf[AOI_ID_COLUMN_NAME].isin(features_gdf[AOI_ID_COLUMN_NAME])].itertuples():
        parcel = feature_attributes(
            gpd.GeoDataFrame([], columns=["class_id", area_name, f"clipped_{area_name}", f"unclipped_{area_name}"]),
            classes_df,
            country=country,
            parcel_geom=row.geometry if hasgeom else None,
            primary_decision=primary_decision,
            calc_buffers=calc_buffers,
        )
        parcel[AOI_ID_COLUMN_NAME] = row._asdict()[AOI_ID_COLUMN_NAME]
        rollups.append(parcel)
    # Combine, validate and return
    rollup_df = pd.DataFrame(rollups)
    if len(rollup_df) != len(parcels_gdf):
        raise RuntimeError(f"Parcel count validation error: {len(rollup_df)=} not equal to {len(parcels_gdf)=}")
    return rollup_df
