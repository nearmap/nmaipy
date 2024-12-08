from pathlib import Path
from typing import List, Optional, Union
import warnings

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, Point

from nmaipy import log
import nmaipy.reference_code
from nmaipy.constants import *

TRUE_STRING = "Y"
FALSE_STRING = "N"
PRIMARY_FEATURE_HIGH_CONF_THRESH = 0.9

# All area values are in squared metres
DEFAULT_FILTERING = {
    "min_size": {
        BUILDING_LIFECYCLE_ID: 4,
        BUILDING_ID: 4,
        BUILDING_NEW_ID: 4,
        ROOF_ID: 4,
        TRAMPOLINE_ID: 1,
        POOL_ID: 4,
        CONSTRUCTION_ID: 5,
        SOLAR_ID: 1,
        PLAYGROUND_ID: 2,
    },
    "min_confidence": {
        BUILDING_LIFECYCLE_ID: 0.65,
        BUILDING_ID: 0.65,
        BUILDING_NEW_ID: 0.65,
        ROOF_ID: 0.58,
        TRAMPOLINE_ID: 0.6,
        POOL_ID: 0.55,
        CONSTRUCTION_ID: 0.8,
        SOLAR_ID: 0.61,
        # Roof Conditions
        CLASS_1050_TARP: 0.52,  # "tarp",
        CLASS_1052_RUST: 0.50,  # "rust",
        CLASS_1079_MISSING_SHINGLES: 0.50,  # "missing_shingles",
        CLASS_1139_DEBRIS: 0.50,  # "debris",
        CLASS_1140_EXPOSED_DECK: 0.51,  # "exposed_deck",
        CLASS_1051_PONDING: 0.50,  # "ponding",
        CLASS_1144_STAINING: 0.50,  # "staining",
        CLASS_1146_WORN_SHINGLES: 0.50,  # "worn_shingles",
        CLASS_1147_EXPOSED_UNDERLAYMENT: 0.59,  # "exposed_underlayment",
        CLASS_1149_PATCHING: 0.50,  # "patching",
        CLASS_1186_STRUCTURAL_DAMAGE: 0.50,  # "structural_damage",
        # Roof Shapes
        CLASS_1013_HIP: 0.50, # Keep fixed at 0.50 due to rate filing
        CLASS_1014_GABLE: 0.50, # Keep fixed at 0.50 due to rate filing
        CLASS_1015_DUTCH_GABLE: 0.57,
        CLASS_1019_GAMBREL: 0.70,
        CLASS_1020_CONICAL: 0.50,  # turret / conical. This is normall small part on the roof. It's hard to be larger than 0.58
        CLASS_1173_PARAPET: 0.50,  # check the definition of ontology. If ths parpet is the edges of the roof, then it's hard to be larger than 0.5
        CLASS_1174_MANSARD: 0.64,
        CLASS_1176_JERKINHEAD: 0.71,
        CLASS_1178_QUONSET: 0.52,
        CLASS_1180_BOWSTRING_TRUSS: 0.58,
        # Roof Materials
        CLASS_1191_FLAT: 0.50,
        CLASS_1007_TILE: 0.55,
        CLASS_1008_ASPHALT_SHINGLE: 0.50, # Keep fixed at 0.50 due to rate filing
        CLASS_1009_METAL_PANEL: 0.50, # Keep fixed at 0.50 due to rate filing
        CLASS_1100_BALLASTED: 0.64,
        CLASS_1101_MOD_BIT: 0.50,
        CLASS_1103_TPO: 0.53,
        CLASS_1104_EPDM: 0.57,
        CLASS_1105_WOOD_SHAKE: 0.61,
        CLASS_1160_CLAY_TILE: 0.63,
        CLASS_1163_SLATE: 0.58,
        CLASS_1165_BUILT_UP: 0.50,
        CLASS_1168_ROOF_COATING: 0.53,
    },
    "min_fidelity": {
        BUILDING_ID: 0.15,
        BUILDING_NEW_ID: 0.15,
        ROOF_ID: 0.15,
    },
    "min_area_in_parcel": {
        BUILDING_LIFECYCLE_ID: 0, # Point classes have no clipped area, so we can't filter them out based on area.
        BUILDING_ID: 4,
        BUILDING_NEW_ID: 4,
        ROOF_ID: 4,
        TRAMPOLINE_ID: 1,
        POOL_ID: 4,
        CONSTRUCTION_ID: 5,
        SOLAR_ID: 1,
    },
    "min_ratio_in_parcel": {
        BUILDING_LIFECYCLE_ID: 0,
        BUILDING_ID: 0,  # Defer to more complex algorithm for building and roof - important for large buildings.
        BUILDING_NEW_ID: 0,
        BUILDING_UNDER_CONSTRUCTION_ID: 0,
        ROOF_ID: 0,
        TRAMPOLINE_ID: 0.5,
        POOL_ID: 0.5,
        CONSTRUCTION_ID: 0.5,
        SOLAR_ID: 0.5,
        CAR_ID: 0.5,
        WHEELED_CONSTRUCTION_VEHICLE_ID: 0.5,
        CONSTRUCTION_CRANE_ID: 0.5,
        BOAT_ID: 0.5,
        SILO_ID: 0.5,
        SKYLIGHT_ID: 0.5,
        PLAYGROUND_ID: 0.5,
    },
    "building_style_filtering": {
        BUILDING_LIFECYCLE_ID: True,
        BUILDING_ID: True,
        BUILDING_NEW_ID: True,
        BUILDING_UNDER_CONSTRUCTION_ID: True,
        ROOF_ID: True,
    },
}

BUFFER_ZONES_M = dict(
    buffer_0ft=0.0,
    buffer_5ft=1.524,
    buffer_10ft=3.048,
    buffer_30ft=9.144,
    buffer_100ft=30.48,
)

BUFFER_CLASSES = {
    "tree": VEG_MEDHIGH_ID,
    "woody_veg": VEG_WOODY_COMPOSITE_ID,
    "roof": ROOF_ID,
    "yard_debris": CLASS_1111_YARD_DEBRIS,
}

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
        if suffix == "csv":
            parcels_gdf = pd.read_csv(path, index_col=id_column)
        elif suffix == "psv":
            parcels_gdf = pd.read_csv(path, sep="|", index_col=id_column)
        elif suffix == "tsv":
            parcels_gdf = pd.read_csv(path, sep="\t")
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
                parcels_gdf, geometry=geometry,
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
            logger.warning(f"Missing {AOI_ID_COLUMN_NAME} column in parcel data - generating unique IDs")
            parcels_gdf.index.name = id_column  # Set a new unique ordered index for reference
        else: # The index must already be there as a column
            logger.warning(f"Moving {AOI_ID_COLUMN_NAME} to be the index - generating unique IDs")
            parcels_gdf = parcels_gdf.set_index(id_column)
    if parcels_gdf.index.duplicated().any():
        raise ValueError(f"Duplicate IDs found for {id_column=}")
    return parcels_gdf


def filter_features_in_parcels(features_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame, region: str) -> gpd.GeoDataFrame:
    """
    Drop features that are not considered as "inside" or "belonging to" a parcel. These fall into two categories:
     - Features that are considered noise (small or low confidence)
     - Features that intersect with the parcel boundary only because of a misalignment between the parcel boundary and
       the feature data.

    Default thresholds to make theses decisions are defined, but can be overwritten at runtime.

    Args:
        features_gdf: Features data (see nmaipy.FeatureApi.get_features_gdf_bulk)

    Returns: Filtered features_gdf GeoDataFrame
    """
    if len(features_gdf) == 0:
        return features_gdf
    gdf = features_gdf.copy()

    if "clipped_area_sqm" in gdf.columns and "unclipped_area_sqm" in gdf.columns:
        suffix = "sqm"
    elif "clipped_area_sqft" in gdf.columns and "unclipped_area_sqft" in gdf.columns:
        suffix = "sqft"
    else:
        raise Exception(
            f"We need consistent meters or feet to do filtering calculations, but we did not have the necessary columns... they are {gdf.columns.tolist()} in length {len(gdf)} dataframe"
        )

    # Calculate the ratio of a feature that falls within the parcel
    gdf["intersection_ratio"] = gdf["clipped_area_" + suffix] / gdf["unclipped_area_" + suffix]
    gdf["intersection_ratio"] = gdf["intersection_ratio"].fillna(1) # If unclipped area is zero, assume the feature is fully inside the parcel

    # Filter small features
    gdf = gdf[gdf.class_id.map(DEFAULT_FILTERING["min_size"]).fillna(0) <= gdf["unclipped_area_" + suffix]]

    # Filter low confidence features
    gdf = gdf[gdf.class_id.map(DEFAULT_FILTERING["min_confidence"]).fillna(0) <= gdf.confidence]

    # Filter low fidelity features. If fidelity not present, assume 1 (perfect shape) to avoid rejection.
    gdf = gdf[gdf.class_id.map(DEFAULT_FILTERING["min_fidelity"]).fillna(0) <= gdf.fidelity.fillna(1)]

    building_style_ids = DEFAULT_FILTERING["building_style_filtering"].keys()
    out_gdf_building_style = []
    gdf_non_building_style = gdf[~gdf.class_id.isin(building_style_ids)]

    if not isinstance(aoi_gdf, gpd.GeoDataFrame):
        logger.warning("AOI geometries not available, skipping building style filtering")
    else:
        for aoi_id in gdf.index.unique():  # Loop over each AOI in the set
            gdf_aoi = gdf.loc[[aoi_id]]
            gdf_aoi_buildings = gdf_aoi[gdf_aoi.class_id.isin(building_style_ids)]
            if len(gdf_aoi_buildings) == 0:
                continue # Skip if there are no buildings in the AOI
            features_from_aoi = aoi_gdf.loc[[aoi_id]]
            aoi_poly = (
                features_from_aoi.to_crs(AREA_CRS[region]).iloc[0].geometry
            )  # Get in metric projection for area/geospatial calcs in metres.
            building_statuses = []
            for building_poly in gdf_aoi_buildings.to_crs(AREA_CRS[region]).geometry: # Loop through buildings in the AOI
                building_status = nmaipy.reference_code.get_building_status(building_poly, aoi_poly)
                building_statuses.append(building_status)
            building_statuses = pd.DataFrame(building_statuses)
            building_statuses.index = gdf_aoi_buildings.index
            gdf_aoi_buildings = pd.concat([gdf_aoi_buildings, building_statuses], axis=1) # Append extra columns for all buildings in this parcel
            gdf_aoi_buildings = gdf_aoi_buildings[gdf_aoi_buildings.building_keep].drop(columns=["building_keep"]) # Remove any we should filter out
            out_gdf_building_style.append(gdf_aoi_buildings)

        if len(out_gdf_building_style) > 0:
            out_gdf_building_style = pd.concat(out_gdf_building_style)
            gdf = pd.concat([gdf_non_building_style, out_gdf_building_style])
        else:
            gdf = gdf_non_building_style

    # Filter based on area and ratio in parcel
    area_mask = gdf.class_id.map(DEFAULT_FILTERING["min_area_in_parcel"]).fillna(0) <= gdf["clipped_area_" + suffix]
    ratio_mask = gdf.class_id.map(DEFAULT_FILTERING["min_ratio_in_parcel"]).fillna(0) <= gdf.intersection_ratio
    gdf = gdf[area_mask & ratio_mask]

    # Only drop objects where the parent has been explicitly removed as above (otherwise we drop solar panels without a building request, etc.)
    if "parent_id" in gdf:
        feature_ids_removed = set(features_gdf.feature_id) - set(gdf.feature_id)
        parent_removed = gdf.parent_id.isin(feature_ids_removed)
        no_parent = (gdf.parent_id == "") | gdf.parent_id.isna()
        gdf = gdf.loc[~parent_removed | no_parent]

    return gdf


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


def flatten_building_lifecycle_damage_attributes(attributes: List[dict]) -> dict:
    """
    Flatten building lifecycle damage attributes
    """

    flattened = {}
    for attribute in attributes:
        if "damage" in attribute:
            damage_dic = attribute["damage"]["femaCategoryConfidences"]
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

        if class_id in DEFAULT_FILTERING["building_style_filtering"].keys():
            for col in ["building_small", "building_multiparcel"]:
                if col in class_features_gdf.columns:
                    s = col.split("_")[1]
                    parcel[f"{name}_{s}_count"] = len(class_features_gdf[class_features_gdf[col]])

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
                primary_feature = class_features_gdf.sort_values(["clipped_area_sqm", "unclipped_area_sqm"], ascending=False).iloc[0]

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
                parcel[f"primary_{name}_area_sqft"] = primary_feature.area_sqft
                parcel[f"primary_{name}_clipped_area_sqft"] = round(primary_feature.clipped_area_sqft, 1)
                parcel[f"primary_{name}_unclipped_area_sqft"] = round(primary_feature.unclipped_area_sqft, 1)
            else:
                parcel[f"primary_{name}_area_sqm"] = primary_feature.area_sqm
                parcel[f"primary_{name}_clipped_area_sqm"] = round(primary_feature.clipped_area_sqm, 1)
                parcel[f"primary_{name}_unclipped_area_sqm"] = round(primary_feature.unclipped_area_sqm, 1)
            parcel[f"primary_{name}_confidence"] = primary_feature.confidence
            if class_id in DEFAULT_FILTERING["building_style_filtering"].keys():
                parcel[f"primary_{name}_fidelity"] = primary_feature.fidelity

            # Add roof and building attributes
            if class_id in DEFAULT_FILTERING["building_style_filtering"].keys():
                for col in ["building_small", "building_multiparcel"]:
                    s = col.split("_")[1]
                    parcel[f"primary_{name}_{s}"] = primary_feature[col]
                if class_id == ROOF_ID:
                    primary_attributes = flatten_roof_attributes(primary_feature.attributes, country=country)
                    primary_attributes["feature_id"] = primary_feature.feature_id
                elif class_id in [BUILDING_ID, BUILDING_NEW_ID]:
                    primary_attributes = flatten_building_attributes(primary_feature.attributes, country=country)
                else:
                    primary_attributes = {}

                for key, val in primary_attributes.items():
                    parcel[f"primary_{name}_" + str(key)] = val
            if class_id == BUILDING_LIFECYCLE_ID:
                # Provide the confidence values for each damage rating class for the primary building lifecycle feature
                primary_attributes = flatten_building_lifecycle_damage_attributes(primary_feature.attributes)
                for key, val in primary_attributes.items():
                    parcel[f"primary_{name}_" + str(key)] = val

            if class_id == ROOF_ID:
                if not calc_buffers:
                    continue

                assert parcel_geom is not None, "Parcel geometry must be provided for buffer calculations"

                # Convert everything to area-based CRS upfront
                area_crs = AREA_CRS[country]
                parcel_geom_area = gpd.GeoSeries([parcel_geom], crs=LAT_LONG_CRS).to_crs(area_crs)[0]
                primary_roof_geom_area = gpd.GeoSeries([primary_feature.geometry_feature], crs=LAT_LONG_CRS).to_crs(area_crs)[0]

                # Get geodataframe of only the relevant features for buffering, and convert to area projection
                buffer_features_gdf = gpd.GeoDataFrame(
                    features_gdf[features_gdf.class_id.isin(BUFFER_CLASSES.values())],
                    geometry="geometry_feature",
                    crs=LAT_LONG_CRS
                    )
                buffer_features_gdf = buffer_features_gdf.to_crs(area_crs)

                # Calculate buffers for each distance
                for buffer_name, buffer_dist in BUFFER_ZONES_M.items():
                    # Create column names
                    parcel[f"primary_roof_{buffer_name}_zone_sqm"] = None
                    parcel[f"primary_roof_{buffer_name}_all_buffer_classes_sqm"] = None
                    for bc_name in BUFFER_CLASSES.keys():
                        parcel[f"primary_roof_{buffer_name}_{bc_name}_sqm"] = None

                for buffer_name, buffer_dist in BUFFER_ZONES_M.items():
                    # Create buffered regions around roofs
                    buffered_primary_roof_geom_area = primary_roof_geom_area.buffer(buffer_dist)

                    if not buffered_primary_roof_geom_area.within(parcel_geom_area):
                        # Skip if the buffer protrudes outside the parcel
                        continue

                    # Proceed calculating intersections etc. with the buffered primary roof
                    parcel[f"primary_roof_{buffer_name}_zone_sqm"] = buffered_primary_roof_geom_area.area

                    # Trim buffer features to only the buffer zone, and drop any that are empty
                    bf_trimmed_gdf = buffer_features_gdf.copy()
                    bf_trimmed_gdf["geometry_feature"] = bf_trimmed_gdf.intersection(buffered_primary_roof_geom_area)
                    bf_trimmed_gdf = bf_trimmed_gdf[~bf_trimmed_gdf.is_empty]

                    # Calculate total area of union of all buffer features
                    buffer_union = bf_trimmed_gdf.union_all()
                    parcel[f"primary_roof_{buffer_name}_all_buffer_classes_sqm"] = buffer_union.area

                    # TODO: Figure out what to do about the interior of the primary roof - is it part of the zone?
                    # TODO: Are zones exclusive rings, or is each just larger than the previous?

                    for bc_name, bc_id in BUFFER_CLASSES.items():
                        bc_gdf = bf_trimmed_gdf[bf_trimmed_gdf.class_id == bc_id]

                        # Calculate union of all features of this class before getting area
                        bc_union = bc_gdf.union_all()
                        parcel[f"primary_roof_{buffer_name}_{bc_name}_sqm"] = bc_union.area

            elif class_id == BUILDING_LIFECYCLE_ID:
                # Add aggregated damage across whole parcel, weighted by building lifecycle area
                # TODO: Finish this.
                pass
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
    assert parcels_gdf.index.name == AOI_ID_COLUMN_NAME
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
            calc_buffers=calc_buffers,
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
            rollup_df[col] = rollup_df[col].round(2)
            logger.debug("Rounded column %s to 2 decimal places", col)
    return rollup_df
