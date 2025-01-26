import json
import time
import warnings
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon
from tqdm import tqdm

import nmaipy.reference_code
from nmaipy import log
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
        CLASS_1013_HIP: 0.50,  # Keep fixed at 0.50 due to rate filing
        CLASS_1014_GABLE: 0.50,  # Keep fixed at 0.50 due to rate filing
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
        CLASS_1008_ASPHALT_SHINGLE: 0.50,  # Keep fixed at 0.50 due to rate filing
        CLASS_1009_METAL_PANEL: 0.50,  # Keep fixed at 0.50 due to rate filing
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
        BUILDING_LIFECYCLE_ID: 0,  # Point classes have no clipped area, so we can't filter them out based on area.
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

BUFFER_UNION_CLASSES = {
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
            logger.warning(f"Missing {AOI_ID_COLUMN_NAME} column in parcel data - generating unique IDs")
            parcels_gdf.index.name = id_column  # Set a new unique ordered index for reference
        else:  # The index must already be there as a column
            logger.warning(f"Moving {AOI_ID_COLUMN_NAME} to be the index - generating unique IDs")
            parcels_gdf = parcels_gdf.set_index(id_column)
    if parcels_gdf.index.duplicated().any():
        raise ValueError(f"Duplicate IDs found for {id_column=}")
    return parcels_gdf


def filter_features_in_parcels(
    features_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame, region: str, clip_multiparcel_buildings: bool = True
) -> gpd.GeoDataFrame:
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
    if features_gdf is None:
        return features_gdf
    elif len(features_gdf) == 0:
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
    gdf["intersection_ratio"] = gdf["intersection_ratio"].fillna(
        1
    )  # If unclipped area is zero, assume the feature is fully inside the parcel

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
        return gdf

    # Keep track of the AOIs we applied multiparcel building clipping to
    aois_with_clipped_buildings = []

    # Filter out buildings that are not in the AOI, and clip geometries of multiparcel buildings
    for aoi_id in tqdm(
        gdf.index.unique(), desc="Filter buildings not in AOI and clip geometries of multiparcel buildings"
    ):  # Loop over each AOI in the set
        gdf_aoi = gdf.loc[[aoi_id]]
        gdf_aoi_buildings = gdf_aoi[gdf_aoi.class_id.isin(building_style_ids)]
        if len(gdf_aoi_buildings) == 0:
            continue  # Skip if there are no buildings in the AOI
        features_from_aoi = aoi_gdf.loc[[aoi_id]]
        aoi_poly = (
            features_from_aoi.to_crs(AREA_CRS[region]).iloc[0].geometry
        )  # Get in metric projection for area/geospatial calcs in metres.
        building_statuses = []
        for building_poly in gdf_aoi_buildings.to_crs(AREA_CRS[region]).geometry:  # Loop through buildings in the AOI
            building_status = nmaipy.reference_code.get_building_status(building_poly, aoi_poly)
            building_statuses.append(building_status)
        building_statuses = pd.DataFrame(building_statuses)
        building_statuses.index = gdf_aoi_buildings.index
        gdf_aoi_buildings = pd.concat(
            [gdf_aoi_buildings, building_statuses], axis=1
        )  # Append extra columns for all buildings in this parcel
        gdf_aoi_buildings = gdf_aoi_buildings[gdf_aoi_buildings.building_keep].drop(
            columns=["building_keep"]
        )  # Remove any we should filter out

        if clip_multiparcel_buildings:
            # Clip any building that is "multiparcel" to the intersection with the AOI
            multiparcel_mask = gdf_aoi_buildings["building_multiparcel"]
            if multiparcel_mask.any():
                aoi_poly_api_crs = features_from_aoi.iloc[0].geometry
                new_geometries = gdf_aoi_buildings[multiparcel_mask].intersection(aoi_poly_api_crs).geometry
                gdf_aoi_buildings.loc[multiparcel_mask, "geometry"] = new_geometries

                aois_with_clipped_buildings.append(aoi_id)

        out_gdf_building_style.append(gdf_aoi_buildings)

    if len(out_gdf_building_style) > 0:
        out_gdf_building_style = pd.concat(out_gdf_building_style)
        gdf = pd.concat([gdf_non_building_style, out_gdf_building_style])
    else:
        gdf = gdf_non_building_style

    # Filter all featuers based on area and ratio in parcel
    area_mask = gdf.class_id.map(DEFAULT_FILTERING["min_area_in_parcel"]).fillna(0) <= gdf["clipped_area_" + suffix]
    ratio_mask = gdf.class_id.map(DEFAULT_FILTERING["min_ratio_in_parcel"]).fillna(0) <= gdf.intersection_ratio
    gdf = gdf[area_mask & ratio_mask]

    # Only drop objects where the parent has been explicitly removed as above (otherwise we drop solar panels without a building request, etc.)
    feature_ids_removed = set(features_gdf.feature_id) - set(gdf.feature_id)
    parent_removed = gdf.parent_id.isin(feature_ids_removed)
    no_parent = (gdf.parent_id == "") | gdf.parent_id.isna()
    gdf = gdf.loc[~parent_removed | no_parent]

    # For features with a parent, do an intersection to reduce them to the area of the parent, in the case of multiparcel buildings
    has_parent = ~((gdf.parent_id == "") | gdf.parent_id.isna())
    idx_cols = [AOI_ID_COLUMN_NAME, "feature_id"]
    features_to_update = gdf[has_parent].copy().reset_index().set_index(idx_cols)

    # Wherever a parent exists in the same AOI, identify the parent geometry as a column next to the child feature "geometry".
    gdf_parent_lookup = (
        features_to_update["geometry"]
        .reset_index()
        .rename(columns={"feature_id": "parent_id", "geometry": "parent_geometry"})
    )
    features_to_update = (
        features_to_update.reset_index()
        .merge(gdf_parent_lookup, on=["aoi_id", "parent_id"], how="left")
        .set_index(idx_cols)
    )

    # Update our knowledge of which features actually have a parent (as some may have a parent ID that isn't in the dataframe)
    has_parent = features_to_update.parent_geometry.notna()

    # Update geometry to the intersection of "geometry" and "parent_geometry" for all features with a parent - otherwise leave as is
    features_to_update.loc[has_parent, "geometry"] = features_to_update[has_parent]["geometry"].intersection(
        gpd.GeoSeries(features_to_update[has_parent]["parent_geometry"], crs=features_to_update.crs), align=False
    )

    new_area = features_to_update.geometry.to_crs(AREA_CRS[region]).area

    # Recalculate areas for clipped features
    if "clipped_area_sqm" in gdf.columns:
        features_to_update.loc[has_parent, "clipped_area_sqm"] = new_area
    if "clipped_area_sqft" in gdf.columns:
        features_to_update.loc[has_parent, "clipped_area_sqft"] = new_area * METERS_TO_FEET * METERS_TO_FEET

    # Update gdf with the new information, from rows matching AOI_ID_COLUMN_NAME and feature_id
    gdf = gdf.reset_index().set_index(idx_cols)
    assert not gdf.index.has_duplicates
    assert not features_to_update.index.has_duplicates
    gdf.update(features_to_update)
    gdf = gdf.reset_index().set_index(AOI_ID_COLUMN_NAME)

    # Get all of the structural damage composite features
    all_damage_gdf = gdf[gdf["class_id"] == CLASS_1186_STRUCTURAL_DAMAGE]

    # Keep track of the AOIs that need their attributes updated
    aois_with_structural_damage = []

    # Iterate over the AOIs with structural damage composite features
    for aoi_id in tqdm(all_damage_gdf.index.unique(), desc="Update structural damage composite features"):
        # Get the structural damage composite features in this AOI
        # NOTE: Need the inside brackets to ensure that we are always returning a dataframe for downstream operations
        aoi_damage_gdf = all_damage_gdf.loc[[aoi_id]]

        # Skip this AOI if the total structrual damage area is 0
        if aoi_damage_gdf["clipped_area_sqm"].sum() == 0:
            continue

        # Add this AOI to the list of AOIs that need their attributes updated
        aois_with_structural_damage.append(aoi_id)

        # Filter for only the features in this AOI
        # NOTE: This was needed just in case some features span multiple AOIs
        aoi_gdf = gdf.loc[aoi_id]
        aoi_roof_gdf = aoi_gdf[aoi_gdf["class_id"] == ROOF_ID]

        # Get the building lifecycle features in this AOI with structural damage composite children
        lifecycle_gdf = aoi_gdf[aoi_gdf["feature_id"].isin(aoi_damage_gdf["parent_id"])]

        # Iterate over the building lifecycle features with structural damage composite children
        for lifecycle_row in lifecycle_gdf.itertuples():
            # Get the structural damage composite features that are children of this building lifecycle feature
            damage_gdf = aoi_damage_gdf[aoi_damage_gdf["parent_id"] == lifecycle_row.feature_id]

            # Get the roofs that intersect with this building lifecycle feature
            # Reproject both roof rows and building lifecycle rows to a projected CRS (EPSG:3857) for the intersection
            lifecycle_geometry = gpd.GeoSeries(lifecycle_row.geometry, crs="EPSG:4326")
            aoi_roof_gdf_3857 = aoi_roof_gdf.to_crs("EPSG:3857")
            lifecycle_geometry_3857 = lifecycle_geometry.to_crs("EPSG:3857").unary_union
            intersecting_roof_gdf = aoi_roof_gdf[aoi_roof_gdf_3857["geometry"].intersects(lifecycle_geometry_3857)]

            if intersecting_roof_gdf.empty:
                # If there are no roof rows that intersect
                roof_covers_damage = False
            else:
                # Do these intersecting roofs fully contain the structural damage composite features inside of their geometries?
                roof_covers_damage = (
                    intersecting_roof_gdf["geometry"]
                    .to_crs("EPSG:3857")
                    .unary_union.contains(damage_gdf["geometry"].to_crs("EPSG:3857").unary_union)
                )

            # If the roofs fully cover the structural damage composite features, then we will use the roof geometries
            # and not the building lifecycle geometries since they look better.
            # We just need to update the roof structural damage attributes so the rollup gets calculated correctly
            if roof_covers_damage:
                # Iterate over the roofs
                for roof_row in intersecting_roof_gdf.itertuples():
                    # Get the structural damage composite features that intersect with this roof
                    roof_geometry = gpd.GeoSeries(roof_row.geometry, crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
                    intersecting_damage_gdf = damage_gdf[
                        damage_gdf["geometry"].to_crs("EPSG:3857").intersects(roof_geometry)
                    ]

                    # Change the parent ID of these structural damage composite features to the roof feature ID
                    # NOTE: This will allow for the attributes to get calculated correctly in the rollup
                    gdf.loc[
                        (gdf.index == aoi_id) & (gdf["feature_id"].isin(intersecting_damage_gdf["feature_id"])),
                        "parent_id",
                    ] = roof_row.feature_id

                # Remove this building lifecycle to avoid confusion downstream
                gdf = gdf[
                    ~(
                        (gdf.index == aoi_id)
                        & (gdf["feature_id"] == lifecycle_row.feature_id)
                        & (gdf["class_id"] == BUILDING_LIFECYCLE_ID)
                    )
                ]
            else:
                # If the roofs do not fully cover the structural damage composite features, then we will replace the
                # the roof features with the building lifecycle feature

                # Update the building lifecycle feature to be a roof feature
                lifecycle_mask = (gdf.index == aoi_id) & (gdf["feature_id"] == lifecycle_row.feature_id)
                gdf.loc[lifecycle_mask, "class_id"] = ROOF_ID
                gdf.loc[lifecycle_mask, "internal_class_id"] = 1002
                gdf.loc[lifecycle_mask, "description"] = "Roof"

                if intersecting_roof_gdf.empty:
                    # If there are no roofs that intersect, then manually create the roof attributes with just the structural damage
                    # NOTE: Just needs a placeholder so the component can be updated in the next step for the rollup

                    # Read in the empty roof attributes from a json file
                    current_dir = Path(__file__).parent
                    roof_attributes_file_path = current_dir / "empty_roof_attributes.json"
                    with open(roof_attributes_file_path) as f:
                        empty_roof_attributes = json.load(f)
                    # Copy the empty roof attributes over to the building lifecycle feature
                    gdf.loc[lifecycle_mask, "attributes"] = gdf.loc[lifecycle_mask].apply(
                        lambda _: empty_roof_attributes, axis=1
                    )
                else:
                    # Copy the roof attributes from one of the roofs over to the building lifecycle feature
                    # NOTE: Doesn't matter which one we copy over as these will get updated in the next step
                    # NOTE: Index (aoi_id is not unique at this point so need to use nlargest)
                    largest_intersecting_roof = intersecting_roof_gdf.nlargest(1, "clipped_area_sqm")
                    gdf.loc[lifecycle_mask, "attributes"] = largest_intersecting_roof["attributes"]

                    # Update the parent ID of all intersecting roof child features to the building lifecycle feature ID
                    # NOTE: This is needed so that the rollup gets calculated correctly
                    intersecting_roof_feature_ids = intersecting_roof_gdf["feature_id"]
                    gdf.loc[gdf["parent_id"].isin(intersecting_roof_feature_ids), "parent_id"] = (
                        lifecycle_row.feature_id
                    )

                    # Remove all of the intersecting roof features (these have been replaced by the building lifecycle feature)
                    gdf = gdf[~((gdf.index == aoi_id) & (gdf["feature_id"].isin(intersecting_roof_feature_ids)))]

    print("Start updating attributes for clipped buildings and structural damage composite features")
    start_time = time.time()

    # Get all of the features that have attributes
    # NOTE: For efficiency reasons, we will limit the AOIs that either have multiparcel buildings that were clipped or have
    # structural damage composite features that need their attributes updated. Also, we will only update the roof attributes
    # for the retro pipeline.
    aois_needing_updates = set(aois_with_clipped_buildings + aois_with_structural_damage)
    roofs_needing_updates_df = gdf[
        gdf.index.isin(aois_needing_updates) & (gdf["class_id"] == ROOF_ID) & (gdf["attributes"].astype(bool))
    ]

    # Get all of the children of the roofs that need their attributes updated
    children_needing_updates_df = gdf[
        gdf.index.isin(aois_needing_updates) & (gdf["parent_id"].isin(roofs_needing_updates_df["feature_id"]))
    ]

    # Expand the children dataframe to include the clipped area of the parent roof needed for the calculations
    expanded_children_df = children_needing_updates_df.reset_index().merge(
        roofs_needing_updates_df.reset_index()[["aoi_id", "feature_id", "clipped_area_sqm"]].rename(
            columns={
                "aoi_id": "roof_aoi_id",
                "feature_id": "roof_feature_id",
                "clipped_area_sqm": "roof_clipped_area_sqm",
            }
        ),
        how="left",
        left_on=["aoi_id", "parent_id"],
        right_on=["roof_aoi_id", "roof_feature_id"],
    )

    # Group the children dataframe by the parent ID and class ID and calculate the aggregates
    grouped_children_df = expanded_children_df.groupby(["aoi_id", "parent_id", "class_id"]).agg(
        internal_class_id=("internal_class_id", "first"),
        child_description=("description", "first"),
        weighted_confidence_numerator=(
            "confidence",
            lambda x: np.sum(x * expanded_children_df.loc[x.index, "clipped_area_sqm"]),
        ),
        clipped_area_sqm=("clipped_area_sqm", "sum"),
        clipped_area_sqft=("clipped_area_sqft", "sum"),
        roof_clipped_area_sqm=(
            "roof_clipped_area_sqm",
            "first",
        ),  # Use first value of `parent_clipped_area_sqm` (assuming it's constant per group)
    )

    # Calculate the area-weighted confidence score and handle division by zero
    grouped_children_df["weighted_confidence"] = (
        grouped_children_df["weighted_confidence_numerator"] / grouped_children_df["clipped_area_sqm"]
    ).where(grouped_children_df["clipped_area_sqm"] > 0, np.nan)

    # Calculate the ratio, handling division by zero
    grouped_children_df["ratio"] = (
        grouped_children_df["clipped_area_sqm"] / grouped_children_df["roof_clipped_area_sqm"]
    ).where(grouped_children_df["roof_clipped_area_sqm"] > 0, 0)

    # Convert the description_to_class_ids mapping to a DataFrame
    description_to_class_ids = {
        "Roof material": {
            "parent_info": {
                "parent_class_id": "89c7d478-58de-56bd-96d2-e71e27a36905",
                "parent_internal_class_id": 3,
            },
            "children": [
                {
                    "class_id": "516fdfd5-0be9-59fe-b849-92faef8ef26e",
                    "internal_class_id": 1007,
                    "child_description": "Tile",
                },
                {
                    "class_id": "4bbf8dbd-cc81-5773-961f-0121101422be",
                    "internal_class_id": 1008,
                    "child_description": "Shingle",
                },
                {
                    "class_id": "4424186a-0b42-5608-a5a0-d4432695c260",
                    "internal_class_id": 1009,
                    "child_description": "Metal",
                },
                {
                    "class_id": "4558c4fb-3ddf-549d-b2d2-471384be23d1",
                    "internal_class_id": 1100,
                    "child_description": "Ballasted",
                },
                {
                    "class_id": "87437e20-d9f5-57e1-8b87-4a9c81ec3b65",
                    "internal_class_id": 1101,
                    "child_description": "Mod-Bit",
                },
                {
                    "class_id": "383930f1-d866-5aa3-9f97-553311f3162d",
                    "internal_class_id": 1103,
                    "child_description": "PVC/TPO",
                },
                {
                    "class_id": "64db6ea0-7248-53f5-b6a6-6ed733c5f9b8",
                    "internal_class_id": 1104,
                    "child_description": "EPDM",
                },
                {
                    "class_id": "9fc4c92e-4405-573e-bce6-102b74ab89a3",
                    "internal_class_id": 1105,
                    "child_description": "Wood Shake",
                },
                {
                    "class_id": "09ed6bf9-182a-5c79-ae59-f5531181d298",
                    "internal_class_id": 1160,
                    "child_description": "Clay Tile",
                },
                {
                    "class_id": "cdc50dcc-e522-5361-8f02-4e30673311bb",
                    "internal_class_id": 1163,
                    "child_description": "Slate",
                },
                {
                    "class_id": "3563c8f1-e81e-52c7-bd56-eaa937010403",
                    "internal_class_id": 1165,
                    "child_description": "Built Up",
                },
                {
                    "class_id": "b2573072-b3a5-5f7c-973f-06b7649665ff",
                    "internal_class_id": 1168,
                    "child_description": "Roof Coating",
                },
            ],
        },
        "Roof types": {
            "parent_info": {
                "parent_class_id": "20a58db2-bc02-531d-98f5-451f88ce1fed",
                "parent_internal_class_id": 4,
            },
            "children": [
                {
                    "class_id": "ac0a5f75-d8aa-554c-8a43-cee9684ef9e9",
                    "internal_class_id": 1013,
                    "child_description": "Hip",
                },
                {
                    "class_id": "59c6e27e-6ef2-5b5c-90e7-31cfca78c0c2",
                    "internal_class_id": 1014,
                    "child_description": "Gable",
                },
                {
                    "class_id": "3719eb40-d6d1-5071-bbe6-379a551bb65f",
                    "internal_class_id": 1015,
                    "child_description": "Dutch Gable",
                },
                {
                    "class_id": "224f98d3-b853-542a-8b18-e1e46e3a8200",
                    "internal_class_id": 1016,
                    "child_description": "Flat (Deprecated)",
                },
                {
                    "class_id": "7ac62320-52f3-5301-94c5-7adf6b93a3b8",
                    "internal_class_id": 1018,
                    "child_description": "Shed",
                },
                {
                    "class_id": "4bb630b9-f9eb-5f95-85b8-f0c6caf16e9b",
                    "internal_class_id": 1019,
                    "child_description": "Gambrel",
                },
                {
                    "class_id": "89582082-e5b8-5853-bc94-3a0392cab98a",
                    "internal_class_id": 1020,
                    "child_description": "Turret",
                },
                {
                    "class_id": "1234ea84-e334-5c58-88a9-6554be3dfc05",
                    "internal_class_id": 1173,
                    "child_description": "Parapet",
                },
                {
                    "class_id": "7eb3b1b6-0d75-5b1f-b41c-b14146ff0c54",
                    "internal_class_id": 1174,
                    "child_description": "Mansard",
                },
                {
                    "class_id": "924afbab-aae6-5c26-92e8-9173e4320495",
                    "internal_class_id": 1176,
                    "child_description": "Jerkinhead",
                },
                {
                    "class_id": "e92bc8a2-9fa3-5094-b3b6-2881d94642ab",
                    "internal_class_id": 1178,
                    "child_description": "Quonset",
                },
                {
                    "class_id": "09b925d2-df1d-599b-89f1-3ffd39df791e",
                    "internal_class_id": 1180,
                    "child_description": "Bowstring Truss",
                },
                {
                    "class_id": "1ab60ef7-e770-5ab6-995e-124676b2be11",
                    "internal_class_id": 1191,
                    "child_description": "Flat",
                },
            ],
        },
        "Roof overhang attributes": {
            "parent_info": {
                "parent_class_id": "7ab56e15-d5d4-51bb-92bd-69e910e82e56",
                "parent_internal_class_id": 5,
            },
            "children": [
                {
                    "class_id": "8e9448bd-4669-5f46-b8f0-840fee25c34c",
                    "internal_class_id": 1045,
                    "child_description": "Tree Overhang",
                },
                {
                    "class_id": "042a1d14-4a23-50dc-aabb-befc9645af3b",
                    "internal_class_id": 1084,
                    "child_description": "Leaf-off Tree Overhang",
                },
                {
                    "class_id": "38c4dd92-868c-582a-a4d5-537c88dcec75",
                    "internal_class_id": 1085,
                    "child_description": "Low Vegetation (0.5m-2m) Overhang",
                },
                {
                    "class_id": "fcbb15ea-93e5-587c-8941-246353817741",
                    "internal_class_id": 1086,
                    "child_description": "Very Low Vegetation (<0.5m) Overhang",
                },
                {
                    "class_id": "1ef797a5-8057-5e8b-a24d-dc7cd8f1fa7b",
                    "internal_class_id": 1087,
                    "child_description": "Power Line Overhang",
                },
            ],
        },
        "Roof Condition": {
            "parent_info": {
                "parent_class_id": "3065525d-3f14-5b9d-8c4c-077f1ad5c694",
                "parent_internal_class_id": 10,
            },
            "children": [
                {
                    "class_id": "f907e625-26b3-59db-a806-d41f62ce1f1b",
                    "internal_class_id": 1049,
                    "child_description": "Structural Damage",
                },
                {
                    "class_id": "abb1f304-ce01-527b-b799-cbfd07551b2c",
                    "internal_class_id": 1050,
                    "child_description": "Roof with Temporary Repair",
                },
                {
                    "class_id": "f41e02b0-adc0-5b46-ac95-8c59aa9fe317",
                    "internal_class_id": 1051,
                    "child_description": "Roof Ponding",
                },
                {
                    "class_id": "526496bf-7344-5024-82d7-77ceb671feb4",
                    "internal_class_id": 1052,
                    "child_description": "Roof Rusting",
                },
                {
                    "class_id": "cfa8951a-4c29-54de-ae98-e5f804c305e3",
                    "internal_class_id": 1053,
                    "child_description": "Tile or Shingle Staining",
                },
                {
                    "class_id": "dec855e2-ae6f-56b5-9cbb-f9967ff8ca12",
                    "internal_class_id": 1079,
                    "child_description": "Missing Roof Tile or Shingle",
                },
                {
                    "class_id": "7218eb36-0d36-5b53-a2fe-6e99c7d950bc",
                    "internal_class_id": 1080,
                    "child_description": "Roof with Permanent Repair",
                },
                {
                    "class_id": "f55813f9-a39d-571d-9688-8d3f76aa35b9",
                    "internal_class_id": 1081,
                    "child_description": "Zinc Staining",
                },
                {
                    "class_id": "8ab218a7-8173-5f1e-a5cb-bb2cd386a73e",
                    "internal_class_id": 1139,
                    "child_description": "Roof Debris",
                },
                {
                    "class_id": "2905ba1c-6d96-58bc-9b1b-5911b3ead023",
                    "internal_class_id": 1140,
                    "child_description": "Exposed Roof Deck",
                },
                {
                    "class_id": "82b4547b-b8c0-5e9a-84c2-5c7564b4586c",
                    "internal_class_id": 1141,
                    "child_description": "Missing Asphalt shingles",
                },
                {
                    "class_id": "94944057-968c-5df3-828b-285091b7e266",
                    "internal_class_id": 1142,
                    "child_description": "Active Ponding",
                },
                {
                    "class_id": "319f552f-f4b7-520d-9b16-c8abb394b043",
                    "internal_class_id": 1144,
                    "child_description": "Roof Staining",
                },
                {
                    "class_id": "97a6f930-82ae-55f2-b856-635e2250af29",
                    "internal_class_id": 1146,
                    "child_description": "Worn Shingles",
                },
                {
                    "class_id": "2322ca41-5d3d-5782-b2b7-1a2ffd0c4b78",
                    "internal_class_id": 1147,
                    "child_description": "Exposed Underlayment",
                },
                {
                    "class_id": "8b30838b-af41-5d1d-bdbd-29e682fe3b00",
                    "internal_class_id": 1149,
                    "child_description": "Roof Patching",
                },
            ],
        },
    }

    # Flatten the dictionary into a DataFrame for parent and child information
    description_to_class_ids_df = pd.DataFrame(
        [
            {
                "description": description,
                "parent_class_id": data["parent_info"]["parent_class_id"],
                "parent_internal_class_id": data["parent_info"]["parent_internal_class_id"],
                "class_id": child["class_id"],
                "internal_class_id": child["internal_class_id"],
                "child_description": child["child_description"],
            }
            for description, data in description_to_class_ids.items()
            for child in data["children"]
        ]
    )

    # Create all possible combinations of (aoi_id, parent_id, description, class_id)
    unique_aoi_parent = grouped_children_df.index.droplevel("class_id").drop_duplicates()
    all_combinations = unique_aoi_parent.to_frame(index=False).merge(description_to_class_ids_df, how="cross")

    # Merge with the grouped_children_df and fill missing values
    expanded_df = all_combinations.merge(
        grouped_children_df.reset_index(),
        on=["aoi_id", "parent_id", "class_id", "internal_class_id", "child_description"],
        how="left",
    )

    # Fill missing values with defaults
    expanded_df["weighted_confidence"] = expanded_df["weighted_confidence"].fillna(np.nan)
    expanded_df["clipped_area_sqm"] = expanded_df["clipped_area_sqm"].fillna(0)
    expanded_df["clipped_area_sqft"] = expanded_df["clipped_area_sqft"].fillna(0)
    expanded_df["ratio"] = expanded_df["ratio"].fillna(0)

    # Reorganize the DataFrame
    final_df = expanded_df.sort_values(["aoi_id", "parent_id", "description", "class_id"]).set_index(
        ["aoi_id", "parent_id", "description", "class_id"]
    )

    def create_attributes(group):
        attributes = []
        for description, desc_group in group.groupby("description"):
            # Parent-specific info (assumes the same for all rows in the group)
            parent_class_id = desc_group["parent_class_id"].iloc[0]
            parent_internal_class_id = desc_group["parent_internal_class_id"].iloc[0]

            # Create the components list
            components = [
                {
                    "classId": row["class_id"],
                    "internalClassId": row["internal_class_id"],
                    "description": row["child_description"],
                    "confidence": row["weighted_confidence"],
                    "areaSqm": row["clipped_area_sqm"],
                    "areaSqft": row["clipped_area_sqft"],
                    "ratio": row["ratio"],
                }
                for _, row in desc_group.iterrows()  # Iterate explicitly over rows
            ]

            # Append the dictionary for this description
            attributes.append(
                {
                    "classId": parent_class_id,
                    "internalClassId": parent_internal_class_id,
                    "description": description,
                    "components": components,
                }
            )
        return attributes

    # Reset the index to ensure "class_id" and other columns are accessible as columns
    final_df_reset = final_df.reset_index()

    # Group by aoi_id and parent_id and apply the function
    result_df = final_df_reset.groupby(["aoi_id", "parent_id"]).apply(create_attributes).reset_index(name="attributes")

    # Rename parent_id to feature_id
    result_df = result_df.rename(columns={"parent_id": "feature_id"})

    # Based on the aoi_id and parent_id, update the attributes in the gdf
    gdf = gdf.merge(result_df, left_on=["aoi_id", "feature_id"], right_on=["aoi_id", "feature_id"], how="left")
    gdf["attributes"] = gdf["attributes_y"].combine_first(gdf["attributes_x"])
    gdf = gdf.drop(columns=["attributes_x", "attributes_y"])
    gdf = gdf.set_index("aoi_id")

    # Print the time taken to update the attributes
    print(f"Finished updating attributes in {time.time() - start_time:.2f} seconds")

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
            elif primary_decision == "nearest_no_filter":
                primary_point = Point(primary_lon, primary_lat)
                primary_point = gpd.GeoSeries(primary_point).set_crs("EPSG:4326").to_crs("EPSG:3857")[0]
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
                primary_roof_geom_area = gpd.GeoSeries([primary_feature.geometry_feature], crs=LAT_LONG_CRS).to_crs(
                    area_crs
                )[0]

                # Get geodataframe of only the relevant features for buffering, and convert to area projection
                buffer_features_gdf = gpd.GeoDataFrame(
                    features_gdf[features_gdf.class_id.isin(BUFFER_CLASSES.values())],
                    geometry="geometry_feature",
                    crs=LAT_LONG_CRS,
                )
                buffer_features_gdf = buffer_features_gdf.to_crs(area_crs)

                # Calculate buffers for each distance
                for buffer_name, buffer_dist in BUFFER_ZONES_M.items():
                    # Create column names
                    parcel[f"primary_roof_{buffer_name}_zone_sqm"] = None
                    parcel[f"primary_roof_{buffer_name}_buffer_union_classes_sqm"] = None
                    for bc_name in BUFFER_CLASSES.keys():
                        parcel[f"primary_roof_{buffer_name}_{bc_name}_sqm"] = None

                for buffer_name, buffer_dist in BUFFER_ZONES_M.items():
                    # Create buffered regions around roofs
                    buffered_primary_roof_geom_area = primary_roof_geom_area.buffer(buffer_dist)

                    if not buffered_primary_roof_geom_area.within(parcel_geom_area):
                        # logger.info(features_gdf)
                        # Skip if the buffer protrudes outside the parcel
                        # logger.info(
                        #     f"""skipping buffer calculation for aoi_id:feature_id {features_gdf["aoi_id"].tolist()[0]}, nmaipy feature_id {features_gdf["feature_id"].tolist()[0]}; buffer {buffer_dist} extends outside parcel."""
                        # )
                        # logger.info(gpd.GeoSeries([primary_feature.geometry_feature], crs=LAT_LONG_CRS).iloc[0])
                        continue

                    # Proceed calculating intersections etc. with the buffered primary roof
                    parcel[f"primary_roof_{buffer_name}_zone_sqm"] = buffered_primary_roof_geom_area.area

                    # Trim buffer features to only the buffer zone, and drop any that are empty
                    bf_trimmed_gdf = buffer_features_gdf.copy()
                    bf_trimmed_gdf["geometry_feature"] = bf_trimmed_gdf.intersection(buffered_primary_roof_geom_area)
                    bf_trimmed_gdf = bf_trimmed_gdf[~bf_trimmed_gdf.is_empty]

                    # Calculate total area of union of all buffer features from the BUFFER_UNION_CLASSES dictionary
                    buffer_union = bf_trimmed_gdf.query("class_id in @BUFFER_UNION_CLASSES.values()").union_all()
                    parcel[f"primary_roof_{buffer_name}_buffer_union_classes_sqm"] = buffer_union.area

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
    if primary_decision == "nearest" or primary_decision == "nearest_no_filter":
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
    for aoi_id, group in tqdm(df.reset_index().groupby(AOI_ID_COLUMN_NAME), desc="Processing AOI Rollups"):
        if primary_decision == "nearest" or primary_decision == "nearest_no_filter":
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
    for row in tqdm(
        parcels_gdf[~parcels_gdf.index.isin(features_gdf.index)].itertuples(), desc="Processing Empty AOIs"
    ):
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
    return rollup_df
