"""
Geometry utilities for geospatial operations across all API clients.

This module provides common geometric operations used by multiple API clients:
- Polygon manipulation and conversion
- Spatial clipping and intersection
- Grid creation and splitting for large AOIs
- Coordinate extraction for caching
"""
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from nmaipy.constants import API_CRS, AREA_CRS, AOI_ID_COLUMN_NAME, SQUARED_METERS_TO_SQUARED_FEET


def polygon_to_coordstring(poly: Polygon) -> str:
    """
    Convert a Shapely polygon to a comma-separated coordinate string.

    This format is used by various Nearmap APIs for polygon parameters.

    Args:
        poly: Shapely Polygon geometry

    Returns:
        Comma-separated string of flattened coordinates: "lon1,lat1,lon2,lat2,..."

    Example:
        >>> poly = Polygon([(-74.275, 40.642), (-74.274, 40.642), ...])
        >>> coordstring = polygon_to_coordstring(poly)
        >>> # Returns: "-74.275,40.642,-74.274,40.642,..."
    """
    coords = poly.exterior.coords[:]
    flat_coords = np.array(coords).ravel()
    coordstring = ",".join(flat_coords.astype(str))
    return coordstring


def clip_features_to_polygon(
    feature_poly_series: gpd.GeoSeries,
    geometry: Union[Polygon, MultiPolygon],
    region: str
) -> gpd.GeoDataFrame:
    """
    Clip feature geometries to a background polygon and recalculate areas.

    Takes feature polygons and clips them to a boundary geometry (e.g., parcel or AOI),
    then recalculates the clipped areas in both square meters and square feet.

    Args:
        feature_poly_series: GeoSeries of feature geometries to clip
        geometry: Background polygon to use as clipping mask
        region: Country/region code (e.g., 'us', 'au') for CRS selection

    Returns:
        GeoDataFrame with clipped geometries and updated area columns:
            - geometry: Clipped geometry
            - clipped_area_sqm: Clipped area in square meters (rounded to 1 decimal)
            - clipped_area_sqft: Clipped area in square feet (rounded to integer)
    """
    assert isinstance(feature_poly_series, gpd.GeoSeries)

    # Clip geometries to the boundary
    gdf_clip = gpd.GeoDataFrame(
        geometry=feature_poly_series.intersection(geometry),
        crs=feature_poly_series.crs
    )

    # Calculate clipped areas
    clipped_area_sqm = gdf_clip.to_crs(AREA_CRS[region]).area
    gdf_clip["clipped_area_sqft"] = (clipped_area_sqm * SQUARED_METERS_TO_SQUARED_FEET).round()
    gdf_clip["clipped_area_sqm"] = clipped_area_sqm.round(1)

    return gdf_clip


def extract_coords_for_cache(request_string: str) -> Tuple[str, str]:
    """
    Extract lon/lat coordinates from a request string for cache organization.

    Parses query parameters to find polygon coordinates and returns the first
    lon/lat pair as strings, rounded to integers. This is used to organize
    cache files into a directory structure.

    Args:
        request_string: URL request string with query parameters

    Returns:
        Tuple of (lon, lat) as strings, rounded to nearest integer

    Example:
        >>> url = "https://api.com?polygon=-74.275,40.642,-74.274,40.641"
        >>> lon, lat = extract_coords_for_cache(url)
        >>> # Returns: ("-74", "41")
    """
    # Extract query parameters
    r = request_string.split("?")[-1]
    dic = dict([token.split("=") for token in r.split("&")])

    # Get first lon/lat from polygon coordinates
    lon, lat = np.array(dic["polygon"].split(",")).astype("float").round().astype("int").astype("str")[:2]
    return lon, lat


def create_grid(df: gpd.GeoDataFrame, cell_size: float) -> gpd.GeoDataFrame:
    """
    Create a grid of square cells covering the extent of a GeoDataFrame.

    Generates a regular grid of square polygons that covers the total bounds
    of the input GeoDataFrame. Grid cells are in the same CRS as the input.

    Args:
        df: GeoDataFrame defining the extent to grid
        cell_size: Size of each grid cell (in CRS units, typically degrees)

    Returns:
        GeoDataFrame containing square polygon grid cells
    """
    minx, miny, maxx, maxy = df.total_bounds
    w, h = (cell_size, cell_size)

    rows = int(np.ceil((maxy - miny) / h))
    cols = int(np.ceil((maxx - minx) / w))

    polygons = []
    for i in range(cols):
        for j in range(rows):
            polygons.append(
                Polygon([
                    (minx + i * w, miny + (j + 1) * h),
                    (minx + (i + 1) * w, miny + (j + 1) * h),
                    (minx + (i + 1) * w, miny + j * h),
                    (minx + i * w, miny + j * h),
                ])
            )

    grid = gpd.GeoDataFrame({"geometry": polygons}, crs=df.crs)
    return grid


def split_geometry_into_grid(
    geometry: Union[Polygon, MultiPolygon],
    cell_size: float
) -> gpd.GeoDataFrame:
    """
    Split a large geometry into a grid of smaller cells.

    This is used to divide large AOIs into manageable chunks for parallel API requests.
    The geometry is overlaid with a grid, and only grid cells that intersect the
    geometry are returned.

    Args:
        geometry: Polygon or MultiPolygon to split (should be in API_CRS / EPSG:4326)
        cell_size: Size of grid cells in degrees

    Returns:
        GeoDataFrame where each row is a grid cell intersecting the geometry

    Example:
        >>> from shapely.geometry import box
        >>> large_aoi = box(-74.3, 40.6, -74.2, 40.7)  # ~10km square
        >>> grid = split_geometry_into_grid(large_aoi, cell_size=0.01)
        >>> # Returns grid of ~100 cells covering the AOI
    """
    # Create GeoDataFrame from geometry
    df = gpd.GeoDataFrame(geometry=[geometry], crs=API_CRS)

    # Create grid covering the extent
    df_gridded = create_grid(df, cell_size)

    # Overlay geometry with grid to get only intersecting cells
    df_gridded = gpd.overlay(df, df_gridded, keep_geom_type=True)

    # Explode any multipolygons into separate polygons
    # explicit index_parts added to avoid warning
    df_gridded = df_gridded.explode(index_parts=True)

    # Ensure result is in API CRS
    df_gridded = df_gridded.to_crs(API_CRS)

    return df_gridded


def combine_features_from_grid(
    features_gdf: gpd.GeoDataFrame,
    connected_class_ids: list = None
) -> gpd.GeoDataFrame:
    """
    Combine features from multiple grid cells, removing duplicates and merging connected features.

    When an AOI is split into a grid and queried separately, some features may appear in
    multiple grid cells (especially for connected/continuous features like vegetation).
    This function:
    1. Removes duplicate discrete features (same feature_id and geometry)
    2. Merges geometries of connected features that were split across grid cells
    3. Sums clipped areas for merged features

    Args:
        features_gdf: GeoDataFrame with features from gridded queries
                     Must have 'feature_id' column and AOI_ID_COLUMN_NAME index
        connected_class_ids: Optional list of class IDs that represent connected features
                           (currently not used, but available for future enhancement)

    Returns:
        GeoDataFrame with combined/deduplicated features, maintaining same structure

    Note:
        Returns empty GeoDataFrame with proper structure if input is None or empty.
    """
    # Handle empty or None input
    if features_gdf is None or len(features_gdf) == 0:
        empty_gdf = gpd.GeoDataFrame(columns=['geometry'], crs=API_CRS)
        empty_gdf.index.name = AOI_ID_COLUMN_NAME
        return empty_gdf

    # Columns that don't require aggregation (take first value)
    agg_cols_first = [
        AOI_ID_COLUMN_NAME,
        "class_id",
        "description",
        "confidence",
        "parent_id",
        "unclipped_area_sqm",
        "unclipped_area_sqft",
        "attributes",
        "damage",
        "belongs_to_parcel",
        "survey_date",
        "mesh_date",
        "fidelity",
    ]

    # Columns with clipped areas that should be summed when geometries are merged
    agg_cols_sum = [
        "area_sqm",
        "area_sqft",
        "clipped_area_sqm",
        "clipped_area_sqft",
    ]

    # Filter to only existing columns
    existing_agg_cols_first = [col for col in agg_cols_first if col in features_gdf.columns]
    existing_agg_cols_sum = [col for col in agg_cols_sum if col in features_gdf.columns]

    # Remove duplicate geometries, then dissolve remaining features by feature_id
    features_gdf_dissolved = (
        features_gdf
        .drop_duplicates(["feature_id", "geometry"])
        .filter(existing_agg_cols_first + ["geometry", "feature_id"], axis=1)
        .dissolve(by="feature_id", aggfunc="first")
        .reset_index()
        .set_index("feature_id")
    )

    # Sum clipped areas for merged features
    features_gdf_summed = (
        features_gdf
        .filter(existing_agg_cols_sum + ["feature_id"], axis=1)
        .groupby("feature_id")
        .aggregate(dict([c, "sum"] for c in existing_agg_cols_sum))
    )

    # Join dissolved geometries with summed areas
    features_gdf_out = (
        features_gdf_dissolved
        .join(features_gdf_summed)
        .reset_index()
        .set_index(AOI_ID_COLUMN_NAME)
    )

    return features_gdf_out
