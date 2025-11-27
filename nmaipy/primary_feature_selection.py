"""
Primary Feature Selection Utilities

This module provides generic algorithms for selecting the "primary" (most important)
feature from a collection of features within an AOI. These algorithms are used across
different API clients (Feature API, Roof Age API) to provide consistent rollup behavior.

Selection Methods:
- "largest": Select feature with largest area (clipped or unclipped)
- "nearest": Select feature nearest to a target point, preferring high-confidence features

The selection logic is independent of the specific feature type (roof, building, roof instance, etc.)
and operates purely on geometric and attribute criteria.
"""
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from nmaipy import log

logger = log.get_logger()

# Default high confidence threshold for nearest selection
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.9


def select_primary_by_largest(
    gdf: gpd.GeoDataFrame,
    area_col: str = "area",
    secondary_area_col: Optional[str] = None,
) -> pd.Series:
    """
    Select the primary feature based on largest area.

    Args:
        gdf: GeoDataFrame containing features to select from
        area_col: Name of the primary area column to sort by
        secondary_area_col: Optional secondary area column for tie-breaking
                           (e.g., use unclipped area if clipped areas are equal)

    Returns:
        pandas Series representing the selected primary feature (single row)

    Raises:
        ValueError: If gdf is empty or area_col not found

    Example:
        >>> roofs_gdf = gpd.GeoDataFrame({...})  # Multiple roofs for one property
        >>> primary_roof = select_primary_by_largest(roofs_gdf, area_col="area")
    """
    if len(gdf) == 0:
        raise ValueError("Cannot select primary feature from empty GeoDataFrame")

    if area_col not in gdf.columns:
        raise ValueError(f"Area column '{area_col}' not found in GeoDataFrame. Available columns: {list(gdf.columns)}")

    # Sort by primary area column (descending)
    sort_cols = [area_col]
    sort_ascending = [False]

    # Add secondary sort column if provided and exists
    if secondary_area_col and secondary_area_col in gdf.columns:
        sort_cols.append(secondary_area_col)
        sort_ascending.append(False)

    sorted_gdf = gdf.sort_values(sort_cols, ascending=sort_ascending)

    return sorted_gdf.iloc[0]


def select_primary_by_nearest(
    gdf: gpd.GeoDataFrame,
    target_lat: float,
    target_lon: float,
    confidence_col: Optional[str] = "confidence",
    high_confidence_threshold: float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    geometry_col: str = "geometry",
) -> pd.Series:
    """
    Select the primary feature based on proximity to a target point.

    If a confidence column is provided, high-confidence features (confidence >= threshold)
    are preferred. If multiple high-confidence features exist, the nearest one is selected.
    If no high-confidence features exist, the nearest feature overall is selected.

    Args:
        gdf: GeoDataFrame containing features to select from (must be in EPSG:4326)
        target_lat: Target latitude (EPSG:4326)
        target_lon: Target longitude (EPSG:4326)
        confidence_col: Optional name of confidence column (0-1 scale). Set to None to disable.
        high_confidence_threshold: Threshold for "high confidence" (default: 0.9)
        geometry_col: Name of geometry column to use for distance calculation

    Returns:
        pandas Series representing the selected primary feature (single row)

    Raises:
        ValueError: If gdf is empty or geometries are invalid

    Example:
        >>> roofs_gdf = gpd.GeoDataFrame({...})  # Multiple roofs near a geocoded point
        >>> primary_roof = select_primary_by_nearest(
        ...     roofs_gdf, target_lat=37.7749, target_lon=-122.4194,
        ...     confidence_col="trustScore", high_confidence_threshold=0.8
        ... )

    Notes:
        - Input geometries must be in EPSG:4326 (WGS84)
        - Distance calculations are performed in EPSG:3857 (Web Mercator) for accuracy
        - If no confidence_col is provided, simply selects the nearest feature
    """
    if len(gdf) == 0:
        raise ValueError("Cannot select primary feature from empty GeoDataFrame")

    # Create target point in EPSG:4326, then convert to EPSG:3857 for distance calculation
    target_point = Point(target_lon, target_lat)
    target_point_3857 = gpd.GeoSeries([target_point], crs="EPSG:4326").to_crs("EPSG:3857")[0]

    # Handle geometry column - may be named differently (e.g., "geometry_feature")
    if geometry_col not in gdf.columns:
        # Try to find an appropriate geometry column
        if "geometry" in gdf.columns:
            geometry_col = "geometry"
        else:
            # Look for any geometry column
            geom_cols = [c for c in gdf.columns if "geometry" in c.lower()]
            if geom_cols:
                geometry_col = geom_cols[0]
                logger.debug(f"Using geometry column: {geometry_col}")
            else:
                raise ValueError(f"No valid geometry column found. Available columns: {list(gdf.columns)}")

    # Convert features to EPSG:3857 for distance calculation
    # Check if the GeoDataFrame has an active geometry and if it matches what we need
    try:
        active_geom_name = gdf.geometry.name
        gdf_for_distance = gdf if active_geom_name == geometry_col else gdf.set_geometry(geometry_col)
    except AttributeError:
        # No active geometry column, set it explicitly
        gdf_for_distance = gdf.set_geometry(geometry_col)

    # Ensure we have a valid CRS
    if gdf_for_distance.crs is None:
        logger.warning("GeoDataFrame has no CRS set, assuming EPSG:4326")
        gdf_for_distance = gdf_for_distance.set_crs("EPSG:4326")

    gdf_3857 = gdf_for_distance.to_crs("EPSG:3857")

    # Calculate distances
    distances = gdf_3857.geometry.distance(target_point_3857)

    # Filter to high-confidence features if confidence column provided and exists
    if confidence_col and confidence_col in gdf.columns:
        high_conf_mask = gdf[confidence_col] >= high_confidence_threshold

        if high_conf_mask.any():
            # Select nearest high-confidence feature
            high_conf_distances = distances[high_conf_mask]
            nearest_idx = high_conf_distances.idxmin()
            logger.debug(
                f"Selected high-confidence feature (>= {high_confidence_threshold}) "
                f"at distance {distances[nearest_idx]:.1f}m"
            )
        else:
            # No high-confidence features, select nearest overall
            nearest_idx = distances.idxmin()
            logger.debug(
                f"No high-confidence features found, selected nearest feature "
                f"at distance {distances[nearest_idx]:.1f}m"
            )
    else:
        # No confidence filtering, select nearest
        nearest_idx = distances.idxmin()
        logger.debug(f"Selected nearest feature at distance {distances[nearest_idx]:.1f}m")

    return gdf.loc[nearest_idx]


def select_primary(
    gdf: gpd.GeoDataFrame,
    method: str = "largest",
    area_col: str = "area",
    secondary_area_col: Optional[str] = None,
    target_lat: Optional[float] = None,
    target_lon: Optional[float] = None,
    confidence_col: Optional[str] = "confidence",
    high_confidence_threshold: float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    geometry_col: str = "geometry",
) -> pd.Series:
    """
    Convenience function to select primary feature using specified method.

    This is the main entry point for primary feature selection. It dispatches
    to the appropriate selection algorithm based on the method parameter.

    Args:
        gdf: GeoDataFrame containing features to select from
        method: Selection method - "largest" or "nearest"
        area_col: Area column name (for "largest" method)
        secondary_area_col: Optional secondary area column (for "largest" method)
        target_lat: Target latitude (for "nearest" method)
        target_lon: Target longitude (for "nearest" method)
        confidence_col: Confidence column name (for "nearest" method)
        high_confidence_threshold: High confidence threshold (for "nearest" method)
        geometry_col: Geometry column name (for "nearest" method)

    Returns:
        pandas Series representing the selected primary feature

    Raises:
        ValueError: If method is invalid or required parameters are missing

    Example:
        >>> # Select largest roof by clipped area
        >>> primary = select_primary(
        ...     roofs_gdf, method="largest",
        ...     area_col="clipped_area_sqm", secondary_area_col="unclipped_area_sqm"
        ... )
        >>>
        >>> # Select nearest high-confidence roof instance
        >>> primary = select_primary(
        ...     roof_instances_gdf, method="nearest",
        ...     target_lat=37.7749, target_lon=-122.4194,
        ...     confidence_col="trustScore"
        ... )
    """
    if len(gdf) == 0:
        raise ValueError("Cannot select primary feature from empty GeoDataFrame")

    if method == "largest" or method == "largest_intersection":
        return select_primary_by_largest(gdf, area_col, secondary_area_col)

    elif method == "nearest":
        if target_lat is None or target_lon is None:
            raise ValueError(
                "For 'nearest' method, must provide both target_lat and target_lon"
            )
        return select_primary_by_nearest(
            gdf, target_lat, target_lon, confidence_col, high_confidence_threshold, geometry_col
        )

    else:
        raise ValueError(
            f"Invalid selection method '{method}'. Must be 'largest', 'largest_intersection', or 'nearest'"
        )
