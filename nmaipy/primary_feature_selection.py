"""
Primary Feature Selection Utilities

This module provides generic algorithms for selecting the "primary" (most important)
feature from a collection of features within an AOI. These algorithms are used across
different API clients (Feature API, Roof Age API) to provide consistent rollup behavior.

Selection Methods:
- "largest": Select feature with largest area (clipped or unclipped)
- "nearest": Select feature nearest to a target point, preferring non-small features
             that contain or are very close to the target point
- "optimal": Try nearest method first, fall back to largest if no suitable feature found

The selection logic is independent of the specific feature type (roof, building, roof instance, etc.)
and operates purely on geometric and attribute criteria.
"""
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from nmaipy import log
from nmaipy.constants import NEAREST_TOLERANCE_METERS
from nmaipy.reference_code import BUILDING_SMALL_MAX_AREA_SQM

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
    area_col: str = "area",
    small_area_threshold: float = BUILDING_SMALL_MAX_AREA_SQM,
    distance_tolerance: float = NEAREST_TOLERANCE_METERS,
) -> Optional[pd.Series]:
    """
    Select the primary feature based on containment or proximity to a target point.

    Selection logic (in order of priority):
    1. If target point falls within a non-small feature (area > small_area_threshold),
       that feature is selected as primary.
    2. Else, if the nearest non-small feature is within distance_tolerance meters,
       that feature is selected as primary.
    3. Otherwise, returns None (no suitable primary found by nearest method).

    Within each step, high-confidence features are preferred if confidence_col is provided.

    Args:
        gdf: GeoDataFrame containing features to select from (must be in EPSG:4326)
        target_lat: Target latitude (EPSG:4326)
        target_lon: Target longitude (EPSG:4326)
        confidence_col: Optional name of confidence column (0-1 scale). Set to None to disable.
        high_confidence_threshold: Threshold for "high confidence" (default: 0.9)
        geometry_col: Name of geometry column to use for containment/distance calculation
        area_col: Name of area column for filtering small features (in square meters)
        small_area_threshold: Features at or below this area are considered "small" and
                             deprioritized (default: 30 sqm from BUILDING_SMALL_MAX_AREA_SQM)
        distance_tolerance: Maximum distance in meters to consider a feature as "nearest"
                           primary (default: 1m from NEAREST_TOLERANCE_METERS)

    Returns:
        pandas Series representing the selected primary feature, or None if no suitable
        feature found by the nearest/containment criteria.

    Raises:
        ValueError: If gdf is empty or geometries are invalid

    Example:
        >>> roofs_gdf = gpd.GeoDataFrame({...})  # Multiple roofs near a geocoded point
        >>> primary_roof = select_primary_by_nearest(
        ...     roofs_gdf, target_lat=37.7749, target_lon=-122.4194,
        ...     confidence_col="trustScore", area_col="area"
        ... )
        >>> if primary_roof is None:
        ...     # Fall back to largest method
        ...     primary_roof = select_primary_by_largest(roofs_gdf, area_col="area")

    Notes:
        - Input geometries must be in EPSG:4326 (WGS84)
        - Distance calculations are performed in EPSG:3857 (Web Mercator) for accuracy
        - "Small" features (e.g., sheds, outbuildings) are excluded from primary selection
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

    # Filter to non-small features based on area
    # Prefer pre-computed area column (more accurate), fall back to geometry-based calculation
    if area_col in gdf.columns:
        non_small_mask = gdf[area_col] > small_area_threshold
    else:
        # Calculate area from geometry in EPSG:3857
        # Note: EPSG:3857 has area distortion (~20-30% at mid-latitudes), but acceptable
        # for the small building threshold check. For accurate areas, use pre-computed column.
        logger.debug(f"Area column '{area_col}' not found, calculating area from geometry (EPSG:3857)")
        areas_sqm = gdf_3857.geometry.area
        non_small_mask = areas_sqm > small_area_threshold

    if not non_small_mask.any():
        logger.debug(f"No non-small features (area > {small_area_threshold} sqm) found")
        return None

    # Step 1: Check if target point falls within any non-small feature
    gdf_3857_non_small = gdf_3857[non_small_mask]
    contains_mask = gdf_3857_non_small.geometry.contains(target_point_3857)

    if contains_mask.any():
        # Target point is within at least one non-small feature
        containing_features = gdf.loc[contains_mask[contains_mask].index]

        # Apply confidence filtering if available
        selected_idx = _select_best_by_confidence(
            containing_features, confidence_col, high_confidence_threshold
        )
        logger.debug(f"Target point contained within non-small feature (idx={selected_idx})")
        return gdf.loc[selected_idx]

    # Step 2: Check if nearest non-small feature is within tolerance
    distances = gdf_3857_non_small.geometry.distance(target_point_3857)

    within_tolerance_mask = distances <= distance_tolerance
    if within_tolerance_mask.any():
        # At least one non-small feature is within tolerance
        features_in_tolerance = gdf.loc[within_tolerance_mask[within_tolerance_mask].index]

        # Apply confidence filtering, then select nearest among candidates
        selected_idx = _select_best_by_confidence_and_distance(
            features_in_tolerance,
            distances.loc[features_in_tolerance.index],
            confidence_col,
            high_confidence_threshold,
        )
        logger.debug(
            f"Selected nearest non-small feature within {distance_tolerance}m tolerance "
            f"(idx={selected_idx}, distance={distances[selected_idx]:.2f}m)"
        )
        return gdf.loc[selected_idx]

    # Step 3: No suitable feature found by nearest method
    nearest_distance = distances.min() if len(distances) > 0 else float("inf")
    logger.debug(
        f"No non-small feature contains target or is within {distance_tolerance}m tolerance. "
        f"Nearest non-small feature is {nearest_distance:.2f}m away."
    )
    return None


def _select_best_by_confidence(
    gdf: gpd.GeoDataFrame,
    confidence_col: Optional[str],
    high_confidence_threshold: float,
) -> int:
    """Select the best feature index, preferring high-confidence features."""
    if confidence_col and confidence_col in gdf.columns:
        high_conf_mask = gdf[confidence_col] >= high_confidence_threshold
        if high_conf_mask.any():
            # Return first high-confidence feature
            return gdf[high_conf_mask].index[0]
    # Return first feature if no confidence filtering
    return gdf.index[0]


def _select_best_by_confidence_and_distance(
    gdf: gpd.GeoDataFrame,
    distances: pd.Series,
    confidence_col: Optional[str],
    high_confidence_threshold: float,
) -> int:
    """Select the best feature index, preferring high-confidence then nearest."""
    if confidence_col and confidence_col in gdf.columns:
        high_conf_mask = gdf[confidence_col] >= high_confidence_threshold
        if high_conf_mask.any():
            # Select nearest among high-confidence features
            high_conf_distances = distances[high_conf_mask]
            return high_conf_distances.idxmin()
    # Select nearest overall
    return distances.idxmin()


def select_primary_optimal(
    gdf: gpd.GeoDataFrame,
    target_lat: float,
    target_lon: float,
    area_col: str = "area",
    secondary_area_col: Optional[str] = None,
    confidence_col: Optional[str] = "confidence",
    high_confidence_threshold: float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    geometry_col: str = "geometry",
    small_area_threshold: float = BUILDING_SMALL_MAX_AREA_SQM,
    distance_tolerance: float = NEAREST_TOLERANCE_METERS,
) -> pd.Series:
    """
    Select the primary feature using "optimal" strategy: nearest first, then largest fallback.

    This method combines the nearest and largest selection strategies:
    1. First, try to select a feature using the "nearest" method (containment/proximity)
    2. If nearest returns None (no feature contains or is close to the target point),
       fall back to selecting the largest feature.

    This is the recommended method for building-like objects where we have a target
    geocode (lat/lon) but want to ensure we always return a primary feature.

    Args:
        gdf: GeoDataFrame containing features to select from (must be in EPSG:4326)
        target_lat: Target latitude (EPSG:4326)
        target_lon: Target longitude (EPSG:4326)
        area_col: Name of area column for size filtering and largest fallback
        secondary_area_col: Optional secondary area column for tie-breaking in largest
        confidence_col: Optional name of confidence column (0-1 scale). Set to None to disable.
        high_confidence_threshold: Threshold for "high confidence" (default: 0.9)
        geometry_col: Name of geometry column to use for containment/distance calculation
        small_area_threshold: Features at or below this area are considered "small"
        distance_tolerance: Maximum distance in meters for nearest selection

    Returns:
        pandas Series representing the selected primary feature (always returns a result)

    Raises:
        ValueError: If gdf is empty

    Example:
        >>> roofs_gdf = gpd.GeoDataFrame({...})  # Multiple roofs for a property
        >>> primary_roof = select_primary_optimal(
        ...     roofs_gdf, target_lat=37.7749, target_lon=-122.4194,
        ...     area_col="area", confidence_col="trustScore"
        ... )
    """
    if len(gdf) == 0:
        raise ValueError("Cannot select primary feature from empty GeoDataFrame")

    # Try nearest method first (containment/proximity with non-small filtering)
    result = select_primary_by_nearest(
        gdf,
        target_lat,
        target_lon,
        confidence_col=confidence_col,
        high_confidence_threshold=high_confidence_threshold,
        geometry_col=geometry_col,
        area_col=area_col,
        small_area_threshold=small_area_threshold,
        distance_tolerance=distance_tolerance,
    )

    if result is not None:
        return result

    # Fall back to largest method
    logger.debug("Nearest method returned no result, falling back to largest")
    return select_primary_by_largest(gdf, area_col, secondary_area_col)


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
    small_area_threshold: float = BUILDING_SMALL_MAX_AREA_SQM,
    distance_tolerance: float = NEAREST_TOLERANCE_METERS,
) -> Optional[pd.Series]:
    """
    Convenience function to select primary feature using specified method.

    This is the main entry point for primary feature selection. It dispatches
    to the appropriate selection algorithm based on the method parameter.

    Args:
        gdf: GeoDataFrame containing features to select from
        method: Selection method - "largest", "nearest", or "optimal"
        area_col: Area column name (for "largest" and size filtering in "nearest"/"optimal")
        secondary_area_col: Optional secondary area column (for "largest" method)
        target_lat: Target latitude (for "nearest" and "optimal" methods)
        target_lon: Target longitude (for "nearest" and "optimal" methods)
        confidence_col: Confidence column name (for "nearest" and "optimal" methods)
        high_confidence_threshold: High confidence threshold (for "nearest"/"optimal")
        geometry_col: Geometry column name (for "nearest" and "optimal" methods)
        small_area_threshold: Features at or below this area are considered "small"
        distance_tolerance: Maximum distance in meters for nearest selection

    Returns:
        pandas Series representing the selected primary feature.
        For "nearest" method, may return None if no suitable feature found.
        For "largest" and "optimal" methods, always returns a result.

    Raises:
        ValueError: If method is invalid or required parameters are missing

    Example:
        >>> # Select largest roof by clipped area
        >>> primary = select_primary(
        ...     roofs_gdf, method="largest",
        ...     area_col="clipped_area_sqm", secondary_area_col="unclipped_area_sqm"
        ... )
        >>>
        >>> # Select using optimal strategy (nearest with largest fallback)
        >>> primary = select_primary(
        ...     roof_instances_gdf, method="optimal",
        ...     target_lat=37.7749, target_lon=-122.4194,
        ...     area_col="area", confidence_col="trustScore"
        ... )
    """
    if len(gdf) == 0:
        raise ValueError("Cannot select primary feature from empty GeoDataFrame")

    if method == "largest" or method == "largest_intersection":
        return select_primary_by_largest(gdf, area_col, secondary_area_col)

    elif method == "nearest":
        if target_lat is None or target_lon is None or pd.isna(target_lat) or pd.isna(target_lon):
            logger.debug(
                "For 'nearest' method, lat/lon is required but was null - falling back to 'largest'"
            )
            return select_primary_by_largest(gdf, area_col, secondary_area_col)
        return select_primary_by_nearest(
            gdf,
            target_lat,
            target_lon,
            confidence_col,
            high_confidence_threshold,
            geometry_col,
            area_col,
            small_area_threshold,
            distance_tolerance,
        )

    elif method == "optimal":
        if target_lat is None or target_lon is None or pd.isna(target_lat) or pd.isna(target_lon):
            logger.debug(
                "For 'optimal' method, lat/lon is required but was null - falling back to 'largest'"
            )
            return select_primary_by_largest(gdf, area_col, secondary_area_col)
        return select_primary_optimal(
            gdf,
            target_lat,
            target_lon,
            area_col,
            secondary_area_col,
            confidence_col,
            high_confidence_threshold,
            geometry_col,
            small_area_threshold,
            distance_tolerance,
        )

    else:
        raise ValueError(
            f"Invalid selection method '{method}'. Must be 'largest', 'largest_intersection', 'nearest', or 'optimal'"
        )
