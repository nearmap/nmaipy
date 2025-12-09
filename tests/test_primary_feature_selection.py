"""Tests for primary feature selection utilities."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from nmaipy.primary_feature_selection import (
    select_primary,
    select_primary_by_largest,
    select_primary_by_nearest,
    select_primary_optimal,
)


class TestSelectPrimaryByNearest:
    """Tests for the select_primary_by_nearest function."""

    def _create_test_gdf(self, features: list) -> gpd.GeoDataFrame:
        """Create a test GeoDataFrame from feature definitions."""
        gdf = gpd.GeoDataFrame(features, geometry="geometry")
        gdf = gdf.set_crs("EPSG:4326")
        return gdf

    def test_containment_selects_feature_containing_target(self):
        """When target is inside a non-small feature, that feature is selected."""
        # Feature that contains the target point
        containing_feature = Polygon([
            (-122.42, 37.77), (-122.42, 37.78),
            (-122.41, 37.78), (-122.41, 37.77), (-122.42, 37.77)
        ])
        # Feature that doesn't contain the target
        other_feature = Polygon([
            (-122.40, 37.77), (-122.40, 37.78),
            (-122.39, 37.78), (-122.39, 37.77), (-122.40, 37.77)
        ])

        gdf = self._create_test_gdf([
            {"id": 1, "area_sqm": 100, "confidence": 0.9, "geometry": containing_feature},
            {"id": 2, "area_sqm": 200, "confidence": 0.95, "geometry": other_feature},
        ])

        # Target inside containing_feature
        result = select_primary_by_nearest(
            gdf, target_lat=37.775, target_lon=-122.415,
            area_col="area_sqm", confidence_col="confidence"
        )

        assert result["id"] == 1, "Should select feature containing target point"

    def test_proximity_selects_nearest_within_tolerance(self):
        """When no feature contains target, nearest within tolerance is selected."""
        # Feature with target point exactly on its boundary (distance ~0m)
        # Target at 37.7750, -122.4200 - feature boundary touches this point
        feature_on_boundary = Polygon([
            (-122.4201, 37.7750), (-122.4201, 37.7760),
            (-122.4200, 37.7760), (-122.4200, 37.7750), (-122.4201, 37.7750)
        ])
        # Feature far away
        feature_far = Polygon([
            (-122.43, 37.78), (-122.43, 37.79),
            (-122.42, 37.79), (-122.42, 37.78), (-122.43, 37.78)
        ])

        gdf = self._create_test_gdf([
            {"id": 1, "area_sqm": 100, "confidence": 0.85, "geometry": feature_on_boundary},
            {"id": 2, "area_sqm": 200, "confidence": 0.95, "geometry": feature_far},
        ])

        # Target on the boundary of feature_on_boundary (distance ~0, within 1m tolerance)
        result = select_primary_by_nearest(
            gdf, target_lat=37.7750, target_lon=-122.4200,
            area_col="area_sqm", confidence_col="confidence"
        )

        assert result["id"] == 1, "Should select feature whose boundary touches target (within tolerance)"

    def test_returns_none_when_no_feature_within_tolerance(self):
        """When no feature contains or is within tolerance, returns None."""
        feature1 = Polygon([
            (-122.40, 37.70), (-122.40, 37.71),
            (-122.39, 37.71), (-122.39, 37.70), (-122.40, 37.70)
        ])
        feature2 = Polygon([
            (-122.38, 37.70), (-122.38, 37.71),
            (-122.37, 37.71), (-122.37, 37.70), (-122.38, 37.70)
        ])

        gdf = self._create_test_gdf([
            {"id": 1, "area_sqm": 100, "confidence": 0.9, "geometry": feature1},
            {"id": 2, "area_sqm": 200, "confidence": 0.95, "geometry": feature2},
        ])

        # Target far from both features
        result = select_primary_by_nearest(
            gdf, target_lat=37.80, target_lon=-122.50,
            area_col="area_sqm", confidence_col="confidence"
        )

        assert result is None, "Should return None when no feature within tolerance"

    def test_excludes_small_features(self):
        """Small features (<= 30sqm) are excluded from primary selection."""
        # Small feature that contains target
        small_containing = Polygon([
            (-122.42, 37.77), (-122.42, 37.78),
            (-122.41, 37.78), (-122.41, 37.77), (-122.42, 37.77)
        ])
        # Large feature that doesn't contain target
        large_outside = Polygon([
            (-122.40, 37.77), (-122.40, 37.78),
            (-122.39, 37.78), (-122.39, 37.77), (-122.40, 37.77)
        ])

        gdf = self._create_test_gdf([
            {"id": 1, "area_sqm": 25, "confidence": 0.9, "geometry": small_containing},  # Small
            {"id": 2, "area_sqm": 200, "confidence": 0.95, "geometry": large_outside},  # Large
        ])

        # Target inside small_containing, but it should be ignored
        result = select_primary_by_nearest(
            gdf, target_lat=37.775, target_lon=-122.415,
            area_col="area_sqm", confidence_col="confidence"
        )

        # Should return None since only non-small feature is far away
        assert result is None, "Small features should be excluded"

    def test_calculates_area_from_geometry_when_no_area_column(self):
        """When area column is missing, calculates area from geometry."""
        # Large feature - will be > 30sqm in EPSG:3857
        large_feature = Polygon([
            (-122.42, 37.77), (-122.42, 37.78),
            (-122.41, 37.78), (-122.41, 37.77), (-122.42, 37.77)
        ])

        gdf = self._create_test_gdf([
            {"id": 1, "confidence": 0.9, "geometry": large_feature},
        ])

        # No area_sqm column - should calculate from geometry
        result = select_primary_by_nearest(
            gdf, target_lat=37.775, target_lon=-122.415,
            area_col="area_sqm",  # Not present in gdf
            confidence_col="confidence"
        )

        assert result["id"] == 1, "Should calculate area from geometry and select feature"

    def test_prefers_high_confidence_among_containing_features(self):
        """Among multiple containing features, prefers high-confidence ones."""
        # Two overlapping features, both contain target
        feature1 = Polygon([
            (-122.42, 37.77), (-122.42, 37.78),
            (-122.41, 37.78), (-122.41, 37.77), (-122.42, 37.77)
        ])
        feature2 = Polygon([
            (-122.42, 37.77), (-122.42, 37.78),
            (-122.41, 37.78), (-122.41, 37.77), (-122.42, 37.77)
        ])

        gdf = self._create_test_gdf([
            {"id": 1, "area_sqm": 100, "confidence": 0.85, "geometry": feature1},
            {"id": 2, "area_sqm": 100, "confidence": 0.95, "geometry": feature2},
        ])

        result = select_primary_by_nearest(
            gdf, target_lat=37.775, target_lon=-122.415,
            area_col="area_sqm", confidence_col="confidence",
            high_confidence_threshold=0.9
        )

        assert result["id"] == 2, "Should prefer high-confidence feature"

    def test_works_with_clipped_area_column(self):
        """Test with Feature API column name (clipped_area_sqm)."""
        feature = Polygon([
            (-122.42, 37.77), (-122.42, 37.78),
            (-122.41, 37.78), (-122.41, 37.77), (-122.42, 37.77)
        ])

        gdf = self._create_test_gdf([
            {"id": 1, "clipped_area_sqm": 100, "confidence": 0.9, "geometry": feature},
        ])

        result = select_primary_by_nearest(
            gdf, target_lat=37.775, target_lon=-122.415,
            area_col="clipped_area_sqm", confidence_col="confidence"
        )

        assert result["id"] == 1, "Should work with clipped_area_sqm column"


class TestSelectPrimaryOptimal:
    """Tests for the select_primary_optimal function."""

    def _create_test_gdf(self, features: list) -> gpd.GeoDataFrame:
        """Create a test GeoDataFrame from feature definitions."""
        gdf = gpd.GeoDataFrame(features, geometry="geometry")
        gdf = gdf.set_crs("EPSG:4326")
        return gdf

    def test_uses_nearest_when_feature_contains_target(self):
        """Optimal uses nearest method when a feature contains the target."""
        containing = Polygon([
            (-122.42, 37.77), (-122.42, 37.78),
            (-122.41, 37.78), (-122.41, 37.77), (-122.42, 37.77)
        ])
        larger = Polygon([
            (-122.40, 37.76), (-122.40, 37.79),
            (-122.37, 37.79), (-122.37, 37.76), (-122.40, 37.76)
        ])

        gdf = self._create_test_gdf([
            {"id": 1, "area_sqm": 100, "confidence": 0.9, "geometry": containing},
            {"id": 2, "area_sqm": 500, "confidence": 0.95, "geometry": larger},  # Larger
        ])

        result = select_primary_optimal(
            gdf, target_lat=37.775, target_lon=-122.415,
            area_col="area_sqm", confidence_col="confidence"
        )

        assert result["id"] == 1, "Should select containing feature, not largest"

    def test_falls_back_to_largest_when_no_feature_within_tolerance(self):
        """Optimal falls back to largest when no feature contains or is near target."""
        small_feature = Polygon([
            (-122.40, 37.70), (-122.40, 37.71),
            (-122.39, 37.71), (-122.39, 37.70), (-122.40, 37.70)
        ])
        large_feature = Polygon([
            (-122.38, 37.70), (-122.38, 37.73),
            (-122.35, 37.73), (-122.35, 37.70), (-122.38, 37.70)
        ])

        gdf = self._create_test_gdf([
            {"id": 1, "area_sqm": 100, "confidence": 0.95, "geometry": small_feature},
            {"id": 2, "area_sqm": 500, "confidence": 0.9, "geometry": large_feature},
        ])

        # Target far from both
        result = select_primary_optimal(
            gdf, target_lat=37.80, target_lon=-122.50,
            area_col="area_sqm", confidence_col="confidence"
        )

        assert result["id"] == 2, "Should fall back to largest feature"

    def test_always_returns_result(self):
        """Optimal should always return a result (never None)."""
        feature = Polygon([
            (-122.40, 37.70), (-122.40, 37.71),
            (-122.39, 37.71), (-122.39, 37.70), (-122.40, 37.70)
        ])

        gdf = self._create_test_gdf([
            {"id": 1, "area_sqm": 100, "confidence": 0.9, "geometry": feature},
        ])

        # Target very far from feature
        result = select_primary_optimal(
            gdf, target_lat=40.0, target_lon=-120.0,
            area_col="area_sqm", confidence_col="confidence"
        )

        assert result is not None, "Optimal should always return a result"
        assert result["id"] == 1


class TestSelectPrimary:
    """Tests for the main select_primary dispatcher."""

    def _create_test_gdf(self, features: list) -> gpd.GeoDataFrame:
        """Create a test GeoDataFrame from feature definitions."""
        gdf = gpd.GeoDataFrame(features, geometry="geometry")
        gdf = gdf.set_crs("EPSG:4326")
        return gdf

    def test_largest_method(self):
        """Test 'largest' method selects by area."""
        feature1 = Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)])
        feature2 = Polygon([(-2, -2), (-2, 2), (2, 2), (2, -2), (-2, -2)])

        gdf = self._create_test_gdf([
            {"id": 1, "area_sqm": 100, "geometry": feature1},
            {"id": 2, "area_sqm": 400, "geometry": feature2},
        ])

        result = select_primary(gdf, method="largest", area_col="area_sqm")
        assert result["id"] == 2, "Should select largest by area"

    def test_optimal_falls_back_when_lat_lon_missing(self):
        """Test 'optimal' method falls back to 'largest' when lat/lon is missing."""
        feature1 = Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)])
        feature2 = Polygon([(-2, -2), (-2, 2), (2, 2), (2, -2), (-2, -2)])
        gdf = self._create_test_gdf([
            {"id": 1, "area_sqm": 100, "geometry": feature1},
            {"id": 2, "area_sqm": 400, "geometry": feature2},
        ])

        # Should fall back to largest when lat/lon is None
        result = select_primary(gdf, method="optimal", area_col="area_sqm")
        assert result["id"] == 2, "Should fall back to largest when lat/lon is missing"

    def test_nearest_falls_back_when_lat_lon_missing(self):
        """Test 'nearest' method falls back to 'largest' when lat/lon is missing."""
        feature1 = Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)])
        feature2 = Polygon([(-2, -2), (-2, 2), (2, 2), (2, -2), (-2, -2)])
        gdf = self._create_test_gdf([
            {"id": 1, "area_sqm": 100, "geometry": feature1},
            {"id": 2, "area_sqm": 400, "geometry": feature2},
        ])

        # Should fall back to largest when lat/lon is None
        result = select_primary(gdf, method="nearest", area_col="area_sqm")
        assert result["id"] == 2, "Should fall back to largest when lat/lon is missing"

    def test_invalid_method_raises_error(self):
        """Test invalid method raises ValueError."""
        feature = Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)])
        gdf = self._create_test_gdf([{"id": 1, "area_sqm": 100, "geometry": feature}])

        with pytest.raises(ValueError, match="Invalid selection method"):
            select_primary(gdf, method="invalid_method", area_col="area_sqm")
