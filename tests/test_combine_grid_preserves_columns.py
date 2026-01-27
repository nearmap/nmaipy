"""
Test that combine_features_from_grid preserves include parameter columns.

This test verifies the fix for the bug where defensibleSpace and other
include parameter columns were being dropped during grid consolidation.
"""
import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from nmaipy.geometry_utils import combine_features_from_grid
from nmaipy.constants import API_CRS, AOI_ID_COLUMN_NAME


def test_combine_features_preserves_include_columns():
    """Test that include parameter columns are preserved during grid combination."""

    # Create test features with include parameter columns (defensibleSpace-like)
    test_features = gpd.GeoDataFrame({
        'feature_id': [1, 1, 2, 3],  # Feature 1 appears twice (split across grid cells)
        'geometry': [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),  # Adjacent to first
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ],
        AOI_ID_COLUMN_NAME: ['aoi1', 'aoi1', 'aoi1', 'aoi1'],
        'class_id': [13, 13, 13, 13],  # All roofs
        'area_sqm': [1.0, 1.0, 1.0, 1.0],
        'confidence': [0.9, 0.9, 0.8, 0.85],
        # Include parameter columns (defensibleSpace)
        'defensible_space_zone_0_coverage_ratio': [0.75, 0.75, 0.82, 0.90],
        'defensible_space_zone_0_defensible_space_area_sqft': [100, 100, 120, 150],
        'defensible_space_zone_1_coverage_ratio': [0.65, 0.65, 0.70, 0.80],
        # Include parameter columns (hurricaneScore)
        'hurricane_score': [7.5, 7.5, 8.2, 6.8],
        'wind_speed_mph': [130, 130, 140, 125],
        # Include parameter columns (roofSpotlightIndex)
        'rsi_score': [85, 85, 92, 78],
    }, crs=API_CRS)

    # Combine features
    result = combine_features_from_grid(test_features)

    # Verify basic structure
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 3  # Should have 3 unique features (1 was merged)
    assert result.crs == API_CRS

    # Verify include columns are preserved
    include_cols = [
        'defensible_space_zone_0_coverage_ratio',
        'defensible_space_zone_0_defensible_space_area_sqft',
        'defensible_space_zone_1_coverage_ratio',
        'hurricane_score',
        'wind_speed_mph',
        'rsi_score',
    ]

    for col in include_cols:
        assert col in result.columns, f"Column '{col}' should be preserved but was dropped"

    # Verify that values are present (took "first" value when merging)
    assert result['hurricane_score'].notna().all(), "hurricane_score should have values for all features"
    assert result['rsi_score'].notna().all(), "rsi_score should have values for all features"

    # Verify area columns were summed for merged features
    # Feature 1 appeared twice with area 1.0 each, should now be 2.0
    result_reset = result.reset_index()
    feature_1 = result_reset[result_reset['feature_id'] == 1]
    if len(feature_1) > 0:
        feature_1_area = feature_1['area_sqm'].iloc[0]
        assert feature_1_area == 2.0, f"Merged feature should have summed area of 2.0, got {feature_1_area}"


def test_combine_features_preserves_custom_columns():
    """Test that arbitrary custom columns are preserved."""

    # Create test features with custom columns
    test_features = gpd.GeoDataFrame({
        'feature_id': [1, 2],
        'geometry': [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        ],
        AOI_ID_COLUMN_NAME: ['aoi1', 'aoi1'],
        'class_id': [13, 13],
        'area_sqm': [1.0, 1.0],
        # Custom columns that weren't in the original hardcoded list
        'custom_field_1': ['value1', 'value2'],
        'custom_field_2': [42, 43],
        'new_metric_xyz': [1.5, 2.5],
    }, crs=API_CRS)

    # Combine features
    result = combine_features_from_grid(test_features)

    # Verify custom columns are preserved
    assert 'custom_field_1' in result.columns, "Custom column 'custom_field_1' should be preserved"
    assert 'custom_field_2' in result.columns, "Custom column 'custom_field_2' should be preserved"
    assert 'new_metric_xyz' in result.columns, "Custom column 'new_metric_xyz' should be preserved"


def test_combine_features_empty_input():
    """Test that empty input is handled correctly."""

    # Test with None
    result = combine_features_from_grid(None)
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0
    assert result.crs == API_CRS

    # Test with empty GeoDataFrame
    empty_gdf = gpd.GeoDataFrame(columns=['geometry', 'feature_id'], crs=API_CRS)
    result = combine_features_from_grid(empty_gdf)
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0
    assert result.crs == API_CRS


def test_combine_features_area_summing():
    """Test that area columns are still summed correctly after the fix."""

    # Create test features where one feature appears in multiple grid cells
    test_features = gpd.GeoDataFrame({
        'feature_id': [1, 1, 1, 2],  # Feature 1 split across 3 grid cells
        'geometry': [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
        ],
        AOI_ID_COLUMN_NAME: ['aoi1', 'aoi1', 'aoi1', 'aoi1'],
        'class_id': [13, 13, 13, 13],
        'area_sqm': [10.0, 15.0, 12.0, 20.0],
        'area_sqft': [100.0, 150.0, 120.0, 200.0],
        'clipped_area_sqm': [10.0, 15.0, 12.0, 20.0],
        'clipped_area_sqft': [100.0, 150.0, 120.0, 200.0],
        'confidence': [0.9, 0.9, 0.9, 0.8],
    }, crs=API_CRS)

    result = combine_features_from_grid(test_features)

    # Verify we have 2 features
    assert len(result) == 2

    # Verify area summing for feature 1
    result_reset = result.reset_index()
    feature_1 = result_reset[result_reset['feature_id'] == 1].iloc[0]

    assert feature_1['area_sqm'] == pytest.approx(37.0), "area_sqm should be summed: 10+15+12=37"
    assert feature_1['area_sqft'] == pytest.approx(370.0), "area_sqft should be summed: 100+150+120=370"
    assert feature_1['clipped_area_sqm'] == pytest.approx(37.0), "clipped_area_sqm should be summed"
    assert feature_1['clipped_area_sqft'] == pytest.approx(370.0), "clipped_area_sqft should be summed"

    # Verify non-area columns took first value
    assert feature_1['confidence'] == 0.9, "confidence should take first value, not be summed"
