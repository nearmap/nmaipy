"""
Test that combine_features_from_grid preserves include parameter columns.

This test verifies the fix for the bug where defensibleSpace and other
include parameter columns were being dropped during grid consolidation.

Also tests the conflict resolution logic where larger area portions win
when the same feature_id has different values across grid cells.
"""

from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS
from nmaipy.geometry_utils import combine_features_from_grid

DATA_DIR = Path(__file__).parent / "data"
# A bundled NJ parcel queried live with parcelMode off on a ~11 m grid (so its roof and building span
# several cells), plus the same parcel's ungridded response as ground truth.
GRID_CELLS_CSV = DATA_DIR / "test_grid_cells_area_semantics.csv"
UNGRIDDED_CSV = DATA_DIR / "test_grid_cells_area_semantics_ungridded.csv"


def test_combine_features_preserves_include_columns():
    """Test that include parameter columns are preserved during grid combination."""

    # Create test features with include parameter columns (defensibleSpace-like)
    test_features = gpd.GeoDataFrame(
        {
            "feature_id": [
                1,
                1,
                2,
                3,
            ],  # Feature 1 appears twice (split across grid cells)
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),  # Adjacent to first
                Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            ],
            AOI_ID_COLUMN_NAME: ["aoi1", "aoi1", "aoi1", "aoi1"],
            "class_id": [13, 13, 13, 13],  # All roofs
            "area_sqm": [1.0, 1.0, 1.0, 1.0],
            # Feature 1 is a connected patch split across two cells: each cell's area is its clipped piece
            "clipped_area_sqm": [1.0, 1.0, 1.0, 1.0],
            "confidence": [0.9, 0.9, 0.8, 0.85],
            # Include parameter columns (defensibleSpace)
            "defensible_space_zone_0_coverage_ratio": [0.75, 0.75, 0.82, 0.90],
            "defensible_space_zone_0_defensible_space_area_sqft": [100, 100, 120, 150],
            "defensible_space_zone_1_coverage_ratio": [0.65, 0.65, 0.70, 0.80],
            # Include parameter columns (hurricaneScore)
            "hurricane_score": [7.5, 7.5, 8.2, 6.8],
            "wind_speed_mph": [130, 130, 140, 125],
            # Include parameter columns (roofSpotlightIndex)
            "rsi_score": [85, 85, 92, 78],
        },
        crs=API_CRS,
    )

    # Combine features
    result = combine_features_from_grid(test_features)

    # Verify basic structure
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 3  # Should have 3 unique features (1 was merged)
    assert result.crs == API_CRS

    # Verify include columns are preserved
    include_cols = [
        "defensible_space_zone_0_coverage_ratio",
        "defensible_space_zone_0_defensible_space_area_sqft",
        "defensible_space_zone_1_coverage_ratio",
        "hurricane_score",
        "wind_speed_mph",
        "rsi_score",
    ]

    for col in include_cols:
        assert col in result.columns, f"Column '{col}' should be preserved but was dropped"

    # Verify that values are present (took "first" value when merging)
    assert result["hurricane_score"].notna().all(), "hurricane_score should have values for all features"
    assert result["rsi_score"].notna().all(), "rsi_score should have values for all features"

    # Verify area columns were summed for merged features
    # Feature 1 appeared twice with area 1.0 each, should now be 2.0
    result_reset = result.reset_index()
    feature_1 = result_reset[result_reset["feature_id"] == 1]
    if len(feature_1) > 0:
        feature_1_area = feature_1["area_sqm"].iloc[0]
        assert feature_1_area == 2.0, f"Merged feature should have summed area of 2.0, got {feature_1_area}"


def test_combine_features_preserves_custom_columns():
    """Test that arbitrary custom columns are preserved."""

    # Create test features with custom columns
    test_features = gpd.GeoDataFrame(
        {
            "feature_id": [1, 2],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            AOI_ID_COLUMN_NAME: ["aoi1", "aoi1"],
            "class_id": [13, 13],
            "area_sqm": [1.0, 1.0],
            "clipped_area_sqm": [1.0, 1.0],
            # Custom columns that weren't in the original hardcoded list
            "custom_field_1": ["value1", "value2"],
            "custom_field_2": [42, 43],
            "new_metric_xyz": [1.5, 2.5],
        },
        crs=API_CRS,
    )

    # Combine features
    result = combine_features_from_grid(test_features)

    # Verify custom columns are preserved
    assert "custom_field_1" in result.columns, "Custom column 'custom_field_1' should be preserved"
    assert "custom_field_2" in result.columns, "Custom column 'custom_field_2' should be preserved"
    assert "new_metric_xyz" in result.columns, "Custom column 'new_metric_xyz' should be preserved"


def test_combine_features_empty_input():
    """Test that empty input is handled correctly."""

    # Test with None
    result = combine_features_from_grid(None)
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0
    assert result.crs == API_CRS

    # Test with empty GeoDataFrame
    empty_gdf = gpd.GeoDataFrame(columns=["geometry", "feature_id"], crs=API_CRS)
    result = combine_features_from_grid(empty_gdf)
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0
    assert result.crs == API_CRS


def test_combine_features_area_summing():
    """Connected-feature pieces split across grid cells have their areas summed."""

    # Create test features where one feature appears in multiple grid cells
    test_features = gpd.GeoDataFrame(
        {
            "feature_id": [1, 1, 1, 2],  # Feature 1 split across 3 grid cells
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
            ],
            AOI_ID_COLUMN_NAME: ["aoi1", "aoi1", "aoi1", "aoi1"],
            "class_id": [13, 13, 13, 13],
            "area_sqm": [10.0, 15.0, 12.0, 20.0],
            "area_sqft": [100.0, 150.0, 120.0, 200.0],
            # Connected feature: each cell reports only its piece, smaller than the whole patch
            "unclipped_area_sqm": [60.0, 60.0, 60.0, 20.0],
            "unclipped_area_sqft": [600.0, 600.0, 600.0, 200.0],
            "clipped_area_sqm": [10.0, 15.0, 12.0, 20.0],
            "clipped_area_sqft": [100.0, 150.0, 120.0, 200.0],
            "confidence": [0.9, 0.9, 0.9, 0.8],
        },
        crs=API_CRS,
    )

    result = combine_features_from_grid(test_features)

    # Verify we have 2 features
    assert len(result) == 2

    # Verify area summing for feature 1
    result_reset = result.reset_index()
    feature_1 = result_reset[result_reset["feature_id"] == 1].iloc[0]

    assert feature_1["area_sqm"] == pytest.approx(37.0), "area_sqm should be summed: 10+15+12=37"
    assert feature_1["area_sqft"] == pytest.approx(370.0), "area_sqft should be summed: 100+150+120=370"
    assert feature_1["clipped_area_sqm"] == pytest.approx(37.0), "clipped_area_sqm should be summed"
    assert feature_1["clipped_area_sqft"] == pytest.approx(370.0), "clipped_area_sqft should be summed"

    # Verify non-area columns took first value
    assert feature_1["confidence"] == 0.9, "confidence should take first value, not be summed"


def test_larger_clipped_area_wins_conflict():
    """Test that when same feature_id has different values, larger area portion wins."""

    # Feature 1 appears in 2 grid cells with different include param values
    # The cell with larger clipped_area (100) should win
    test_features = gpd.GeoDataFrame(
        {
            "feature_id": [1, 1, 2],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
            ],
            AOI_ID_COLUMN_NAME: ["aoi1", "aoi1", "aoi1"],
            "class_id": [13, 13, 13],
            "clipped_area_sqm": [
                100.0,
                50.0,
                75.0,
            ],  # First chunk is larger for feature 1
            "area_sqm": [100.0, 50.0, 75.0],
            # Conflicting values for feature 1: different hurricane_score
            "hurricane_score": [8.5, 7.0, 6.5],
            "rsi_score": [90, 85, 80],
        },
        crs=API_CRS,
    )

    result = combine_features_from_grid(test_features)

    # Feature 1 should have values from larger chunk (100 sqm)
    result_reset = result.reset_index()
    feature_1 = result_reset[result_reset["feature_id"] == 1].iloc[0]

    assert feature_1["hurricane_score"] == 8.5, "Should use value from larger clipped_area chunk"
    assert feature_1["rsi_score"] == 90, "Should use value from larger clipped_area chunk"

    # Area should be summed
    assert feature_1["clipped_area_sqm"] == pytest.approx(150.0), "Areas should be summed: 100 + 50"


def test_conflict_warning_logged():
    """Test that warning is logged when conflicting values detected for same feature_id."""

    test_features = gpd.GeoDataFrame(
        {
            "feature_id": [1, 1],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            AOI_ID_COLUMN_NAME: ["aoi1", "aoi1"],
            "class_id": [13, 13],
            "clipped_area_sqm": [100.0, 50.0],
            "area_sqm": [100.0, 50.0],
            "hurricane_score": [8.5, 7.0],  # Conflict!
        },
        crs=API_CRS,
    )

    with patch("nmaipy.geometry_utils.logger") as mock_logger:
        combine_features_from_grid(test_features)
        # Should have logged a warning about conflicting values
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "different attribute values" in warning_msg, "Warning should explain the conflict"
        assert "1" in warning_msg, "Warning should mention the feature_id with conflict"


def test_no_warning_when_values_identical():
    """Test that no warning is logged when same feature_id has identical values."""

    test_features = gpd.GeoDataFrame(
        {
            "feature_id": [1, 1],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            AOI_ID_COLUMN_NAME: ["aoi1", "aoi1"],
            "class_id": [13, 13],
            "clipped_area_sqm": [100.0, 50.0],
            "area_sqm": [100.0, 50.0],
            # Same values for both instances - no conflict
            "hurricane_score": [8.5, 8.5],
            "rsi_score": [90, 90],
        },
        crs=API_CRS,
    )

    with patch("nmaipy.geometry_utils.logger") as mock_logger:
        combine_features_from_grid(test_features)
        # Should NOT have logged any warnings since values are identical
        mock_logger.warning.assert_not_called()


def test_merged_grid_areas_match_ungridded_response():
    """Merging a real gridded response reproduces each feature's ungridded area.

    Discrete classes (roof, building) repeat their whole-object area in every cell they touch, so
    summing them multiplied the area by the number of cells. Connected classes (vegetation) report
    only the in-cell piece, so their pieces must still be summed.
    """
    cells = pd.read_csv(GRID_CELLS_CSV)
    cells = gpd.GeoDataFrame(cells, geometry=gpd.GeoSeries.from_wkt(cells["geometry"]), crs=API_CRS)
    # Same shape get_features_gdf_gridded hands to the merge: aoi_id as both index and column.
    cells = cells.set_index(AOI_ID_COLUMN_NAME, drop=False)
    expected = pd.read_csv(UNGRIDDED_CSV).set_index("feature_id")
    cells_per_feature = cells.groupby("feature_id").size()

    roof_cells = cells_per_feature[cells.loc[cells["description"] == "Roof", "feature_id"].iloc[0]]
    assert roof_cells > 1, "Fixture must split the roof across several grid cells to exercise the merge"

    merged = combine_features_from_grid(cells).set_index("feature_id")

    assert set(merged.index) == set(expected.index), "Merge should yield exactly the ungridded feature set"
    for feature_id, row in expected.iterrows():
        got = merged.loc[feature_id]
        # Each cell rounds its piece to 0.1 sqm / 1 sqft, so allow that much per cell.
        n = cells_per_feature[feature_id]
        assert got["area_sqm"] == pytest.approx(
            row["area_sqm"], abs=0.1 * n
        ), f"{row['description']} {feature_id}: merged area_sqm {got['area_sqm']} != ungridded {row['area_sqm']}"
        assert got["area_sqft"] == pytest.approx(
            row["area_sqft"], abs=1.0 * n
        ), f"{row['description']} {feature_id}: merged area_sqft {got['area_sqft']} != ungridded {row['area_sqft']}"
        assert got["clipped_area_sqm"] == pytest.approx(row["clipped_area_sqm"], abs=0.1 * n)
        assert got["unclipped_area_sqm"] == row["unclipped_area_sqm"]


def test_merge_requires_clipped_area_alongside_area():
    """Without clipped_area_* the merge cannot tell discrete from connected features, so it refuses."""
    features = gpd.GeoDataFrame(
        {
            "feature_id": [1, 1],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])],
            AOI_ID_COLUMN_NAME: ["aoi1", "aoi1"],
            "area_sqm": [2.0, 2.0],
        },
        crs=API_CRS,
    )
    with pytest.raises(ValueError, match="clipped_area_sqm"):
        combine_features_from_grid(features)


def test_discrete_area_kept_once_whatever_unclipped_holds():
    """A discrete object repeats its whole area in every cell it touches, so the merge keeps one copy.

    The kind comes from area versus clipped area alone, so a null unclipped_area_* changes nothing, and sqft
    follows the decision made on sqm.
    """
    whole_roof = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    features = gpd.GeoDataFrame(
        {
            "feature_id": ["roof", "roof"],
            "geometry": [whole_roof, whole_roof],
            AOI_ID_COLUMN_NAME: ["aoi1", "aoi1"],
            "area_sqm": [100.0, 100.0],
            "area_sqft": [1076.0, 1076.0],
            "clipped_area_sqm": [60.0, 40.0],
            "clipped_area_sqft": [646.0, 430.0],
            "unclipped_area_sqm": [None, None],
        },
        crs=API_CRS,
    )
    merged = combine_features_from_grid(features).set_index("feature_id").loc["roof"]
    assert merged["area_sqm"] == 100.0, "A discrete object's area is its whole area, kept once"
    assert merged["area_sqft"] == 1076.0, "area_sqft should follow the decision made on sqm"
    assert merged["clipped_area_sqm"] == 100.0, "Clipped pieces should be summed"


def test_merge_refuses_piece_kept_as_whole_object():
    """Pieces whose area differs from their clip in one cell would be treated as discrete and keep only the largest.

    The merged area would then be below the summed clipped area, which no whole object can be, so the merge refuses
    rather than undercount.
    """
    features = gpd.GeoDataFrame(
        {
            "feature_id": [1, 1],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])],
            AOI_ID_COLUMN_NAME: ["aoi1", "aoi1"],
            "area_sqm": [3.0, 2.5],
            "clipped_area_sqm": [3.0, 2.0],
        },
        crs=API_CRS,
    )
    with pytest.raises(ValueError, match="below their summed clipped_area_sqm"):
        combine_features_from_grid(features)
