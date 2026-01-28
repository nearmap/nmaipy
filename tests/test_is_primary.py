"""
Unit tests for the _add_is_primary_column function in exporter.py.

Tests verify that primary features are correctly marked based on rollup data.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    API_CRS,
    BUILDING_ID,
    BUILDING_NEW_ID,
    ROOF_ID,
    ROOF_INSTANCE_CLASS_ID,
)
from nmaipy.exporter import _add_is_primary_column


class TestAddIsPrimaryColumn:
    """Tests for _add_is_primary_column function."""

    def test_basic_primary_marking(self):
        """Verify features matching rollup primary IDs get is_primary=True."""
        features_gdf = gpd.GeoDataFrame(
            {
                AOI_ID_COLUMN_NAME: ["aoi-1", "aoi-1", "aoi-2"],
                "feature_id": ["feat-a", "feat-b", "feat-c"],
                "class_id": [ROOF_ID, ROOF_ID, ROOF_ID],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            },
            crs=API_CRS,
        )

        rollup_df = pd.DataFrame(
            {
                "primary_roof_feature_id": ["feat-a", "feat-c"],
            },
            index=pd.Index(["aoi-1", "aoi-2"], name=AOI_ID_COLUMN_NAME),
        )

        result = _add_is_primary_column(features_gdf, rollup_df)

        assert "is_primary" in result.columns, "is_primary column should be added"
        # feat-a is primary for aoi-1
        assert result[result["feature_id"] == "feat-a"]["is_primary"].iloc[0] == True
        # feat-b is NOT primary for aoi-1
        assert result[result["feature_id"] == "feat-b"]["is_primary"].iloc[0] == False
        # feat-c is primary for aoi-2
        assert result[result["feature_id"] == "feat-c"]["is_primary"].iloc[0] == True

    def test_multiple_classes(self):
        """Verify each class type (Roof, Building, Roof Instance) marks correctly."""
        features_gdf = gpd.GeoDataFrame(
            {
                AOI_ID_COLUMN_NAME: ["aoi-1", "aoi-1", "aoi-1", "aoi-1"],
                "feature_id": ["roof-1", "bldg-1", "bldg-new-1", "instance-1"],
                "class_id": [ROOF_ID, BUILDING_ID, BUILDING_NEW_ID, ROOF_INSTANCE_CLASS_ID],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
            },
            crs=API_CRS,
        )

        rollup_df = pd.DataFrame(
            {
                "primary_roof_feature_id": ["roof-1"],
                "primary_building_feature_id": ["bldg-1"],
                "primary_building_(new_semantic)_feature_id": ["bldg-new-1"],
                "primary_roof_instance_feature_id": ["instance-1"],
            },
            index=pd.Index(["aoi-1"], name=AOI_ID_COLUMN_NAME),
        )

        result = _add_is_primary_column(features_gdf, rollup_df)

        # All features should be marked as primary
        assert result["is_primary"].all(), "All features should be primary for their class"

    def test_class_specific_matching(self):
        """Verify that same feature_id in different classes are handled correctly."""
        # Same feature_id but different class - only roof should be primary
        features_gdf = gpd.GeoDataFrame(
            {
                AOI_ID_COLUMN_NAME: ["aoi-1", "aoi-1"],
                "feature_id": ["shared-id", "shared-id"],
                "class_id": [ROOF_ID, BUILDING_ID],
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs=API_CRS,
        )

        rollup_df = pd.DataFrame(
            {
                "primary_roof_feature_id": ["shared-id"],
                # No primary building
            },
            index=pd.Index(["aoi-1"], name=AOI_ID_COLUMN_NAME),
        )

        result = _add_is_primary_column(features_gdf, rollup_df)

        roof_row = result[result["class_id"] == ROOF_ID].iloc[0]
        building_row = result[result["class_id"] == BUILDING_ID].iloc[0]

        assert roof_row["is_primary"] == True, "Roof with matching ID should be primary"
        assert building_row["is_primary"] == False, "Building with same ID should NOT be primary"

    def test_empty_rollup_none(self):
        """rollup_df=None returns all is_primary=False."""
        features_gdf = gpd.GeoDataFrame(
            {
                AOI_ID_COLUMN_NAME: ["aoi-1"],
                "feature_id": ["feat-a"],
                "class_id": [ROOF_ID],
                "geometry": [Point(0, 0)],
            },
            crs=API_CRS,
        )

        result = _add_is_primary_column(features_gdf, None)

        assert "is_primary" in result.columns
        assert result["is_primary"].iloc[0] == False

    def test_empty_rollup_dataframe(self):
        """Empty rollup DataFrame returns all is_primary=False."""
        features_gdf = gpd.GeoDataFrame(
            {
                AOI_ID_COLUMN_NAME: ["aoi-1"],
                "feature_id": ["feat-a"],
                "class_id": [ROOF_ID],
                "geometry": [Point(0, 0)],
            },
            crs=API_CRS,
        )

        rollup_df = pd.DataFrame()

        result = _add_is_primary_column(features_gdf, rollup_df)

        assert "is_primary" in result.columns
        assert result["is_primary"].iloc[0] == False

    def test_missing_feature_id_column(self):
        """Missing feature_id in features returns all is_primary=False."""
        features_gdf = gpd.GeoDataFrame(
            {
                AOI_ID_COLUMN_NAME: ["aoi-1"],
                "class_id": [ROOF_ID],
                "geometry": [Point(0, 0)],
            },
            crs=API_CRS,
        )

        rollup_df = pd.DataFrame(
            {"primary_roof_feature_id": ["feat-a"]},
            index=pd.Index(["aoi-1"], name=AOI_ID_COLUMN_NAME),
        )

        result = _add_is_primary_column(features_gdf, rollup_df)

        assert "is_primary" in result.columns
        assert result["is_primary"].iloc[0] == False

    def test_missing_class_id_column(self):
        """Missing class_id in features returns all is_primary=False."""
        features_gdf = gpd.GeoDataFrame(
            {
                AOI_ID_COLUMN_NAME: ["aoi-1"],
                "feature_id": ["feat-a"],
                "geometry": [Point(0, 0)],
            },
            crs=API_CRS,
        )

        rollup_df = pd.DataFrame(
            {"primary_roof_feature_id": ["feat-a"]},
            index=pd.Index(["aoi-1"], name=AOI_ID_COLUMN_NAME),
        )

        result = _add_is_primary_column(features_gdf, rollup_df)

        assert "is_primary" in result.columns
        assert result["is_primary"].iloc[0] == False

    def test_no_matching_primary_ids(self):
        """Rollup has primary IDs but none match features."""
        features_gdf = gpd.GeoDataFrame(
            {
                AOI_ID_COLUMN_NAME: ["aoi-1", "aoi-1"],
                "feature_id": ["feat-x", "feat-y"],
                "class_id": [ROOF_ID, ROOF_ID],
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs=API_CRS,
        )

        rollup_df = pd.DataFrame(
            {"primary_roof_feature_id": ["feat-a"]},  # Different ID
            index=pd.Index(["aoi-1"], name=AOI_ID_COLUMN_NAME),
        )

        result = _add_is_primary_column(features_gdf, rollup_df)

        assert not result["is_primary"].any(), "No features should be primary when IDs don't match"

    def test_preserves_index(self):
        """When aoi_id is index, it's preserved after operation."""
        features_gdf = gpd.GeoDataFrame(
            {
                "feature_id": ["feat-a", "feat-b"],
                "class_id": [ROOF_ID, ROOF_ID],
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs=API_CRS,
            index=pd.Index(["aoi-1", "aoi-2"], name=AOI_ID_COLUMN_NAME),
        )

        rollup_df = pd.DataFrame(
            {"primary_roof_feature_id": ["feat-a", "feat-b"]},
            index=pd.Index(["aoi-1", "aoi-2"], name=AOI_ID_COLUMN_NAME),
        )

        result = _add_is_primary_column(features_gdf, rollup_df)

        assert result.index.name == AOI_ID_COLUMN_NAME, "Index name should be preserved"
        assert list(result.index) == ["aoi-1", "aoi-2"], "Index values should be preserved"

    def test_preserves_geodataframe_and_crs(self):
        """Result is still a GeoDataFrame with same CRS."""
        features_gdf = gpd.GeoDataFrame(
            {
                AOI_ID_COLUMN_NAME: ["aoi-1"],
                "feature_id": ["feat-a"],
                "class_id": [ROOF_ID],
                "geometry": [Point(0, 0)],
            },
            crs=API_CRS,
        )

        rollup_df = pd.DataFrame(
            {"primary_roof_feature_id": ["feat-a"]},
            index=pd.Index(["aoi-1"], name=AOI_ID_COLUMN_NAME),
        )

        result = _add_is_primary_column(features_gdf, rollup_df)

        assert isinstance(result, gpd.GeoDataFrame), "Result should be GeoDataFrame"
        assert result.crs == API_CRS, "CRS should be preserved"

    def test_feature_id_string_conversion(self):
        """Integer feature_ids in GDF match string IDs in rollup."""
        # Features have integer feature_id
        features_gdf = gpd.GeoDataFrame(
            {
                AOI_ID_COLUMN_NAME: ["aoi-1", "aoi-1"],
                "feature_id": [12345, 67890],  # integers
                "class_id": [ROOF_ID, ROOF_ID],
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs=API_CRS,
        )

        # Rollup has string feature_id
        rollup_df = pd.DataFrame(
            {"primary_roof_feature_id": ["12345"]},  # string
            index=pd.Index(["aoi-1"], name=AOI_ID_COLUMN_NAME),
        )

        result = _add_is_primary_column(features_gdf, rollup_df)

        # Feature 12345 should match "12345" after string conversion
        assert result[result["feature_id"] == "12345"]["is_primary"].iloc[0] == True
        assert result[result["feature_id"] == "67890"]["is_primary"].iloc[0] == False

    def test_null_primary_ids_in_rollup(self):
        """Null/NaN values in rollup primary columns are handled gracefully."""
        features_gdf = gpd.GeoDataFrame(
            {
                AOI_ID_COLUMN_NAME: ["aoi-1", "aoi-2"],
                "feature_id": ["feat-a", "feat-b"],
                "class_id": [ROOF_ID, ROOF_ID],
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs=API_CRS,
        )

        rollup_df = pd.DataFrame(
            {"primary_roof_feature_id": ["feat-a", None]},  # aoi-2 has no primary
            index=pd.Index(["aoi-1", "aoi-2"], name=AOI_ID_COLUMN_NAME),
        )

        result = _add_is_primary_column(features_gdf, rollup_df)

        assert result[result[AOI_ID_COLUMN_NAME] == "aoi-1"]["is_primary"].iloc[0] == True
        assert result[result[AOI_ID_COLUMN_NAME] == "aoi-2"]["is_primary"].iloc[0] == False
